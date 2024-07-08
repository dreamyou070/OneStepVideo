import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, List, Callable, Dict
from torchvision.utils import save_image
from einops import rearrange, repeat
import inspect
from typing import Any, Dict, Optional
from diffusers.models.transformers.transformer_temporal import TransformerTemporalModelOutput
class AttentionBase:

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def after_step(self):
        pass

    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # after step
            self.after_step()
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):

        # in forward, final attentio is
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


class AttentionStore(AttentionBase):
    def __init__(self, res=[32], min_step=0, max_step=1000):
        super().__init__()
        self.res = res
        self.min_step = min_step
        self.max_step = max_step
        self.valid_steps = 0

        self.self_attns = []  # store the all attns
        self.cross_attns = []

        self.self_attns_step = []  # store the attns in each step
        self.cross_attns_step = []

    def after_step(self):
        if self.cur_step > self.min_step and self.cur_step < self.max_step:
            self.valid_steps += 1
            if len(self.self_attns) == 0:
                self.self_attns = self.self_attns_step
                self.cross_attns = self.cross_attns_step
            else:
                for i in range(len(self.self_attns)):
                    self.self_attns[i] += self.self_attns_step[i]
                    self.cross_attns[i] += self.cross_attns_step[i]
        self.self_attns_step.clear()
        self.cross_attns_step.clear()

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        if attn.shape[1] <= 64 ** 2:  # avoid OOM
            if is_cross:
                self.cross_attns_step.append(attn)
            else:
                self.self_attns_step.append(attn)
        return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)


def regiter_attention_editor_diffusers(unet, editor: AttentionBase):
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):

        def forward(
                hidden_states: torch.Tensor,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                **cross_attention_kwargs,
        ) -> torch.Tensor:
            r"""
            The forward method of the `Attention` class.

            Args:
                hidden_states (`torch.Tensor`):
                    The hidden states of the query.
                encoder_hidden_states (`torch.Tensor`, *optional*):
                    The hidden states of the encoder.
                attention_mask (`torch.Tensor`, *optional*):
                    The attention mask to use. If `None`, no mask is applied.
                **cross_attention_kwargs:
                    Additional keyword arguments to pass along to the cross attention.

            Returns:
                `torch.Tensor`: The output of the attention layer.
            """
            # The `Attention` class can call different attention processors / attention functions
            # here we simply pass along all tensors to the selected processor class
            # For standard processors that are defined here, `**cross_attention_kwargs` is empty

            attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
            unused_kwargs = [k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters]
            cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}

            frame_num = hidden_states.shape[0]

            attn = self
            residual = hidden_states
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)
            input_ndim = hidden_states.ndim
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = attn.to_q(hidden_states)
            is_cross_attention = True
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
                is_cross_attention = False
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            key   = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) # batch, attn_heads, pix_num, head_dim
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if not is_cross_attention: # just self attention

                L, S = query.size(-2), key.size(-2)
                scale_factor = 1 / math.sqrt(query.size(-1))
                attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
                attn_weight = query @ key.transpose(-2, -1) * scale_factor
                attn_weight += attn_bias
                attn_weight = torch.softmax(attn_weight, dim=-1)
                hidden_states = attn_weight @ value

            else :
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor


            return hidden_states

        return forward


    def register_editor(net, count, place_in_unet, net_name):
        for name, subnet in net.named_children():
            final_name = f"{net_name}_{name}"
            #print(f'final_name = {final_name} : subnet.__class__.__name__: {subnet.__class__.__name__}')
            if subnet.__class__.__name__ == 'Attention' and 'motion' not in final_name.lower():  # spatial Transformer layer
                print(f' self or cross attn control ')
                subnet.forward = ca_forward(subnet, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet, final_name)
        return count

    cross_att_count = 0
    for net_name, net in unet.named_children():
        if "down" in net_name:
            cross_att_count += register_editor(net, 0, "down", net_name)
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid", net_name)
        elif "up" in net_name:
            cross_att_count += register_editor(net, 0, "up", net_name)
    editor.num_att_layers = cross_att_count


def regiter_motion_attention_editor_diffusers(unet, editor: AttentionBase):
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """
    def motion_forward(self, full_name):

        def forward(
                hidden_states: torch.Tensor,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                **cross_attention_kwargs,) -> torch.Tensor:

            attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
            quiet_attn_parameters = {"ip_adapter_masks"}
            unused_kwargs = [k for k, _ in cross_attention_kwargs.items() if
                             k not in attn_parameters and k not in quiet_attn_parameters]
            cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}
            attn = self
            residual = hidden_states
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)
            input_ndim = hidden_states.ndim
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            batch_size, sequence_length, _ = (hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)

            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = attn.to_q(hidden_states)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # ------------------------------------------------------------------------------------------------------- #
            # make attention map
            dropout_p = 0.0
            L, S = query.size(-2), key.size(-2)
            scale_factor = 1 / math.sqrt(query.size(-1))
            attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)
            attn_weight = query @ key.transpose(-2, -1) * scale_factor
            attn_weight += attn_bias
            # attention map
            attn_weight = torch.softmax(attn_weight, dim=-1) # [pixel_num, head, frame_num, frame_num]
            # 8192 = 2 * 4096 = 2 * (64 * 64)
            if editor.do_attention_map_check :
                if editor.guidance_scale > 1 :
                    uncon_attn_weight, attn_weight = attn_weight.chunk(2)
                attn_weight = attn_weight.mean(dim=(0,1))
                print(f'attn_weight.shape = {attn_weight.shape}')
                editor.save_attention_map(attn_weight, full_name)




            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
            hidden_states = attn_weight @ value

            # ------------------------------------------------------------------------------------------------------- #
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor
            return hidden_states


        return forward

    def motion_forward_basic(self, layer_name):


        def forward(
                hidden_states: torch.Tensor,
                encoder_hidden_states: Optional[torch.LongTensor] = None,
                timestep: Optional[torch.LongTensor] = None,
                class_labels: torch.LongTensor = None,
                num_frames: int = 1,
                cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                return_dict: bool = True,
        ) -> TransformerTemporalModelOutput:

            input = hidden_states

            do_skip = False
            do_save = True
            for skip_layer in editor.skip_layers:
                if skip_layer in layer_name.lower():
                    do_skip = True
                    do_save = False

            if do_skip and not editor.is_teacher and not editor.is_eval:
                # student save this one (only at training state)
                editor.save_hidden_states(hidden_states=input,
                                          layer_name=layer_name)
                return TransformerTemporalModelOutput(sample=hidden_states)

            else :
                batch_frames, channel, height, width = hidden_states.shape
                batch_size = batch_frames // num_frames
                residual = hidden_states
                # ------------------------------------------------------------------------------------------------------------------------
                # Here Reshaping !!
                # hidden_states = [1, batch*frame, dim, height, width]
                hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, channel, height, width)
                # hidden_states = [batch, frame, dim, height, width]

                hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
                # hidden_states = [batch, dim, frame, height, width]

                hidden_states = self.norm(hidden_states)
                hidden_states = hidden_states.permute(0, 3, 4, 2, 1).reshape(batch_size * height * width, num_frames,
                                                                             channel)
                # hidden_states = [batch*height*width, frame, dim]

                hidden_states = self.proj_in(hidden_states)

                # len of self.transformer_blocks = 1
                # 2. Blocks
                for block in self.transformer_blocks:
                    hidden_states = block(hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        timestep=timestep,
                        cross_attention_kwargs=cross_attention_kwargs,
                        class_labels=class_labels,)

                # 3. Output
                hidden_states = self.proj_out(hidden_states)
                hidden_states = (
                    hidden_states[None, None, :]
                    .reshape(batch_size, height, width, num_frames, channel)
                    .permute(0, 3, 4, 1, 2)
                    .contiguous())
                hidden_states = hidden_states.reshape(batch_frames, channel, height, width)

                output = hidden_states + residual
                if do_skip and editor.is_teacher and not editor.is_eval :
                    # only training state
                    #
                    is_train = not editor.is_eval
                    # ' Teacher Saving Time !'
                    editor.save_hidden_states(hidden_states=output, layer_name=layer_name)

                if not return_dict:
                    return (output,)



                return TransformerTemporalModelOutput(sample=output)

        return forward
    def register_editor(net, count, place_in_unet, net_name):
        for name, subnet in net.named_children():
            final_name = f"{net_name}_{name}"
            # start from "TransformerTemporalModel"
            if subnet.__class__.__name__ == 'TransformerTemporalModel' and 'motion' in final_name.lower():
                # [2] BasicTransformerBlock
                subnet.forward = motion_forward_basic(subnet, final_name)
            if subnet.__class__.__name__ == 'Attention' and 'motion' in final_name.lower():  # spatial Transformer layer
                subnet.forward = motion_forward(subnet, final_name)
                # [3] pos embed
                #original_pos_embed = subnet.pos_embed
                #original_pe = original_pos_embed.pe
                #_, max_seq_len, inner_dim = original_pe.shape
                #subnet.pos_embed  = SinusoidalPositionalEmbedding_custom(inner_dim,
                #                                                         total_frame_num=editor.total_frame_num,
                                                                         # max_seq_length=editor.total_frame_num,
                                                                         # max_seq_length = 32,
                                                                         # frame_num = editor.frame_num,
                #                                                         window_size =  editor.window_size)

            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet, final_name)
        return count

    cross_att_count = 0
    for net_name, net in unet.named_children():
        if "down" in net_name:
            cross_att_count += register_editor(net, 0, "down", net_name)
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid", net_name)
        elif "up" in net_name:
            cross_att_count += register_editor(net, 0, "up", net_name)
    editor.num_att_layers = cross_att_count

import math
class SinusoidalPositionalEmbedding_custom(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 total_frame_num: int = 32,
                 max_seq_length: int = 32,
                 #frame_num: int = 16,
                 window_size : int = 5,):

        super().__init__()

        self.standard_length = 16
        self.max_seq_length = self.standard_length + window_size
        position = torch.arange(self.max_seq_length).unsqueeze(1)
        #position = torch.linspace(0, self.standard_length,
        #                          total_frame_num).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, self.max_seq_length, embed_dim)
        standard = position * div_term
        pe[0, :, 0::2] = torch.sin(position * div_term) # interpolate ?
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        self.window_size = window_size
        self.total_frame_num = total_frame_num

    def forward(self, x):
        _, seq_length, _ = x.shape
        dtype = x.dtype
        device = x.device
        if seq_length == self.standard_length :
            if self.total_frame_num == self.standard_length :
                start_idx = 0
                end_idx = 0 + seq_length
            else :
                start_idx = 1
                end_idx = 1 + seq_length
                #start_idx = self.window_size
                #end_idx = start_idx + seq_length
                #print(f' (1) start_idx = {start_idx} : end_idx = {end_idx}')
            x = x + self.pe[:, start_idx:end_idx, :].to(device=x.device, dtype=x.dtype)
        elif seq_length > self.standard_length :
            start_idx = 0
            end_idx = seq_length
            #print(f' (2) start_idx = {start_idx} : end_idx = {end_idx}')
            window_x = x[:, : self.window_size, :] + self.pe[:, 0, :].unsqueeze(1).repeat(1, self.window_size, 1)
            remain_x = x[:, self.window_size:, :] + self.pe[:, 1 : 1 + self.standard_length, :]
            x = torch.cat([window_x, remain_x], dim=1)
        elif seq_length < self.standard_length :
            window_x = x[:, : self.window_size, :] + self.pe[:, 0, :].unsqueeze(1).repeat(1, self.window_size, 1)
            remain_length = seq_length - self.window_size
            remain_x = x[:, self.window_size:, :] + self.pe[:, 1 : 1 + remain_length, :]
            x = torch.cat([window_x, remain_x], dim=1)

            #end_idx = self.standard_length + self.window_size
            #start_idx = end_idx - seq_length
            #print(f' (3) start_idx = {start_idx} : end_idx = {end_idx}')
        x = x.to(device=device, dtype=dtype)
        return x