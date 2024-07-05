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
                **cross_attention_kwargs,
        ) -> torch.Tensor:

            attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
            unused_kwargs = [k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters]
            cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}
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

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1

            ############################################################################################################
            import math
            dropout_p = 0.0
            scale = None
            L, S = query.size(-2), key.size(-2)
            scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
            attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)
            # -------------------------------------------------------------------------------------- #
            # [1]
            if editor.full_attention:
                attn_weight = query @ key.transpose(-2, -1) * scale_factor
                attn_weight += attn_bias
                attn_weight = torch.softmax(attn_weight, dim=-1) # pixel_num, head, frame_num, frame_num

                attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
                hidden_states = attn_weight @ value

            elif editor.window_attention:
                weights = []
                attn_maps = []
                query_frame_num = query.shape[2]
                for frame_index in range(query_frame_num):
                    inner_query = query[:, :, frame_index, :].unsqueeze(2)
                    #print(f' [0] inner_query (pixel_nun, head, 1, dim) = {inner_query.shape}')
                    res = int(inner_query.shape[0] ** 0.5)
                    # --------------------------------------------------------------------------------------------------------------------
                    window_size = editor.window_size
                    left_window = int(window_size / 2)
                    key_start = max(0, frame_index - left_window)
                    key_end = key_start + window_size #min(frame_index + window_size + 1, query.shape[2])
                    if key_end > query.shape[2]:
                        key_end = query.shape[2]
                        key_start = key_end - window_size
                    inner_key = key[:, :, key_start:key_end, :]

                    # --------------------------------------------------------------------------------------------------------------------
                    inner_attn_weight = inner_query @ inner_key.transpose(-2, -1) * scale_factor  # [1,key_len]
                    inner_attn_weight = torch.softmax(inner_attn_weight, dim=-1).to(query.dtype)   # [pixel_num, head, 1, neighbor_frame=5]
                    attn_maps.append(inner_attn_weight) # pixel_num, head, frame, frame
                    # inner_value should be same
                    inner_value = value[:, :, key_start:key_end, :]
                    #inner_value = value[:, :, frame_index, :].unsqueeze(2)
                    inner_hidden_states = inner_attn_weight @ inner_value # [pixel_num, head, 1, dim]
                    weights.append(inner_hidden_states)
                hidden_states = torch.stack(weights, dim=2) # [pixel_num, head, frame_num, dim]
                # ------------------------------------------------------------------------------------------- #
                attn_weight = torch.cat(attn_maps, dim=2) # frame, neighbors

            ############################################################################################################
            # save attention map !
            # attn_weight = [pixel_num, head, frame_num, frame_num]
            editor.save_attention_map(attn_weight, full_name)

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
            """

            attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
            unused_kwargs = [k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters]
            cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}
            #print(f'self.processor = {self.processor.__class__.__name__}')

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

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            # --------------------------------------------------------------------------------------------------------- #
            # Change Here !
            import math
            dropout_p = 0.0
            scale = None
            L, S = query.size(-2), key.size(-2)
            scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
            inner_dtype = torch.float32
            attn_bias = torch.zeros(L, S, dtype=inner_dtype).to(query.device)
            # -------------------------------------------------------------------------------------- #
            # [1]
            if editor.full_attention:
                attn_weight = query @ key.transpose(-2, -1) * scale_factor
                attn_weight += attn_bias
                attn_weight = torch.softmax(attn_weight, dim=-1)
                attn_weight = attn_weight.to(query.dtype)
                attn_weight = torch.dropout(attn_weight, dropout_p,
                                            train=True)  # pixel_num*2, num_heads, frame_num, frame_num
                hidden_states = attn_weight @ value
            elif editor.window_attention:
                attn_weight = query @ key.transpose(-2, -1) * scale_factor
                attn_weight += attn_bias
                attn_weight = torch.softmax(attn_weight, dim=-1)
                attn_weight = attn_weight.to(query.dtype)
                attn_weight = torch.dropout(attn_weight, dropout_p,
                                            train=True)  # pixel_num*2, num_heads, frame_num, frame_num
                hidden_states = attn_weight @ value
                
                weights = []
                query_frame_num = query.shape[2]
                for frame_index in range(query_frame_num):

                    inner_query = query[:,:,frame_index,:].unsqueeze(2)

                    window_size = editor.window_size
                    key_start = max(0, frame_index - window_size)
                    key_end =   min(frame_index + window_size + 1, query.shape[2])
                    inner_key = key[:, :, key_start:key_end, :]
                    if inner_query.ndim != 4:
                        inner_query = inner_query.unsqueeze(2)
                    if inner_key.ndim != 4:
                        inner_key = inner_key.unsqueeze(2)
                    inner_attn_weight = inner_query @ inner_key.transpose(-2, -1) * scale_factor  # [1,key_len]
                    inner_attn_weight = torch.softmax(inner_attn_weight, dim=-1).to(query.dtype)
                    # inner_value should be same
                    inner_value = value[:, :, key_start:key_end, :]
                    # inner_value = value[:, :, :key_end, :]
                    inner_hidden_states = inner_attn_weight @ inner_value
                    weights.append(inner_hidden_states)
                hidden_states = torch.stack(weights, dim=2)
            


            # --------------------------------------------------------------------------------------------------------- #



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
            """
        return forward

    def motion_forward_basic(self, layer_name):

        def _chunked_feed_forward(ff: nn.Module,
                                  hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int):
            # "feed_forward_chunk_size" can be used to save memory
            if hidden_states.shape[chunk_dim] % chunk_size != 0:
                raise ValueError(
                    f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                )
            # ----------------------------------------------------------------------------------------------------------
            # dimension shrinking
            num_chunks = hidden_states.shape[chunk_dim] // chunk_size

            # module applying hid_slice
            ff_output = torch.cat([ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
                dim=chunk_dim,)
            return ff_output

        def forward(
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                timestep: Optional[torch.LongTensor] = None,
                cross_attention_kwargs: Dict[str, Any] = None,
                class_labels: Optional[torch.LongTensor] = None,
                added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        ) -> torch.Tensor:

            do_skip = False
            for skip_layer in editor.skip_layers:
                if skip_layer in layer_name.lower() :
                    do_skip = True
                    print(f'{layer_name} do skip')

            if not do_skip :
                if cross_attention_kwargs is not None:
                    if cross_attention_kwargs.get("scale", None) is not None:
                        logger.warning(
                            "Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

                # Notice that normalization is always applied before the real computation in the following blocks.
                # 0. Self-Attention
                batch_size = hidden_states.shape[0]

                if self.norm_type == "ada_norm":
                    norm_hidden_states = self.norm1(hidden_states, timestep)
                elif self.norm_type == "ada_norm_zero":
                    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                        hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                    )
                elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
                    norm_hidden_states = self.norm1(hidden_states)
                elif self.norm_type == "ada_norm_continuous":
                    norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
                elif self.norm_type == "ada_norm_single":
                    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
                    ).chunk(6, dim=1)
                    norm_hidden_states = self.norm1(hidden_states)
                    norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
                    norm_hidden_states = norm_hidden_states.squeeze(1)
                else:
                    raise ValueError("Incorrect norm used")

                if self.pos_embed is not None:
                    norm_hidden_states = self.pos_embed(norm_hidden_states)

                # 1. Prepare GLIGEN inputs
                cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
                gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
                if self.norm_type == "ada_norm_zero":
                    attn_output = gate_msa.unsqueeze(1) * attn_output
                elif self.norm_type == "ada_norm_single":
                    attn_output = gate_msa * attn_output

                hidden_states = attn_output + hidden_states
                if hidden_states.ndim == 4:
                    hidden_states = hidden_states.squeeze(1)

                # 1.2 GLIGEN Control
                if gligen_kwargs is not None:
                    hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

                # 3. Cross-Attention
                if self.attn2 is not None:
                    if self.norm_type == "ada_norm":
                        norm_hidden_states = self.norm2(hidden_states, timestep)
                    elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                        norm_hidden_states = self.norm2(hidden_states)
                    elif self.norm_type == "ada_norm_single":
                        # For PixArt norm2 isn't applied here:
                        # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                        norm_hidden_states = hidden_states
                    elif self.norm_type == "ada_norm_continuous":
                        norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
                    else:
                        raise ValueError("Incorrect norm")

                    if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                        norm_hidden_states = self.pos_embed(norm_hidden_states)

                    attn_output = self.attn2(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=encoder_attention_mask,
                        **cross_attention_kwargs,
                    )
                    hidden_states = attn_output + hidden_states

                # 4. Feed-forward
                # i2vgen doesn't have this norm
                if self.norm_type == "ada_norm_continuous":
                    norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
                elif not self.norm_type == "ada_norm_single":
                    norm_hidden_states = self.norm3(hidden_states)

                if self.norm_type == "ada_norm_zero":
                    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

                if self.norm_type == "ada_norm_single":
                    norm_hidden_states = self.norm2(hidden_states)
                    norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

                if self._chunk_size is not None:
                    # "feed_forward_chunk_size" can be used to save memory
                    ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
                else:
                    ff_output = self.ff(norm_hidden_states)

                if self.norm_type == "ada_norm_zero":
                    ff_output = gate_mlp.unsqueeze(1) * ff_output
                elif self.norm_type == "ada_norm_single":
                    ff_output = gate_mlp * ff_output

                hidden_states = ff_output + hidden_states
                if hidden_states.ndim == 4:
                    hidden_states = hidden_states.squeeze(1)

            return hidden_states

        return forward

    def register_editor(net, count, place_in_unet, net_name):
        for name, subnet in net.named_children():
            final_name = f"{net_name}_{name}"
            # start from "TransformerTemporalModel"

            if subnet.__class__.__name__ == 'BasicTransformerBlock' and 'motion' in final_name.lower():
                # [2] BasicTransformerBlock
                subnet.forward = motion_forward_basic(subnet, final_name)
                # [1] Attention
                attn = subnet.attn1
                subnet.attn1.forward = motion_forward(attn, final_name)
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