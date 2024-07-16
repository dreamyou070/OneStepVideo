import os

import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from .masactrl_utils import AttentionBase
from torchvision.utils import save_image


# ------------------------------------------------------------------------------------------------------------------------
# MutualSelfAttentionControl ---------------------------------------------------------------------------------------------
class MutualSelfAttentionControl(AttentionBase):

    """ because attentionbase, self.cur_layer += 1 """

    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70}

    def __init__(self,
                 guidance_scale,
                 start_step=4,
                 start_layer=10, # layper 10
                 layer_idx=None,
                 step_idx=None,
                 total_steps=50,
                 model_type="SD",
                 frame_num = 16,):
        self.guidance_scale = guidance_scale
        """
        Mutual self-attention control for Stable-Diffusion model
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control # from where to start
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            model_type: the model type, SD or SDXL
        """
        super().__init__()
        self.total_steps = total_steps                              # 50
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)     # 16

        self.start_layer = start_layer                              # 10
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.total_layers)) # layer_idx = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        #print(f' MassaCtrl for {self.layer_idx} layers ')

        self.start_step = start_step  # 4
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))           # step_idx =  [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
        #print(" MasaCtrl at denoising steps: ", self.step_idx)
        #print(f" MasaCtrl for {self.step_idx}")
        self.frame_num = frame_num



    def attn_batch(self, q, k, v, sim, #attn,
                   is_cross, place_in_unet, num_heads, **kwargs):
        """        Performing attention for a batch of queries, keys, and values        """

        # q = [16, pix_num, dim]
        # k = [8, pix_num, dim]
        b = q.shape[0] // num_heads # 2
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads) # [(2,8), pix_num, dim] -> [8, (2, pix_num), dim]
        #########################################################################################################
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads) # [(1,8), pix_num, dim] -> [8, (1, pix_num), dim]
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads) # [(1,8), pix_num, dim] -> [8, (1, pix_num), dim]
        #########################################################################################################
        # sim = [8, 2*pix_num, pix_num]
        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale") # [8, pix_num, dim] * [8, pix_num, dim] -> [8, pix_num, pix_num]
        attn = sim.softmax(-1)
        # attn = [8, 2*pix_num, dim]
        out = torch.einsum("h i j, h j d -> h i d", attn, v) # normal attention
        # out = [(8*2), pix_num, dim]
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b) # (2, b, n, d) -> b, n, (2, d)
        return out
    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """        Attention forward function        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        # [2] massctrl forward
        qu, qc = q.chunk(2) # [head*[ref,cur], pix_num, dim]
        ku, kc = k.chunk(2) # [head*[ref,cur], pix_num, dim]
        vu, vc = v.chunk(2) # [head*[ref,cur], pix_num, dim]
        attnu, attnc = attn.chunk(2) # [head*[ref,cur], pix_num, pix_num]

        # what is sim ?
        print(f' start query, qu = {qu.shape}')
        print(f' num_heads = {num_heads}')

        out_u = self.attn_batch(qu,             # total q
                                ku[:num_heads], # only reference k
                                vu[:num_heads],
                                sim[:num_heads],
                                #attnu,
                                is_cross, place_in_unet, num_heads, **kwargs) # [reference, current] -> (b, n, 2d)

        out_c = self.attn_batch(qc,
                                kc[:num_heads],
                                vc[:num_heads],
                                sim[:num_heads], #attnc,
                                is_cross, place_in_unet, num_heads, **kwargs) # [reference, current] -> (b, n, 2d)

        out = torch.cat([out_u, out_c], dim=0)

        return out

class MutualMotionAttentionControl(AttentionBase):

    """ because attentionbase, self.cur_layer += 1 """

    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70}

    def __init__(self,
                 start_step=4,
                 start_layer=10, # layper 10
                 layer_idx=None,
                 step_idx=None,
                 total_steps=50,
                 model_type="SD",
                 guidance_scale = 1,
                 frame_num = 16,
                 full_attention = True,
                 window_attention = False,
                 window_size = 16,
                 total_frame_num = 64,
                 skip_layers = ['down_0_0'],
                 is_teacher = False,
                 is_eval = False,
                 do_attention_map_check = False) :
        """
        Mutual self-attention control for Stable-Diffusion model
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control # from where to start
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            model_type: the model type, SD or SDXL
        """
        super().__init__()
        self.total_steps = total_steps                              # 50
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)     # 16

        self.start_layer = start_layer                              # 10
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.total_layers)) # layer_idx = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        #print(f' MassaCtrl for {self.layer_idx} layers ')

        self.start_step = start_step  # 4
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))           # step_idx =  [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
        #print(" MasaCtrl at denoising steps: ", self.step_idx)
        #print(f" MasaCtrl for {self.step_idx}").
        self.guidance_scale = guidance_scale
        self.frame_num = frame_num
        self.full_attention = full_attention
        self.window_attention = window_attention
        self.window_size = window_size
        self.total_frame_num = total_frame_num
        self.frame_num = 16
        self.group_idx = 0
        self.group_index = 0
        self.attnmap_dict = {}
        self.timestep = 0
        self.skip_layers = skip_layers
        self.layerwise_hidden_dict = {}
        self.is_teacher = is_teacher
        self.is_eval = is_eval
        self.do_attention_map_check = do_attention_map_check


    def attn_batch(self, q, k, v, sim, #attn,
                   is_cross, place_in_unet, num_heads, **kwargs):
        """        Performing attention for a batch of queries, keys, and values        """

        # q = [16, pix_num, dim]
        # k = [8, pix_num, dim]
        b = q.shape[0] // num_heads # 2
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads) # [(2,8), pix_num, dim] -> [8, (2, pix_num), dim]
        #########################################################################################################
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads) # [(1,8), pix_num, dim] -> [8, (1, pix_num), dim]
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads) # [(1,8), pix_num, dim] -> [8, (1, pix_num), dim]
        #########################################################################################################
        # sim = [8, 2*pix_num, pix_num]
        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale") # [8, pix_num, dim] * [8, pix_num, dim] -> [8, pix_num, pix_num]
        attn = sim.softmax(-1)
        # attn = [8, 2*pix_num, dim]
        out = torch.einsum("h i j, h j d -> h i d", attn, v) # normal attention
        # out = [(8*2), pix_num, dim]
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b) # (2, b, n, d) -> b, n, (2, d)
        return out
    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """        Attention forward function        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        # [2] massctrl forward
        qu, qc = q.chunk(2) # [head*[ref,cur], pix_num, dim]
        ku, kc = k.chunk(2) # [head*[ref,cur], pix_num, dim]
        vu, vc = v.chunk(2) # [head*[ref,cur], pix_num, dim]
        attnu, attnc = attn.chunk(2) # [head*[ref,cur], pix_num, pix_num]


        out_u = self.attn_batch(qu,             # total q
                                ku[:num_heads], # only reference k
                                vu[:num_heads],
                                sim[:num_heads],
                                #attnu,
                                is_cross, place_in_unet, num_heads, **kwargs) # [reference, current] -> (b, n, 2d)

        out_c = self.attn_batch(qc,
                                kc[:num_heads],
                                vc[:num_heads],
                                sim[:num_heads], #attnc,
                                is_cross, place_in_unet, num_heads, **kwargs) # [reference, current] -> (b, n, 2d)

        out = torch.cat([out_u, out_c], dim=0)

        return out

    def save_hidden_states(self, hidden_states = None, layer_name = None):
        if layer_name not in self.layerwise_hidden_dict:
            self.layerwise_hidden_dict[layer_name] = []
        self.layerwise_hidden_dict[layer_name].append(hidden_states)


    def save_attention_map(self, attn_map, layer_name):
        if layer_name not in self.attnmap_dict:
            self.attnmap_dict[layer_name] = {}
        self.attnmap_dict[layer_name][f'time_{self.timestep}'] = attn_map

    def set_timestep(self, timestep):
        self.timestep = timestep

    def reset(self):

        self.attnmap_dict = {}
        self.layerwise_hidden_dict = {}
        self.skip_layers = []




class MutualSelfAttentionControlUnion(MutualSelfAttentionControl):
    def __init__(self,
                 start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, model_type="SD"):
        """
        Mutual self-attention control for Stable-Diffusion model with unition source and target [K, V]
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            model_type: the model type, SD or SDXL
        """

        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps, model_type)

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        qu_s, qu_t, qc_s, qc_t = q.chunk(4)
        ku_s, ku_t, kc_s, kc_t = k.chunk(4)
        vu_s, vu_t, vc_s, vc_t = v.chunk(4)
        attnu_s, attnu_t, attnc_s, attnc_t = attn.chunk(4)

        # source image branch
        out_u_s = super().forward(qu_s, ku_s, vu_s, sim, attnu_s, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_s = super().forward(qc_s, kc_s, vc_s, sim, attnc_s, is_cross, place_in_unet, num_heads, **kwargs)
        # target image branch, concatenating source and target [K, V]
        out_u_t = self.attn_batch(qu_t, torch.cat([ku_s, ku_t]), torch.cat([vu_s, vu_t]), sim[:num_heads], attnu_t, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_t = self.attn_batch(qc_t, torch.cat([kc_s, kc_t]), torch.cat([vc_s, vc_t]), sim[:num_heads], attnc_t, is_cross, place_in_unet, num_heads, **kwargs)

        out = torch.cat([out_u_s, out_u_t, out_c_s, out_c_t], dim=0)

        return out


class MutualSelfAttentionControlMask(MutualSelfAttentionControl):
    def __init__(self,  start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50,  mask_s=None, mask_t=None, mask_save_dir=None, model_type="SD"):
        """
        Maske-guided MasaCtrl to alleviate the problem of fore- and background confusion
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            mask_s: source mask with shape (h, w)
            mask_t: target mask with same shape as source mask
            mask_save_dir: the path to save the mask image
            model_type: the model type, SD or SDXL
        """
        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps, model_type)
        self.mask_s = mask_s  # source mask with shape (h, w)
        self.mask_t = mask_t  # target mask with same shape as source mask
        print("Using mask-guided MasaCtrl")
        if mask_save_dir is not None:
            os.makedirs(mask_save_dir, exist_ok=True)
            save_image(self.mask_s.unsqueeze(0).unsqueeze(0), os.path.join(mask_save_dir, "mask_s.png"))
            save_image(self.mask_t.unsqueeze(0).unsqueeze(0), os.path.join(mask_save_dir, "mask_t.png"))

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        if kwargs.get("is_mask_attn") and self.mask_s is not None:
            print("masked attention")
            mask = self.mask_s.unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mask, (H, W)).flatten(0).unsqueeze(0)
            mask = mask.flatten()
            # background
            sim_bg = sim + mask.masked_fill(mask == 1, torch.finfo(sim.dtype).min)
            # object
            sim_fg = sim + mask.masked_fill(mask == 0, torch.finfo(sim.dtype).min)
            sim = torch.cat([sim_fg, sim_bg], dim=0)
        attn = sim.softmax(-1)
        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        B = q.shape[0] // num_heads // 2
        H = W = int(np.sqrt(q.shape[1]))
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)

        out_u_source = self.attn_batch(qu[:num_heads], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_source = self.attn_batch(qc[:num_heads], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)

        out_u_target = self.attn_batch(qu[-num_heads:], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, is_mask_attn=True, **kwargs)
        out_c_target = self.attn_batch(qc[-num_heads:], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, is_mask_attn=True, **kwargs)

        if self.mask_s is not None and self.mask_t is not None:
            out_u_target_fg, out_u_target_bg = out_u_target.chunk(2, 0)
            out_c_target_fg, out_c_target_bg = out_c_target.chunk(2, 0)

            mask = F.interpolate(self.mask_t.unsqueeze(0).unsqueeze(0), (H, W))
            mask = mask.reshape(-1, 1)  # (hw, 1)
            out_u_target = out_u_target_fg * mask + out_u_target_bg * (1 - mask)
            out_c_target = out_c_target_fg * mask + out_c_target_bg * (1 - mask)

        out = torch.cat([out_u_source, out_u_target, out_c_source, out_c_target], dim=0)
        return out


class MutualSelfAttentionControlMaskAuto(MutualSelfAttentionControl):
    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, thres=0.1, ref_token_idx=[1], cur_token_idx=[1], mask_save_dir=None, model_type="SD"):
        """
        MasaCtrl with mask auto generation from cross-attention map
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            thres: the thereshold for mask thresholding
            ref_token_idx: the token index list for cross-attention map aggregation
            cur_token_idx: the token index list for cross-attention map aggregation
            mask_save_dir: the path to save the mask image
        """
        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps, model_type)
        print("Using MutualSelfAttentionControlMaskAuto")
        self.thres = thres
        self.ref_token_idx = ref_token_idx
        self.cur_token_idx = cur_token_idx

        self.self_attns = []
        self.cross_attns = []

        self.cross_attns_mask = None
        self.self_attns_mask = None

        self.mask_save_dir = mask_save_dir
        if self.mask_save_dir is not None:
            os.makedirs(self.mask_save_dir, exist_ok=True)

    def after_step(self):
        self.self_attns = []
        self.cross_attns = []

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        if self.self_attns_mask is not None:
            # binarize the mask
            mask = self.self_attns_mask
            thres = self.thres
            mask[mask >= thres] = 1
            mask[mask < thres] = 0
            sim_fg = sim + mask.masked_fill(mask == 0, torch.finfo(sim.dtype).min)
            sim_bg = sim + mask.masked_fill(mask == 1, torch.finfo(sim.dtype).min)
            sim = torch.cat([sim_fg, sim_bg])

        attn = sim.softmax(-1)

        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out

    def aggregate_cross_attn_map(self, idx):
        attn_map = torch.stack(self.cross_attns, dim=1).mean(1)  # (B, N, dim)
        B = attn_map.shape[0]
        res = int(np.sqrt(attn_map.shape[-2]))
        attn_map = attn_map.reshape(-1, res, res, attn_map.shape[-1])
        image = attn_map[..., idx]
        if isinstance(idx, list):
            image = image.sum(-1)
        image_min = image.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        image_max = image.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        image = (image - image_min) / (image_max - image_min)
        return image

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross:
            # save cross attention map with res 16 * 16
            if attn.shape[1] == 16 * 16:
                self.cross_attns.append(attn.reshape(-1, num_heads, *attn.shape[-2:]).mean(1))

        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        B = q.shape[0] // num_heads // 2
        H = W = int(np.sqrt(q.shape[1]))
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)

        out_u_source = self.attn_batch(qu[:num_heads], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_source = self.attn_batch(qc[:num_heads], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)

        if len(self.cross_attns) == 0:
            self.self_attns_mask = None
            out_u_target = self.attn_batch(qu[-num_heads:], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
            out_c_target = self.attn_batch(qc[-num_heads:], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)
        else:
            mask = self.aggregate_cross_attn_map(idx=self.ref_token_idx)  # (2, H, W)
            mask_source = mask[-2]  # (H, W)
            res = int(np.sqrt(q.shape[1]))
            self.self_attns_mask = F.interpolate(mask_source.unsqueeze(0).unsqueeze(0), (res, res)).flatten()
            if self.mask_save_dir is not None:
                H = W = int(np.sqrt(self.self_attns_mask.shape[0]))
                mask_image = self.self_attns_mask.reshape(H, W).unsqueeze(0)
                save_image(mask_image, os.path.join(self.mask_save_dir, f"mask_s_{self.cur_step}_{self.cur_att_layer}.png"))
            out_u_target = self.attn_batch(qu[-num_heads:], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
            out_c_target = self.attn_batch(qc[-num_heads:], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)

        if self.self_attns_mask is not None:
            mask = self.aggregate_cross_attn_map(idx=self.cur_token_idx)  # (2, H, W)
            mask_target = mask[-1]  # (H, W)
            res = int(np.sqrt(q.shape[1]))
            spatial_mask = F.interpolate(mask_target.unsqueeze(0).unsqueeze(0), (res, res)).reshape(-1, 1)
            if self.mask_save_dir is not None:
                H = W = int(np.sqrt(spatial_mask.shape[0]))
                mask_image = spatial_mask.reshape(H, W).unsqueeze(0)
                save_image(mask_image, os.path.join(self.mask_save_dir, f"mask_t_{self.cur_step}_{self.cur_att_layer}.png"))
            # binarize the mask
            thres = self.thres
            spatial_mask[spatial_mask >= thres] = 1
            spatial_mask[spatial_mask < thres] = 0
            out_u_target_fg, out_u_target_bg = out_u_target.chunk(2)
            out_c_target_fg, out_c_target_bg = out_c_target.chunk(2)

            out_u_target = out_u_target_fg * spatial_mask + out_u_target_bg * (1 - spatial_mask)
            out_c_target = out_c_target_fg * spatial_mask + out_c_target_bg * (1 - spatial_mask)

            # set self self-attention mask to None
            self.self_attns_mask = None

        out = torch.cat([out_u_source, out_u_target, out_c_source, out_c_target], dim=0)
        return out
