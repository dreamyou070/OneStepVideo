import os
import math
import wandb
import random
import logging
import inspect
import argparse
from datetime import datetime
import subprocess
from utils.layer_dictionary import find_layer_name
from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from typing import Dict, Optional, Tuple
from accelerate import Accelerator
import torch
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from accelerate import DistributedDataParallelKwargs
import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models import MotionAdapter
from diffusers.pipelines import AnimateDiffPipeline
import transformers
from transformers import CLIPTextModel, CLIPTokenizer
# from data.dataset import WebVid10M
from data.dataset_gen import DistillWebVid10M
from animatediff.models.unet import UNet3DConditionModel
from accelerate import accelerator
from utils.layer_dictionary import find_layer_name
from attn.masactrl_utils import (regiter_attention_editor_diffusers, regiter_motion_attention_editor_diffusers)
from attn.masactrl import MutualSelfAttentionControl, MutualMotionAttentionControl
from diffusers import LCMScheduler
from diffusers.utils import export_to_gif, load_image
from utils.util import save_videos_grid
import GPUtil
import json
from deepspeed.pipe import PipelineModule

def main(args):

    check_min_version("0.10.0.dev0")
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO, )

    logger.info(f'\n step 1. wandb start')
    log_with = "tensorboard"
    if args.use_wandb:
        log_with = "wandb"
        skip_layers = None
        folder_name = ""
        if args.skip_layers is not None:
            for i, layer_name in enumerate(args.skip_layers):
                if i == len(args.skip_layers) - 1:
                    folder_name += f"{layer_name}"
                else:
                    folder_name += f"{layer_name}_"
            skip_layers = find_layer_name(args.skip_layers)

    logger.info(f'\n step 2. set seed')
    torch.manual_seed(args.seed)

    logger.info(f'\n step 3. preparing accelerator')
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], log_with=log_with)
    is_main_process = accelerator.is_main_process

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16  # here !
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    logger.info(f'\n step 4. saving dir')
    save_folder = os.path.join(output_dir, f"inferece_{args.saved_epoch}")
    os.makedirs(save_folder, exist_ok=True)

    logger.info(f'\n step 5. set model')
    logger.info(f' (5.1) adapter')
    checkpoint_base_dir = os.path.join(args.output_dir, 'checkpoints')
    saved_epoch = args.saved_epoch

    origin_checkpoint_dir = os.path.join(checkpoint_base_dir, f'checkpoint_epoch0.pt')
    origin_state_dict = torch.load(origin_checkpoint_dir, map_location="cpu")

    trained_checkpoint_dir = os.path.join(checkpoint_base_dir, f'checkpoint_epoch49.pt')
    trained_state_dict = torch.load(trained_checkpoint_dir, map_location="cpu") # make adopter

    for k, v in origin_state_dict.items():
        trained_state = trained_state_dict[k]
        origin_state = origin_state_dict[k]
        is_equal = torch.equal(trained_state, origin_state)
        print(f' key = {k}, is_equal = {is_equal}')

    # open checkpoint
    #checkpoint = torch.load(checkpoint_dir, map_location="cpu")
    print(f' Adapter Loading')
    test_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtpe=weight_dtype)
    test_adapter.load_state_dict(trained_state_dict)
    """
    logger.info(f' (4.2) original adapter')
    original_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtpe=weight_dtype)
    """

    logger.info(f' (4.3) test_pipe')
    test_pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism",
                                               motion_adapter=test_adapter,
                                               torch_dtype=torch.float16)
    """
    logger.info(f' (4.4) original_pipe')
    original_pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism",
                                                   motion_adapter=original_adapter,
                                                   torch_dtype=torch.float16)
    
    noise_scheduler = LCMScheduler.from_config(test_pipe.scheduler.config, beta_schedule="linear")
    #original_pipe.scheduler = noise_scheduler
    #original_pipe.to(accelerator.device, dtype=weight_dtype)

    test_pipe.scheduler = noise_scheduler
    test_pipe.to(accelerator.device, dtype=weight_dtype)

    logger.info(f' (4.5) test unet skipping')
    test_unet = test_pipe.unet
    test_attention_editor = MutualMotionAttentionControl(guidance_scale=args.guidance_scale,
                                                         frame_num=16,
                                                         full_attention=args.full_attention,
                                                         window_attention=args.window_attention,
                                                         window_size=args.window_size,
                                                         total_frame_num=args.num_frames,
                                                         skip_layers=skip_layers,
                                                         is_teacher=False)
    regiter_motion_attention_editor_diffusers(test_unet, test_attention_editor)
    test_pipe.unet = test_unet
    test_pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                                adapter_name="lcm-lora")
    test_pipe.set_adapters(["lcm-lora"], [0.8])
    # controller
    origin_attention_editor = MutualMotionAttentionControl(guidance_scale=args.guidance_scale,
                                                            frame_num=16,
                                                            full_attention=args.full_attention,
                                                            window_attention=args.window_attention,
                                                            window_size=args.window_size,
                                                            total_frame_num=args.num_frames,
                                                            skip_layers=skip_layers,
                                                            is_teacher=False)
    regiter_motion_attention_editor_diffusers(original_pipe.unet, origin_attention_editor)
    original_pipe.unet = original_pipe.unet
    original_pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                                    adapter_name="lcm-lora")
    original_pipe.set_adapters(["lcm-lora"], [0.8])

    logger.info(f'\n step 5. inference')
    num_frames = 16
    save_base_dir = f'experiment_20240710_jpg_gif_mp4/general_prompt_num_frames_{num_frames}_0709_inference'
    os.makedirs(save_base_dir, exist_ok=True)

    print(f' \n step 3. inference test')
    prompt_dir = f'./configs/prompts/filtered_captions_val_{args.start_num}_{args.end_num}.txt'
    with open(prompt_dir, 'r') as f:
        prompts = f.readlines()
    guidance_scales = [1.5]
    num_inference_steps = [6]
    n_prompt = "bad quality, worse quality, low resolution"
    seeds = [42]
    for p, prompt in enumerate(prompts):

        save_p = str(p).zfill(3)
        prompt_folder = os.path.join(save_base_dir, f'prompt_idx_{save_p}')
        print(f'prompt_folder = {prompt_folder}')
        os.makedirs(prompt_folder, exist_ok=True)
        # prompt setting
        with open(os.path.join(prompt_folder, 'prompt.txt'), 'w') as f:
            f.write(prompt)
        for guidance_scale in guidance_scales:
            for inference_scale in num_inference_steps:
                for seed in seeds:

                    base_folder = os.path.join(prompt_folder, f'guidance_{guidance_scale}_inference_{inference_scale}')
                    os.makedirs(base_folder, exist_ok=True)

                    # seed setting
                    print(f' test pipe line')

                    output = test_pipe(prompt=prompt,
                                  negative_prompt=n_prompt,
                                  num_frames=num_frames,
                                  guidance_scale=guidance_scale,
                                  num_inference_steps=inference_scale,
                                  generator=torch.Generator("cpu").manual_seed(seed), )
                    test_attention_editor.reset()
                    frames = output.frames[0]
                    #

                    # save_folder = os.path.join(base_folder, f'origin_elapse_time_{elapse_time}')
                    os.makedirs(save_folder, exist_ok=True)
                    export_to_gif(frames, os.path.join(save_folder, f'prompt_{save_p}_seed_{seed}.gif'))

                    print(f' original pipe line')
                    original_output = original_pipe(prompt=prompt,
                                  negative_prompt=n_prompt,
                                  num_frames=num_frames,
                                  guidance_scale=guidance_scale,
                                  num_inference_steps=inference_scale,
                                  generator=torch.Generator("cpu").manual_seed(seed), )
                    origin_attention_editor.reset()
                    origin_frames = original_output.frames[0]
                    export_to_gif(origin_frames, os.path.join(save_folder, f'origin_prompt_{save_p}_seed_{seed}.gif'))
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default='video_distill')
    parser.add_argument('--sub_folder_name', type=str, default='result_sy')
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", default='fp16')
    parser.add_argument('--full_attention', action='store_true')
    parser.add_argument('--window_attention', action='store_true')
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--motion_control', action='store_true')
    parser.add_argument('--num_frames', type=int, default=16)
    from utils import arg_as_list
    parser.add_argument('--skip_layers', type=arg_as_list)
    parser.add_argument('--sample_n_frames', type=int, default=16)
    parser.add_argument('--vlb_weight', type=float, default=1.0)
    parser.add_argument('--distill_weight', type=float, default=1.0)
    parser.add_argument('--loss_feature_weight', type=float, default=1.0)
    parser.add_argument('--guidance_scale', type=float, default=1.5)
    parser.add_argument('--inference_step', type=int, default=6)
    parser.add_argument('--csv_path', type=str, default='data/webvid-10M.csv')
    parser.add_argument('--video_folder', type=str, default='data/webvid-10M')
    parser.add_argument('--random_init', action='store_true')
    parser.add_argument('--adam_weight_decay', type=float, default=1e-2)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--launcher', type = str)
    parser.add_argument('--cfg_random_null_text', action='store_true')
    parser.add_argument('--cfg_random_null_text_ratio', type=float, default=1e-1)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--global_seed', type=int, default=42)
    parser.add_argument('--start_num', type=int, default=0)
    parser.add_argument('--end_num', type=int, default=100)
    parser.add_argument('--saved_epoch', type=int, default=16)
    args = parser.parse_args()
    main(args)