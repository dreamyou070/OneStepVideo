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
from data.dataset import WebVid10M
from animatediff.models.unet import UNet3DConditionModel
from accelerate import accelerator
from utils.layer_dictionary import find_layer_name
from masactrl.masactrl_utils import (regiter_attention_editor_diffusers, regiter_motion_attention_editor_diffusers)
from masactrl.masactrl import MutualSelfAttentionControl, MutualMotionAttentionControl
from diffusers import LCMScheduler
from diffusers.utils import export_to_gif, load_image
from utils.util import save_videos_grid
import GPUtil
import json
from deepspeed.pipe import PipelineModule
import yaml

def main(image_finetune: bool,
         name: str,
         use_wandb: bool,
         launcher: str,

         output_dir: str,
         pretrained_model_path: str,

         train_data: Dict,
         validation_data: Dict,
         cfg_random_null_text: bool = True,
         cfg_random_null_text_ratio: float = 0.1,

         unet_checkpoint_path: str = "",
         unet_additional_kwargs: Dict = {},
         ema_decay: float = 0.9999,
         noise_scheduler_kwargs=None,

         max_train_epoch: int = -1,
         max_train_steps: int = 100,
         validation_steps: int = 100,
         validation_steps_tuple: Tuple = (-1,),

         learning_rate: float = 3e-5,
         scale_lr: bool = False,
         lr_warmup_steps: int = 0,
         lr_scheduler: str = "constant",

         trainable_modules: Tuple[str] = (None,),
         num_workers: int = 32,
         train_batch_size: int = 1,
         adam_beta1: float = 0.9,
         adam_beta2: float = 0.999,
         adam_weight_decay: float = 1e-2,
         adam_epsilon: float = 1e-08,
         max_grad_norm: float = 1.0,
         gradient_accumulation_steps: int = 1,
         gradient_checkpointing: bool = False,
         checkpointing_epochs: int = 5,
         checkpointing_steps: int = -1,

         mixed_precision_training: bool = True,  # False,

         enable_xformers_memory_efficient_attention: bool = True,

         global_seed: int = 42,
         is_debug: bool = False,
         args: argparse.Namespace = None,
         ):

    GPUtil.showUtilization()
    check_min_version("0.10.0.dev0")

    logger = logging.getLogger(__name__)
    # level = INFO
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO, )

    logger.info(f'\n step 1. wandb start')
    present_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_with = "tensorboard"
    if use_wandb:
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
    output_dir = 'experiment'
    os.makedirs(output_dir, exist_ok=True)
    # [1]
    folder_name = args.sub_folder_name
    folder_name = folder_name + "_" + datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    # [2]
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps,
                              kwargs_handlers=[ddp_kwargs],
                              log_with=log_with)
    is_main_process = accelerator.is_main_process
    if is_main_process:
        wandb.init(project=args.project, name=f'experiment_{args.sub_folder_name}')
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16  # here !
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    print(f'weight_dtype: {weight_dtype}')

    logger.info(f'\n step 4. saving dir')
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        sample_save_dir = os.path.join(output_dir, "generate_samples")
        os.makedirs(sample_save_dir, exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))
        with open(os.path.join(output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

    csv_folder = os.path.join(output_dir, "generate_image_csv")
    video_folder = os.path.join(output_dir, "generate_image_video")
    os.makedirs(csv_folder, exist_ok=True)
    os.makedirs(video_folder, exist_ok=True)

    logger.info(f'\n step 4. set model')
    logger.info(f' (4.1) teacher')
    teacher_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtpe=weight_dtype)
    teacher_pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=teacher_adapter,
                                                       torch_dtpe=weight_dtype)
    noise_scheduler = LCMScheduler.from_config(teacher_pipe.scheduler.config, beta_schedule="linear")
    teacher_pipe.scheduler = noise_scheduler
    teacher_pipe.to(accelerator.device, dtype=weight_dtype)

    logger.info(f'\n step 5. prompt_dir')

    prompt_dir = r'configs/prompts/test_prompts.txt'
    with open(prompt_dir, 'r') as f:
        prompts = f.readlines()

    negative_prompt = "ImgFixerPre0.3, glowing face, bad proportions, blurry, blurred composition, low resolution, bad, ugly, bad composition, terrible, 3d, render, comic, manga, flat, watermark, signature, worst quality, low quality, normal quality, lowres, simple background, inaccurate limb, extra fingers, fewer fingers, missing fingers, extra arms, extra legs, inaccurate eyes, bad composition, bad anatomy, error, extra digit, fewer digits, cinnadust, cropped, low res, worst quality, low quality, normal quality, jpeg artifacts, extra digit, fewer digits, trademark, watermark, artist's name, username, signature, text, words, human, blurry, blurred composition, blurry foreground, blurry background"

    csv_dir = os.path.join(csv_folder, f'video_dataset.csv')
    with open(csv_dir, 'w') as f:
        f.write('videoid,name,page_dir\n')

    for videoid, prompt in enumerate(prompts) :
        # student_pipe.enable_vae_slicing()
        # student_motion_controller.is_eval = True
        with torch.no_grad():
            # 16 frame images
            output = teacher_pipe(prompt=prompt,
                                  negative_prompt=negative_prompt,
                                  ip_adapter_image=None,  # ip_adapter_image
                                  num_frames=args.num_frames,  # length = 48
                                  guidance_scale=args.guidance_scale,
                                  num_inference_steps=args.inference_step,
                                  generator=torch.Generator("cpu").manual_seed(0),
                                  motion_controller=None).frames[0]
            video_name = str(videoid).zfill(6)
            video_save_dir = os.path.join(os.path.join(video_folder, f'{video_name}.gif'))
            export_to_gif(output, video_save_dir)
            fps = 10
            wandb.log({"video": wandb.Video(data_or_path=video_save_dir, caption=f'{prompt}', fps=fps)})
            with open(csv_dir, 'w') as f:
                f.write(f'{video_name},{video_save_dir},f{prompt}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument('--project', type=str, default='video_distill')
    parser.add_argument('--sub_folder_name', type=str, default='result_sy')
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
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
    args = parser.parse_args()
    name = Path(args.config).stem
    config = OmegaConf.load(args.config)
    main(name=name, launcher=args.launcher, use_wandb=args.wandb, args=args, **config)