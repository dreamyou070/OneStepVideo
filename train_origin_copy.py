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
import torch.nn.functional as F
from accelerate import DistributedDataParallelKwargs
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.models import MotionAdapter
from diffusers.pipelines import AnimateDiffPipeline
from data.dataset_gen import DistillWebVid10M
from utils.layer_dictionary import find_layer_name
from attn.masactrl_utils import (regiter_attention_editor_diffusers, regiter_motion_attention_editor_diffusers)
from attn.masactrl import MutualSelfAttentionControl, MutualMotionAttentionControl
from diffusers import LCMScheduler
from diffusers.utils import export_to_gif, load_image
import GPUtil
import json


def main(args):
    GPUtil.showUtilization()
    check_min_version("0.10.0.dev0")

    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO, )

    logger.info(f'\n step 1. wandb start')
    log_with = "tensorboard"
    skip_layers, skip_layers_dot = find_layer_name(args.skip_layers)
    if args.use_wandb:
        log_with = "wandb"
        folder_name = ""
        if len(args.skip_layers) > 0:
            for i in range(len(args.skip_layers)):
                layer_name = args.skip_layers[i]
                if i == len(args.skip_layers) - 1:
                    folder_name += f"{layer_name}"
                else:
                    folder_name += f"{layer_name}_"
    logger.info(f'\n step 2. preparing folder')
    torch.manual_seed(args.seed)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    folder_name = args.sub_folder_name
    output_dir = os.path.join(output_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # [2]
    wandb.init(project=args.project, name=f'experiment_{args.sub_folder_name}')
    weight_dtype = torch.float32
    # print(f'weight_dtype: {weight_dtype}')
    # weight_dtype = torch.float16

    logger.info(f'\n step 3. saving dir')
    os.makedirs(f"{output_dir}/samples", exist_ok=True)
    os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    from diffusers import AutoencoderKL, DDIMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers.models import UNet2DConditionModel, UNetMotionModel

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
    unet_2d = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")
    motion_adapter = MotionAdapter.from_pretrained(args.teacher_motion_model_dir, torch_dtpe=weight_dtype)
    #
    if args.random_init :
        motion_adapter_config =MotionAdapter.load_config(args.teacher_motion_model_dir, torch_dtpe=weight_dtype)
        print(f' motion_adapter_config = {motion_adapter_config}')
        motion_adapter = MotionAdapter.from_config(motion_adapter_config,
                                                   torch_dtpe=weight_dtype,)
    #config_student = UNet2DConditionModel.load_config(args.unet_config_path, subfolder=args.unet_config_name)
    print(f' Make 3D Unet !')
    unet = UNetMotionModel.from_unet2d(unet_2d, motion_adapter)
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Set unet trainable parameters
    unet.requires_grad_(False)
    for name, param in unet.named_parameters():
        if 'motion' in name:
            param.requires_grad = True

    trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))  # len 333
    optimizer = torch.optim.AdamW(trainable_params,
                                  lr=args.learning_rate,
                                  betas=(args.adam_beta1, args.adam_beta2),
                                  weight_decay=args.adam_weight_decay,
                                  eps=args.adam_epsilon,
                                  foreach=False)

    logger.info(f"trainable params number: {len(trainable_params)}")
    logger.info(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # Move models to GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)

    logger.info(f'\n step 5. Get the training dataset')
    train_dataset = DistillWebVid10M(csv_path=args.csv_path,
                                     video_folder=args.video_folder,
                                     sample_size=512,
                                     sample_stride=4,
                                     sample_n_frames=args.sample_n_frames,
                                     is_image=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=1,  # train_batch_size,
                                                   shuffle=True,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True,
                                                   drop_last=True, )

    # Get the training iteration
    if args.max_train_steps == -1:
        assert args.max_train_epoch != -1
        max_train_steps = args.max_train_epoch * len(train_dataloader)

    if args.checkpointing_steps == -1:
        assert args.checkpointing_epochs != -1
        checkpointing_steps = args.checkpointing_epochs * len(train_dataloader)

    if args.scale_lr:
        learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size)

    # Scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    )
    noise_scheduler = DDIMScheduler(num_train_timesteps=1000,
                                    beta_start=0.00085,
                                    beta_end=0.012,
                                    beta_schedule="linear",
                                    steps_offset=1,
                                    clip_sample=False)
    validation_pipeline = AnimateDiffPipeline(
        unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler, motion_adapter=motion_adapter
    ).to("cuda")
    noise_scheduler = LCMScheduler.from_config(validation_pipeline.scheduler.config,
                                               beta_schedule="linear")
    print(f'noise_scheduler.config.prediction_type = {noise_scheduler.config.prediction_type}')

    unet.to("cuda", dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps


    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {len(train_dataset)}")
    logging.info(f"  Num Epochs = {num_train_epochs}")
    logging.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logging.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps))
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision_training else None

    for epoch in range(first_epoch, num_train_epochs):
        #train_dataloader.sampler.set_epoch(epoch)
        unet.train()

        # ----------------------------------------------------------------------------------------------------
        # validation
        samples = []
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)
        height = 512
        width = 512
        from animatediff.utils.util import save_videos_grid, zero_rank_print
        prompts = ["A video of a woman, having a selfie"]
        for idx, prompt in enumerate(prompts):
            sample = validation_pipeline(
                prompt,
                generator=generator,
                video_length=args.sample_n_frames,
                height=height,
                width=width,
                num_inference_steps=25,
                guidance_scale=8,
            ).frames[0]
            export_to_gif(sample, f"{output_dir}/samples/sample-{global_step}-{idx}.gif")


            fps = 10
            wandb.log({"video": wandb.Video(data_or_path=f"{output_dir}/samples/sample-{global_step}-{idx}.gif",
                                            caption=f'epoch_{epoch}',
                                            fps=fps)})
        # ------------------------------------------------------------------------------------------------------------------------

        for step, batch in enumerate(train_dataloader):
            ### >>>> Training >>>> ###

            # Convert videos to latent space
            pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
            video_length = pixel_values.shape[1]
            with torch.no_grad():
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                latents = vae.encode(pixel_values).latent_dist
                latents = latents.sample()
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample a random timestep for each video
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            with torch.no_grad():
                prompt_ids = tokenizer(
                    batch['text'], max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
                    return_tensors="pt"
                ).input_ids.to(latents.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            # Mixed-precision training
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            print(f'loss = {loss}')

            optimizer.zero_grad()

            # Backpropagate
            loss.backward()
            torch.nn.utils.clip_grad_norm_(#unet.parameters(),
                                           trainable_params,
                                           args.max_grad_norm)
            # 각 파라미터의 그래디언트를 출력
            print(f'Epoch {epoch}')
            for name, param in unet.named_parameters():
                if param.grad is not None:
                    print(f'Gradient for {name}: {param.grad}')

            optimizer.step()

            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1
            wandb.log({"train_loss": loss.item()}, step=global_step)
        # ------------------------------------------------------------------------------------------------------------------------
        # Save checkpoint
        save_path = os.path.join(output_dir, f"checkpoints")
        state_dict = {
                "epoch": epoch,
                "global_step": global_step,
                "state_dict": unet.state_dict(),
            }
        if step == len(train_dataloader) - 1:
            torch.save(state_dict, os.path.join(save_path, f"checkpoint-epoch-{epoch + 1}.ckpt"))
        else:
            torch.save(state_dict, os.path.join(save_path, f"checkpoint.ckpt"))
        logging.info(f"Saved state to {save_path} (global_step: {global_step})")
        logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)

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
    parser.add_argument('--csv_path', type=str, default='data/webvid-10M.csv')
    parser.add_argument('--video_folder', type=str, default='data/webvid-10M')
    parser.add_argument('--random_init', action='store_true')
    parser.add_argument('--adam_weight_decay', type=float, default=1e-2)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--name', type=str, default='video_distill')
    parser.add_argument('--output_dir', type=str, default='experiment')
    parser.add_argument('--pretrained_model_path', type=str, default='')
    parser.add_argument('--teacher_motion_model_dir', type=str, default="wangfuyun/AnimateLCM")
    parser.add_argument('--cfg_random_null_text', action='store_true')
    parser.add_argument('--cfg_random_null_text_ratio', type=float, default=0.1)
    parser.add_argument('--unet_checkpoint_path', type=str, default='')
    parser.add_argument('--unet_additional_kwargs', type=Dict)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--noise_scheduler_kwargs', type=Dict)
    parser.add_argument('--max_train_epoch', type=int, default=-1)
    parser.add_argument('--max_train_steps', type=int, default=-1)
    parser.add_argument('--validation_steps', type=int, default=100)
    parser.add_argument('--validation_steps_tuple', type=Tuple, default=(-1,))
    parser.add_argument('--scale_lr', action='store_true')
    parser.add_argument('--lr_warmup_steps', type=int, default=0)
    parser.add_argument('--lr_scheduler', type=str, default='constant')
    parser.add_argument('--trainable_modules', type=arg_as_list, default="['motion_modules.']")
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--adam_epsilon', type=float, default=1e-08)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--checkpointing_epochs', type=int, default=5)
    parser.add_argument('--checkpointing_steps', type=int, default=-1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--mixed_precision_training', action='store_true')
    parser.add_argument('--enable_xformers_memory_efficient_attention', action='store_true')
    parser.add_argument('--is_debug', action='store_true')
    args = parser.parse_args()
    name = Path(args.config).stem
    main(args)