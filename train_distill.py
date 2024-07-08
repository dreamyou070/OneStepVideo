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
from masactrl.masactrl_utils import (regiter_attention_editor_diffusers,regiter_motion_attention_editor_diffusers)
from masactrl.masactrl import MutualSelfAttentionControl, MutualMotionAttentionControl
from diffusers import LCMScheduler
from diffusers.utils import export_to_gif, load_image
from utils.util import save_videos_grid
import GPUtil
import json
from deepspeed.pipe import PipelineModule
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

        mixed_precision_training: bool = True, # False,

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
        weight_dtype = torch.float16 # here !
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    print(f'weight_dtype: {weight_dtype}')

    logger.info(f'\n step 4. saving dir')
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))
        # save argument
        # save args as dict

        with open(os.path.join(output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

    logger.info(f'\n step 4. set model')
    logger.info(f' (4.1) teacher')
    teacher_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM",torch_dtpe = weight_dtype)
    teacher_pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=teacher_adapter, torch_dtpe=weight_dtype)
    noise_scheduler = LCMScheduler.from_config(teacher_pipe.scheduler.config, beta_schedule="linear")
    teacher_pipe.scheduler = noise_scheduler
    teacher_pipe.to(accelerator.device, dtype=weight_dtype)
    logger.info(f' (4.2) student')
    student_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM",torch_dtpe = weight_dtype)
    student_pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism",motion_adapter=student_adapter,torch_dtpe = weight_dtype)
    student_pipe.scheduler = noise_scheduler
    student_pipe.to(accelerator.device, dtype=weight_dtype)
    logger.info(f' (4.3) vae, text_encoder, unet')
    vae = teacher_pipe.vae
    tokenizer = teacher_pipe.tokenizer
    text_encoder = teacher_pipe.text_encoder
    teacher_unet = teacher_pipe.unet
    student_unet = student_pipe.unet
    logger.info(f' (4.4) motion controller')

    window_size = 16
    guidance_scale = args.guidance_scale
    inference_step = args.inference_step
    student_motion_controller = None
    teacher_motion_controller = None
    if args.motion_control :
        student_motion_controller = MutualMotionAttentionControl(guidance_scale=guidance_scale,
                                                                 frame_num=16,
                                                                 full_attention=args.full_attention,
                                                                 window_attention=args.window_attention,
                                                                 window_size=window_size,
                                                                 total_frame_num=args.num_frames,
                                                                 skip_layers=skip_layers,
                                                                 is_teacher = False)  # 32
        regiter_motion_attention_editor_diffusers(student_unet, student_motion_controller)
        teacher_motion_controller = MutualMotionAttentionControl(guidance_scale=guidance_scale,
                                                                 frame_num=16,
                                                                 full_attention=args.full_attention,
                                                                 window_attention=args.window_attention,
                                                                 window_size=window_size,
                                                                 total_frame_num=args.num_frames,
                                                                 skip_layers=skip_layers,
                                                                 is_teacher = True)  # 32
        regiter_motion_attention_editor_diffusers(teacher_unet, teacher_motion_controller)

    logger.info(f' (4.5) other modules freeze')
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    teacher_unet.requires_grad_(False)
    student_unet.requires_grad_(False)
    teacher_unet.to(accelerator.device, dtype=weight_dtype)
    for name, param in student_unet.named_parameters():
        for trainable_module_name in trainable_modules:
            if trainable_module_name in name:
                param.requires_grad = True
                break
    trainable_params = list(filter(lambda p: p.requires_grad, student_unet.parameters()))
    optimizer = torch.optim.AdamW(trainable_params,lr=learning_rate,betas=(adam_beta1, adam_beta2),
                                  weight_decay=adam_weight_decay,eps=adam_epsilon,foreach = False)
    if gradient_checkpointing:
        student_unet.enable_gradient_checkpointing()

    logger.info(f'\n step 5. Get the training dataset')
    train_dataset = WebVid10M(csv_path=r'/share0/dreamyou070/dreamyou070/MyData/video/webvid-10M/webvid-10M-csv/0000_1.csv',
                              video_folder=r'/share0/dreamyou070/dreamyou070/MyData/video/webvid-10M/webvid-10M-video',
                              sample_size=512,
                              sample_stride=4,
                              sample_n_frames=args.sample_n_frames,
                              is_image=False)
    print(f' *** len of train_dataset: {len(train_dataset)}')
    # DataLoaders creation : (if number of dataset is lower than train_batch_size, it cannot count)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=1, #train_batch_size,
                                                   shuffle=True,
                                                   num_workers=num_workers,
                                                   pin_memory=True,
                                                   drop_last=True,)

    logger.info(f'\n step 6. training argument')
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)
    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    num_processes = 1
    if scale_lr:
        num_processes = 1
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * num_processes)

    logger.info(f' (6.2) learning rate scheduler')
    lr_scheduler = get_scheduler(lr_scheduler,
                                 optimizer=optimizer,
                                 num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
                                 num_training_steps=max_train_steps * gradient_accumulation_steps)

    logger.info(f'\n step 7. model to accelerator')
    student_unet, optimizer, train_dataloader,lr_scheduler = accelerator.prepare(student_unet,
                                                                                 optimizer,
                                                                                 train_dataloader,
                                                                                 lr_scheduler)
    print(f' *** len of train_dataloader: {len(train_dataloader)}')
    print(f' *** gradient_accumulation_steps : {gradient_accumulation_steps}')
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    print(f'num_update_steps_per_epoch: {num_update_steps_per_epoch}')
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    
    
    logger.info(f'\n step 8. Train')
    total_batch_size = train_batch_size * num_processes * gradient_accumulation_steps

    if is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Num Epochs = {num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not is_main_process)
    progress_bar.set_description("Steps")

    prompt = "A video of a woman, having a selfie"
    negative_prompt = "ImgFixerPre0.3, glowing face, bad proportions, blurry, blurred composition, low resolution, bad, ugly, bad composition, terrible, 3d, render, comic, manga, flat, watermark, signature, worst quality, low quality, normal quality, lowres, simple background, inaccurate limb, extra fingers, fewer fingers, missing fingers, extra arms, extra legs, inaccurate eyes, bad composition, bad anatomy, error, extra digit, fewer digits, cinnadust, cropped, low res, worst quality, low quality, normal quality, jpeg artifacts, extra digit, fewer digits, trademark, watermark, artist's name, username, signature, text, words, human, blurry, blurred composition, blurry foreground, blurry background"

    # Support mixed-precision training
    #scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None
    for epoch in range(first_epoch, num_train_epochs):
        
        #student_unet = accelerator.prepare(student_unet)
        student_unet.train()

        for step, batch in enumerate(train_dataloader):

            if args.motion_control:
                student_motion_controller.reset()
                teacher_motion_controller.reset()
                student_motion_controller.is_eval = False
                teacher_motion_controller.is_eval = False

            if cfg_random_null_text:
                batch['text'] = [name if random.random() > cfg_random_null_text_ratio else "" for name in batch['text']]

            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                if not image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value[None, ...]
                        save_folder = os.path.join(f'{output_dir}/sanity_check')
                        print(f' *** save_folder: {save_folder}')
                        os.makedirs(save_folder, exist_ok=True)
                        file_name = '-'.join(text.replace('/', '').split()[:10]) # about caption
                        file_name = f'{file_name}.gif'
                        save_videos_grid(pixel_value,
                                         os.path.join(save_folder, file_name),
                                         rescale=True)
                else:
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value / 2. + 0.5
                        torchvision.utils.save_image(pixel_value,
                                                     f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}.png")

            pixel_values = batch["pixel_values"]
            video_length = pixel_values.shape[1]
            with torch.no_grad():
                if not image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                    #with torch.cuda.amp.autocast():
                    # -------------------------------------------------------------------------------------------------------------------------
                    # how can here be out of memory ?
                    with torch.no_grad():
                        latents = vae.encode(pixel_values).latent_dist
                        latents = latents.sample()
                        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                else:
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()
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
                prompt_ids = tokenizer(batch['text'], max_length=tokenizer.model_max_length, padding="max_length",
                                        truncation=True, return_tensors="pt").input_ids.to(latents.device)
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
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                # with mixed_precision_training :
                teacher_model_pred = teacher_unet(noisy_latents, timesteps, encoder_hidden_states).sample
                if args.motion_control:
                    t_hdict = teacher_motion_controller.layerwise_hidden_dict  #
                    teacher_motion_controller.reset()
                    teacher_motion_controller.layerwise_hidden_dict = {}

                # student unet problem !!
                student_model_pred = student_unet(noisy_latents, timesteps, encoder_hidden_states).sample
                if args.motion_control:
                    s_hdict = student_motion_controller.layerwise_hidden_dict  #
                    student_motion_controller.reset()
                    student_motion_controller.layerwise_hidden_dict = {}

                # [1] task loss
                loss_vlb = F.mse_loss(student_model_pred.float(), target.float(), reduction="mean")
                # [2] matching loss
                loss_distill = F.mse_loss(student_model_pred.float(), teacher_model_pred.float(), reduction="mean")
                # [3] feature matching loss
                loss = loss_vlb + loss_distill

                if args.motion_control :
                    loss_feature = 0
                    for layer_name in s_hdict.keys():
                        s_h = s_hdict[layer_name]
                        t_h = t_hdict[layer_name]
                        for s_h_, t_h_ in zip(s_h, t_h):
                            loss_feature += F.mse_loss(s_h_.float(), t_h_.float(), reduction="mean")
                    student_motion_controller.reset()
                    teacher_motion_controller.reset()
                    loss = args.vlb_weight * loss_vlb + args.distill_weight * loss_distill + args.loss_feature_weight * loss_feature


            # ----------------------------------------------------------------------------------------------------
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            global_step += 1
            if is_main_process and (not is_debug) and use_wandb:
                wandb.log({"train_loss": loss.item()}, step=global_step)

            if args.motion_control:
                student_motion_controller.reset()
                teacher_motion_controller.reset()

        if args.motion_control:
            student_motion_controller.reset()
            teacher_motion_controller.reset()

        # ---------------------------------------------------------------------------------------------------- #
        # [1] loss logging
        accelerator.wait_for_everyone()
        if use_wandb and is_main_process:
            # [1.1] video save
            logger.info(f' * make validation pipeline')
            # ---------------------------------------------------------------------------------------------------------------- #
            # teacher_pipe.enable_vae_slicing()
            if args.motion_control:
                teacher_motion_controller.is_eval = True
                student_motion_controller.is_eval = True
            # ---------------------------------------------------------------------------------------------------------------- #
            validation_pipeline = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism",
                                                                      unet=accelerator.unwrap_model(student_unet),
                                                                      motion_adapter=student_adapter,)
            #validation_pipeline = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism",
            #                                                          unet=teacher_unet,
            #                                                          motion_adapter=teacher_adapter,)
            # ---------------------------------------------------------------------------------------------------------------- #
            #unwrap_unet = accelerator.unwrap_model(student_unet)
            #validation_pipeline = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism",
            #                                                          unet=unwrap_unet)
            validation_pipeline.scheduler = noise_scheduler
            validation_pipeline.to(accelerator.device, dtype=weight_dtype)
            #student_pipe.enable_vae_slicing()
            #student_motion_controller.is_eval = True
            with torch.no_grad():
            # ---------------------------------------------------------------------------------------------------------------- #
                output = validation_pipeline(prompt=prompt,
                                             negative_prompt=negative_prompt,
                                             ip_adapter_image=None,  # ip_adapter_image
                                             num_frames=args.num_frames,  # length = 48
                                             guidance_scale=guidance_scale,
                                             num_inference_steps=inference_step,
                                             generator=torch.Generator("cpu").manual_seed(0),
                                             motion_controller=student_motion_controller,
                                             window_size=window_size,).frames[0]
            if args.motion_control:
                student_motion_controller.reset()
                teacher_motion_controller.reset()
            fps = 10
            # ----------------------------
            video_save_dir = os.path.join(f"{output_dir}/samples", f"{name}_epoch_{epoch}.gif")
            export_to_gif(output, video_save_dir)
            wandb.log({"video": wandb.Video(data_or_path=video_save_dir, caption=f'epoch_{epoch}', fps=fps)})
            print(f' delete validation_pipeline')
            del validation_pipeline
        # ---------------------------------------------------------------------------------------------------- #
        # [2] save checkpoint
        if is_main_process :
            unwrapped_model = accelerator.unwrap_model(student_unet)
            # [1] save model config
            unwrapped_model.save_config(os.path.join(output_dir, f"checkpoints"))
            # [2] save model state
            torch.save({"epoch": epoch, "global_step": global_step, "state_dict": unwrapped_model.state_dict()},
                       os.path.join(output_dir, f"checkpoints/checkpoint_epoch{epoch}.ckpt"))
            #logging.info(f"Saved state to {save_path} (global_step: {global_step})")
        if args.motion_control:
            student_motion_controller.reset()
            teacher_motion_controller.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument('--project', type=str, default='video_distill')
    parser.add_argument('--sub_folder_name', type=str, default='result_sy')
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision",default = 'fp16')
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