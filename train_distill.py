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
                        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO, )

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



    logger.info(f'\n step 2. preparing accelerator')
    torch.manual_seed(args.seed)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    folder_name = args.sub_folder_name
    folder_name = folder_name + "_" + datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    # [2]
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
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

    logger.info(f'\n step 3. saving dir')
    if is_main_process:
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        with open(os.path.join(output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

    logger.info(f'\n step 4. set model')
    logger.info(f' (4.1) teacher')
    teacher_adapter = MotionAdapter.from_pretrained(args.teacher_motion_model_dir,torch_dtpe = weight_dtype)
    teacher_adapter_state_dict = teacher_adapter.state_dict()
    teacher_pipe = AnimateDiffPipeline.from_pretrained(args.pretrained_model_path,
                                                       motion_adapter=teacher_adapter,
                                                       torch_dtpe=weight_dtype)
    noise_scheduler = LCMScheduler.from_config(teacher_pipe.scheduler.config, beta_schedule="linear")
    teacher_pipe.scheduler = noise_scheduler
    teacher_pipe.to(accelerator.device, dtype=weight_dtype)
    logger.info(f' (4.2) student')
    student_adapter = MotionAdapter.from_pretrained(args.teacher_motion_model_dir,
                                                    torch_dtpe = weight_dtype)
    student_config = student_adapter.config
    if args.random_init :
        student_adapter = MotionAdapter(**student_config)
    student_pipe = AnimateDiffPipeline.from_pretrained(args.pretrained_model_path,
                                                       motion_adapter=student_adapter,
                                                       torch_dtpe = weight_dtype)
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
        #  ['up_blocks_3_motion_modules_0']
        skip_layers = find_layer_name(args.skip_layers)
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
    teacher_unet.to(accelerator.device, dtype=weight_dtype)
    student_unet.requires_grad_(False)
    skil_layers = []
    for name, param in student_unet.named_parameters():
        for trainable_module_name in args.trainable_modules:
            if trainable_module_name in name :
                un = 0
                name_dot = name.replace('.', '_')
                for skip_layer in skip_layers :
                    if skip_layer in name_dot :
                        un += 1
                if un == 0 :
                    param.requires_grad = True
                else :
                    # skip layer
                    skil_layers.append(name)
                    #print(f' skip layer name = {name}')
                    param.requires_grad = False
            else :
                param.requires_grad = False
    original_state_dict = student_unet.state_dict()
    new_state_dict = {}
    for key in original_state_dict.keys():
        if key in skil_layers :
            new_state_dict[key] = original_state_dict[key] * 100
        else :
            new_state_dict[key] = original_state_dict[key]
    student_unet.load_state_dict(new_state_dict)

    trainable_params = list(filter(lambda p: p.requires_grad, student_unet.parameters())) # len 333
    optimizer = torch.optim.AdamW(trainable_params,
                                  lr=args.learning_rate,
                                  betas=(args.adam_beta1, args.adam_beta2),
                                  weight_decay=args.adam_weight_decay,
                                  eps=args.adam_epsilon,
                                  foreach = False)
    if args.gradient_checkpointing:
        student_unet.enable_gradient_checkpointing()

    logger.info(f'\n step 5. Get the training dataset')
    train_dataset = DistillWebVid10M(csv_path=args.csv_path,
                                     video_folder=args.video_folder,
                                     sample_size=512,
                                     sample_stride=4,
                                     sample_n_frames=args.sample_n_frames,
                                     is_image=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=1, #train_batch_size,
                                                   shuffle=True,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True,
                                                   drop_last=True,)

    logger.info(f'\n step 6. training argument')
    if args.max_train_steps == -1:
        assert args.max_train_epoch != -1
        max_train_steps = args.max_train_epoch * len(train_dataloader)
    if args.checkpointing_steps == -1:
        assert args.checkpointing_epochs != -1
        checkpointing_steps = args.checkpointing_epochs * len(train_dataloader)

    num_processes = 1
    if args.scale_lr:
        num_processes = 1
        learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * num_processes)

    logger.info(f' (6.2) learning rate scheduler')
    lr_scheduler = get_scheduler(args.lr_scheduler,
                                 optimizer=optimizer,
                                 num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
                                 num_training_steps=max_train_steps * args.gradient_accumulation_steps)

    logger.info(f'\n step 7. model to accelerator')
    student_adapter, student_unet, optimizer, train_dataloader,lr_scheduler = accelerator.prepare(student_adapter, student_unet,
                                                                                                  optimizer, train_dataloader,
                                                                                                  lr_scheduler)

    student_motion_controller = MutualMotionAttentionControl(guidance_scale=guidance_scale,
                                                             frame_num=16,
                                                             full_attention=args.full_attention,
                                                             window_attention=args.window_attention,
                                                             window_size=window_size,
                                                             total_frame_num=args.num_frames,
                                                             skip_layers=skip_layers,
                                                             is_teacher=False)  # 32
    regiter_motion_attention_editor_diffusers(student_unet.modules(), student_motion_controller)

    print(f'student_unet skip layers = {student_motion_controller.skip_layers}')



    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) # 100
    num_train_epochs = 100
    logger.info(f'\n step 8. Train')
    total_batch_size = args.train_batch_size * num_processes * args.gradient_accumulation_steps

    if is_main_process:
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
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not is_main_process)
    progress_bar.set_description("Steps")

    prompt = "A video of a woman, having a selfie"
    call_lcm = 0
    for epoch in range(first_epoch, num_train_epochs):
        # ---------------------------------------------------------------------------------------------------- #
        # [1] loss logging
        accelerator.wait_for_everyone()
        # ---------------------------------------------------------------------------------------------------- #
        # [2] model saving
        """
        if is_main_process :
            unwrapped_model = accelerator.unwrap_model(student_unet)
            trained_value = unwrapped_model.state_dict()
            original_value = teacher_adapter_state_dict
            save_state_dict= {}
            for trained_key, trained_value in trained_value.items():
                if 'motion' in trained_key and trained_key in original_value.keys():
                    save_state_dict[trained_key] = trained_value.to('cpu')
            torch.save(save_state_dict, os.path.join(output_dir, f"checkpoints/checkpoint_epoch{epoch}.pt"))
            # [3] validation
            del unwrapped_model
        """

        # ---------------------------------------------------------------------------------------------------- #
        # [4] training
        student_unet.train()
        print(f' before train, student_unet = {student_unet.__class__.__name__}')
        for step, batch in enumerate(train_dataloader):
            if args.motion_control:
                student_motion_controller.reset()
                teacher_motion_controller.reset()
            if args.cfg_random_null_text:
                batch['text'] = [name if random.random() > args.cfg_random_null_text_ratio else "" for name in batch['text']]
            # Data batch sanity check
            """
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
            """
            pixel_values = batch["pixel_values"]
            video_length = pixel_values.shape[1]
            with torch.no_grad():
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * 0.18215
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
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

            with torch.cuda.amp.autocast(enabled=args.mixed_precision_training):
                teacher_model_pred = teacher_unet(noisy_latents, timesteps,encoder_hidden_states).sample
                if args.motion_control:
                    teacher_motion_controller.reset()
                #if args.motion_control:
                #    t_hdict = teacher_motion_controller.layerwise_hidden_dict  #
                #    teacher_motion_controller.reset()
                #    teacher_motion_controller.layerwise_hidden_dict = {}
                # student unet problem !!
                #if args.motion_control:
                #    student_unet = accelerator.unwrap_model(student_unet)
                #    student_motion_controller = MutualMotionAttentionControl(guidance_scale=guidance_scale,
                #                                                             frame_num=16,
                #                                                             full_attention=args.full_attention,
                #                                                             window_attention=args.window_attention,
                #                                                             window_size=window_size,
                #                                                             total_frame_num=args.num_frames,
                #                                                             skip_layers=skip_layers,
                #                                                             is_teacher=False)  # 32
                #    skip_layers = student_motion_controller.skip_layers
                #    regiter_motion_attention_editor_diffusers(student_unet, student_motion_controller)
                #    student_unet, student_motion_controller = accelerator.prepare(student_unet, student_motion_controller)
                student_model_pred = student_unet(noisy_latents, timesteps, encoder_hidden_states).sample
                if args.motion_control:
                    student_motion_controller.reset()
                #if args.motion_control:
                #    s_hdict = student_motion_controller.layerwise_hidden_dict  #
                #    student_motion_controller.layerwise_hidden_dict = {}
                # [1] task loss
                loss_vlb = F.mse_loss(student_model_pred.float(), target.float(), reduction="mean")
                loss_distill = F.mse_loss(student_model_pred.float(), teacher_model_pred.float(), reduction="mean")
                loss = loss_vlb + loss_distill
                torch.nn.utils.clip_grad_norm_(trainable_params,args.max_grad_norm)
                ####################################################################################################
                """
                if args.motion_control :
                    loss_feature = 0
                    for layer_name in s_hdict.keys():
                        s_h = s_hdict[layer_name]
                        t_h = t_hdict[layer_name]
                        for s_h_, t_h_ in zip(s_h, t_h):
                            loss_feature += F.mse_loss(s_h_.float(), t_h_.float(), reduction="mean")
                    student_motion_controller.reset()
                    teacher_motion_controller.reset()
                """
                # loss
                #loss = args.vlb_weight * loss_vlb + \
                #       args.distill_weight * loss_distill + \
                #       args.loss_feature_weight * loss_feature

            # skip_layers
            # ----------------------------------------------------------------------------------------------------
            # Backward pass
            # gradient only not skip_layers
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            global_step += 1
            if is_main_process and args.use_wandb:
                wandb.log({"train_loss": loss.item()}, step=global_step)

            if args.motion_control:
                student_motion_controller.reset()
                teacher_motion_controller.reset()

            ################################################################################

            if args.use_wandb and is_main_process and step % 2 == 0  :
                logger.info(f' ******************************************* ')
                logger.info(f' * step = {step} : make validation pipeline')
                call_lcm += 1
                unwrapped_model = accelerator.unwrap_model(student_unet).eval()
                regiter_motion_attention_editor_diffusers(unwrapped_model, student_motion_controller)
                validation_pipeline = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism",
                                                                          unet=unwrapped_model,)
                                                                          #motion_adapter=student_adapter, )
                validation_pipeline.scheduler = noise_scheduler
                if call_lcm == 1 :
                    validation_pipeline.load_lora_weights("wangfuyun/AnimateLCM",
                                                          weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                                                          adapter_name="lcm-lora")
                validation_pipeline.set_adapters(["lcm-lora"], [0.8])
                validation_pipeline.to(accelerator.device, dtype=weight_dtype)
                with torch.no_grad():
                    output = validation_pipeline(prompt=prompt,
                                                 negative_prompt="bad quality, worse quality, low resolution",
                                                 num_frames=args.num_frames,  # length = 16
                                                 guidance_scale=guidance_scale, # 1.5
                                                 num_inference_steps=inference_step,
                                                 generator=torch.Generator("cpu").manual_seed(0)).frames[0]
                if args.motion_control:
                    student_motion_controller.reset()
                    teacher_motion_controller.reset()
                fps = 10
                video_save_dir = os.path.join(f"{output_dir}/samples", f"{name}_epoch_{epoch}.gif")
                export_to_gif(output, video_save_dir)
                wandb.log({"video": wandb.Video(data_or_path=video_save_dir, caption=f'step_{step}', fps=fps)})
                print(f' delete validation_pipeline')
                del validation_pipeline, unwrapped_model

        if args.motion_control:
            student_motion_controller.reset()
            teacher_motion_controller.reset()
        # ---------------------------------------------------------------------------------------------------- #
        # [2] save checkpoint
        accelerator.wait_for_everyone()


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
    parser.add_argument('--trainable_modules', type=arg_as_list , default="['motion_modules.']")
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