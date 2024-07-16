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
        if  len(args.skip_layers) > 0:
            for i  in range(len(args.skip_layers)) :
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
    #print(f'weight_dtype: {weight_dtype}')
    #weight_dtype = torch.float16

    logger.info(f'\n step 3. saving dir')
    os.makedirs(f"{output_dir}/samples", exist_ok=True)
    os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    logger.info(f'\n step 4. set model')
    logger.info(f' (4.1) teacher')
    teacher_adapter = MotionAdapter.from_pretrained(args.teacher_motion_model_dir, torch_dtpe=weight_dtype)
    teacher_adapter_state_dict = teacher_adapter.state_dict()
    teacher_pipe = AnimateDiffPipeline.from_pretrained(args.pretrained_model_path, motion_adapter=teacher_adapter, torch_dtpe=weight_dtype)
    noise_scheduler = LCMScheduler.from_config(teacher_pipe.scheduler.config, beta_schedule="linear")
    teacher_pipe.scheduler = noise_scheduler

    logger.info(f' (4.2) student')
    student_adapter = MotionAdapter.from_pretrained(args.teacher_motion_model_dir,
                                                    torch_dtpe=weight_dtype).to(device, dtype=weight_dtype)
    student_config = student_adapter.config
    if args.random_init: # make another module
        student_adapter = MotionAdapter(**student_config)
    student_pipe = AnimateDiffPipeline.from_pretrained(args.pretrained_model_path,
                                                       motion_adapter=student_adapter,
                                                       torch_dtpe=weight_dtype)
    student_pipe.scheduler = noise_scheduler

    logger.info(f' (4.3) vae, text_encoder, unet')
    vae = teacher_pipe.vae
    tokenizer = teacher_pipe.tokenizer
    text_encoder = teacher_pipe.text_encoder
    teacher_unet = teacher_pipe.unet
    student_unet = student_pipe.unet
    """
    for name,param in student_unet.named_parameters():
        if 'motion' in name:
            print(f'[{name}] motion value = {param}')
            # up_blocks.3.motion_modules.2.transformer_blocks.0.norm3.bias
    """

    logger.info(f' (4.4) motion controller')
    window_size = 16
    guidance_scale = args.guidance_scale
    inference_step = args.inference_step
    student_motion_controller = None
    teacher_motion_controller = None
    if args.motion_control:
        student_motion_controller = MutualMotionAttentionControl(guidance_scale=guidance_scale,
                                                                 frame_num=16,
                                                                 full_attention=args.full_attention,
                                                                 window_attention=args.window_attention,
                                                                 window_size=window_size,
                                                                 total_frame_num=args.num_frames,
                                                                 skip_layers=skip_layers,
                                                                 is_teacher=False)  # 32
        regiter_motion_attention_editor_diffusers(student_unet, student_motion_controller)

        teacher_motion_controller = MutualMotionAttentionControl(guidance_scale=guidance_scale,
                                                                 frame_num=16,
                                                                 full_attention=args.full_attention,
                                                                 window_attention=args.window_attention,
                                                                 window_size=window_size,
                                                                 total_frame_num=args.num_frames,
                                                                 skip_layers=skip_layers,
                                                                 is_teacher=True)  # 32
        regiter_motion_attention_editor_diffusers(teacher_unet, teacher_motion_controller)

    logger.info(f' (4.5) other modules freeze')
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.to(device, dtype=weight_dtype)
    teacher_unet.requires_grad_(False)
    teacher_unet.to(device, dtype=weight_dtype)
    student_unet.to(device, dtype=weight_dtype)
    student_adapter = student_adapter.to(device, dtype=weight_dtype)
    print(f' [1] student_adapter: {student_adapter.device}')

    #################################################################################################################
    # [1] 3D Unet
    trainable_params = []
    student_unet.requires_grad_(False)
    for name, param in student_unet.named_parameters():
        if 'motion' in name:
            param.requires_grad = True
            if skip_layers_dot is not None:
                for skip_layer in skip_layers_dot:
                    if skip_layer in name :
                        param.requires_grad = False
                        break


    #################################################################################################################
    # [2] adapter
    student_adapter.requires_grad_(True)
    if len(skip_layers_dot) > 0 :
        for name, param in student_adapter.named_parameters():
            for skip_layer_dot in skip_layers_dot:
                if skip_layer_dot in name:
                    param.requires_grad = False
                    break

    for name, param in student_unet.named_parameters():
        if param.requires_grad :
            if 'norm' in name :
                param.data.fill_(1)
            trainable_params.append(param)

        # up_blocks.3.motion_modules.2.proj_out.bias

    print(f' [2] student_adapter: {student_adapter.device}')
    from utils.optimizer_module import AdamW
    optimizer = AdamW(trainable_params, #student_unet.parameters(),
                      lr=args.learning_rate,
                      betas=(args.adam_beta1, args.adam_beta2),
                      weight_decay=args.adam_weight_decay,
                      eps=args.adam_epsilon,
                      foreach=False)

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
                                                   batch_size=1,  # train_batch_size,
                                                   shuffle=True,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True,
                                                   drop_last=True, )

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
    lr_scheduler = get_scheduler(args.lr_scheduler, # 'constant'
                                 optimizer=optimizer,
                                 num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
                                 num_training_steps=max_train_steps * args.gradient_accumulation_steps)

    logger.info(f'\n step 7. model to accelerator')
    student_unet = student_unet.to(device, dtype=weight_dtype)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)  # 100
    num_train_epochs = 100

    logger.info(f'\n step 8. set animate pipeline')
    validation_pipeline = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism",
                                                              unet=student_unet,
                                                              vae=vae,
                                                              text_encoder=text_encoder,
                                                              motion_adapter=student_adapter, )
    validation_pipeline.scheduler = noise_scheduler


    logger.info(f'\n step 8. Train')
    total_batch_size = args.train_batch_size * num_processes * args.gradient_accumulation_steps
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

    prompt = "A video of a woman, having a selfie"
    call_lcm = 0
    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision_training else None
    print(f' [3] student_adapter: {student_adapter.device}')
    for epoch in range(first_epoch, num_train_epochs):

        if args.use_wandb:
            call_lcm += 1
            validation_pipeline.unet = student_unet
            #validation_pipeline.load_lora_weights("wangfuyun/AnimateLCM",
            #                                       weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
            #                                       adapter_name="lcm-lora")
            #validation_pipeline.set_adapters(["lcm-lora"], [0.8])
            #validation_pipeline.to(device, dtype=weight_dtype)
            with torch.no_grad():
                output = validation_pipeline(prompt=prompt,
                                             negative_prompt="bad quality, worse quality, low resolution",
                                             num_frames=args.num_frames,  # length = 16
                                             guidance_scale=guidance_scale,  # 1.5
                                             num_inference_steps=inference_step,
                                             generator=torch.Generator('cpu').manual_seed(0)).frames[0]
            validation_pipeline.unload_lora_weights()
            if args.motion_control:
                student_motion_controller.reset()
                teacher_motion_controller.reset()
            fps = 10
            video_save_dir = os.path.join(f"{output_dir}/samples", f"epoch_{epoch}.gif")
            export_to_gif(output, video_save_dir)
            wandb.log({"video": wandb.Video(data_or_path=video_save_dir, caption=f'epoch_{epoch}', fps=fps)})


        # ---------------------------------------------------------------------------------------------------- #
        # [2] model saving
        trained_value = student_unet.state_dict()
        save_state_dict= {}
        for trained_key, trained_value in trained_value.items():
            if 'motion' in trained_key :
                save_state_dict[trained_key] = trained_value.to('cpu')
        torch.save(save_state_dict, os.path.join(output_dir, f"checkpoints/checkpoint_epoch{epoch}.pt"))
        if epoch != 0 :
            before_epoch = epoch - 1

            before_state_dict = torch.load(os.path.join(output_dir, f"checkpoints/checkpoint_epoch{before_epoch}.pt"), map_location="cpu")
            present_state_dict = torch.load(os.path.join(output_dir, f"checkpoints/checkpoint_epoch{epoch}.pt"), map_location="cpu")
            for key, value in present_state_dict.items():
                present_state = present_state_dict[key]
                before_state = before_state_dict[key]
                if torch.equal(present_state, before_state):
                    print(f'epoch {epoch} {key} is equal')
        # ---------------------------------------------------------------------------------------------------- #
        """
        for name, param in student_unet.named_parameters():
            if param.requires_grad:
                print(f' [{name}] trainable ')
        """
        # ---------------------------------------------------------------------------------------------------- #
        # [4] training
        #student_unet.train()
        for step, batch in enumerate(train_dataloader):
            if args.motion_control:
                student_motion_controller.reset()
                teacher_motion_controller.reset()
            if args.cfg_random_null_text:
                batch['text'] = [name if random.random() > args.cfg_random_null_text_ratio else "" for name in
                                 batch['text']]
            # Data batch sanity check
            
            if epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                """
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
                    latents = vae.encode(pixel_values.to(device, dtype=weight_dtype)).latent_dist
                    latents = latents.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * 0.18215
            noise = torch.randn_like(latents).to(device, dtype=weight_dtype)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            with torch.no_grad():
                prompt_ids = tokenizer(batch['text'], max_length=tokenizer.model_max_length, padding="max_length",
                                       truncation=True, return_tensors="pt").input_ids.to(latents.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0].to(device, dtype=weight_dtype)

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            # Mixed-precision training

            with torch.no_grad():
                teacher_model_pred = teacher_unet(noisy_latents, timesteps, encoder_hidden_states).sample
                if args.motion_control:
                    t_hdict = teacher_motion_controller.layerwise_hidden_dict  #
                    teacher_motion_controller.reset()
                    teacher_motion_controller.layerwise_hidden_dict = {}
            
            #with torch.cuda.amp.autocast(enabled=args.mixed_precision_training):
            #print(f'noisy_latents = {noisy_latents.shape}')

            student_model_pred = student_unet(noisy_latents, timesteps, encoder_hidden_states).sample
            if args.motion_control:
                s_hdict = student_motion_controller.layerwise_hidden_dict
                student_motion_controller.layerwise_hidden_dict = {}
                loss_feature = 0
                for layer_name in s_hdict.keys():
                    s_h = s_hdict[layer_name]
                    t_h = t_hdict[layer_name]
                    for s_h_, t_h_ in zip(s_h, t_h):
                        loss_feature += F.mse_loss(s_h_.float(), t_h_.float(), reduction="mean")
                student_motion_controller.reset()
                teacher_motion_controller.reset()

            # [1] task loss
            loss_vlb = F.mse_loss(student_model_pred.float(), target.float(), reduction="mean")
            loss_distill = F.mse_loss(student_model_pred.float(), teacher_model_pred.float(), reduction="mean")
            loss = args.vlb_weight * loss_vlb + args.distill_weight * loss_distill
            if args.motion_control:
                loss = args.vlb_weight * loss_vlb + args.distill_weight * loss_distill + args.loss_feature_weight * loss_feature
            # backpropagation checking !
            # param.grad who calculate ?

            # --> param grad autograd
            print('=========================================')
            for name, param in student_unet.named_parameters():
                if name == 'up_blocks.2.motion_modules.1.transformer_blocks.0.ff.net.2.bias' :
                    a = param[0]
            print(f'ORDER: {global_step + 1}')
            print(f'[Before Backward]\nf{a}')
            # up_blocks.3.motion_modules.1.transformer_blocks.0.attn2.to_out.0.weight[0,0]
            optimizer.zero_grad()
            loss.backward()
            # check the gradient
            """
            for name, param in student_unet.named_parameters():
                if param.requires_grad:
                    print(f'[{name}] motion value = {param.grad}')
            """
            # how the optimizer to step

            # Backpropagate
            if args.mixed_precision_training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params,args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                #loss.backward()
                # Backprop check
                #for name, param in student_unet.named_parameters():
                #    if param.requires_grad:
                #        print(f'[{name}] motion value = {param.grad}')
                # optimizer params_grup grad checking
                # for name, param in optimizer.named_parameters():
                #     if param.requires_grad:
                #         print(f'(optimizer) [{name}] motion value = {param.grad}')
                #torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(student_unet.parameters(), args.max_grad_norm)
                ############################################
                # 매개변수 및 그라디언트 출력
                optimizer.step() #####################
                i = 0
                for name, param in student_unet.named_parameters():
                    if param.requires_grad:
                        #print(f'[{name}] value = {param.data}')
                        param.data = optimizer.param_groups[0]['params'][i]
                        # change from param to optimizer.param_groups[0]['params'][i]

                        #student_unet.named_parameters()[name] = optimizer.param_groups[0]['params'][i]
                        #par = student_unet.named_parameters()[name]
                        #print(f'[{name}] motion value = {par}')
                        i += 1



                for i, param_group in enumerate(optimizer.param_groups):
                    print(f"Param group {i}:")
                    for key, value in param_group.items():
                        if key == 'params':
                            print(f"  {key}: {[p.shape for p in value]}")  # 파라미터의 크기 출력



                for name, param in student_unet.named_parameters():
                    if name == 'up_blocks.2.motion_modules.1.transformer_blocks.0.ff.net.2.bias':
                        a = param[0]
                print(f'[Before Backward]\nf{a}')
                #param_groups = optimizer.param_groups

                post_check_dict = {}
                for name, param in student_unet.named_parameters():
                    if 'motion' in name:
                        post_check_dict[name] = param
                #for name in check_dict.keys():
                #    if not torch.equal(check_dict[name], post_check_dict[name]):
                #        print(f'epoch {epoch} step {step} {name} is ""not"" equal')
                        
            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1
            wandb.log({"train_loss": loss.item()}, step=global_step)
            wandb.log({"loss_vlb": loss_vlb.item()}, step=global_step)
            wandb.log({"loss_distill": loss_distill.item()}, step=global_step)
            if args.motion_control:
                if type(loss_feature) == torch.Tensor:
                    wandb.log({"loss_feature": loss_feature.item()}, step=global_step)

            if args.motion_control:
                student_motion_controller.reset()
                teacher_motion_controller.reset()

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