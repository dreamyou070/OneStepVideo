import torch
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif, load_image
from PIL import Image
import argparse, os
import shutil
import yaml
from utils.layer_dictionary import find_layer_name
from masactrl.masactrl_utils import (regiter_attention_editor_diffusers,
                                     regiter_motion_attention_editor_diffusers)
from masactrl.masactrl import MutualSelfAttentionControl, MutualMotionAttentionControl
import logging
from datetime import datetime
import wandb
def main(args) :

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.info(f'\n step 1. wandb start')
    present_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = f'{args.save_base_dir}_{present_time}'
    wandb.init(project=args.project,
               name=f'{save_dir}')

    logger.info(f'\n step 2. make Motion Base Pipeline with LCM Scheduler')
    # AnimateLCM_sd15_t2v.ckpt
    # AnimateLCM_sd15_t2v_lora.ckpt
    logger.info(f' (2.1) motion adapter')
    student_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=torch.float16)
    student_pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=student_adapter,
                                                      torch_dtype=torch.float16)
    student_pipe.scheduler = LCMScheduler.from_config(student_pipe.scheduler.config, beta_schedule="linear")
    logger.info(f' (2.2) LCM Lora')
    student_pipe.load_lora_weights("wangfuyun/AnimateLCM",
                           weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
    #pipe.load_lora_weights("guoyww/animatediff-motion-lora-zoom-in",
    #                       weight_name="diffusion_pytorch_model.safetensors",
    #                       adapter_name="motion-lora")
    #pipe.set_adapters(["lcm-lora", "motion-lora"], adapter_weights=[0.8, 1.2])
    #pipe.set_adapters(["lcm-lora"], adapter_weights=[0.8])

    logger.info(f' (2.3) (image condition) IP-Adapter')
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    pipe.set_ip_adapter_scale(0.6)

    logger.info(f'\n step 3. set controller')
    test_file_dir = r'__assets__/test_single.txt'
    with open(test_file_dir, 'r') as f:
        datas = f.readlines()
    logger.info(f' (3.1) inference args')
    unet = pipe.unet
    inference_steps = [args.inference_steps] # 6
    guidance_scales = [1.5]
    ip_adapter_scales = [0.6]
    logger.info(f' (3.2) self controller')
    if args.self_control :
        self_controller = MutualSelfAttentionControl(guidance_scale =guidance_scales[0], frame_num = 16,)
        regiter_attention_editor_diffusers(unet, self_controller)
    logger.info(f' (3.3) motion controller')
    motion_controller = None
    window_size = args.window_size
    if args.motion_control :
        skip_layers = find_layer_name(args.skip_layers)
        motion_controller = MutualMotionAttentionControl(guidance_scale =guidance_scales[0],
                                                         frame_num = 16,
                                                         full_attention=args.full_attention,
                                                         window_attention=args.window_attention,
                                                         window_size=window_size,
                                                         total_frame_num=args.num_frames,
                                                         skip_layers = skip_layers) # 32
        regiter_motion_attention_editor_diffusers(unet, motion_controller)

    logger.info(f'\n step 4. Inference')
    for inference_step in inference_steps:
        save_folder = os.path.join('result', save_dir)
        if args.motion_control :
            for layer in args.skip_layers :
                save_folder += f'_{layer}'
        os.makedirs(save_folder)

        for guidance_scale in guidance_scales:
            for ip_adapter_scale in ip_adapter_scales:
                pipe.set_ip_adapter_scale(ip_adapter_scale)
                for data in datas:
                    image_dir, prompt = data.split('||')
                    image_dir = f'__assets__/imgs/{image_dir}'
                    name = os.path.splitext(os.path.split(image_dir)[-1])[0]
                    print(f' (2) n_prompt')
                    negative_prompt = args.n_prompt
                    print(f' (3) image prompt')
                    ip_adapter_image = Image.open(image_dir).convert("RGB")
                    pipe.enable_vae_slicing()
                    #pipe.enable_model_cpu_offload()
                    pipe.to('cuda')
                    start_time = datetime.now()

                    print(f' inference with motion_control !')
                    print(f'motion_controller: {motion_controller}')
                    output = pipe(prompt=prompt,
                                  negative_prompt=negative_prompt,
                                  ip_adapter_image=ip_adapter_image, # ip_adapter_image
                                  num_frames=args.num_frames, # length = 48
                                  guidance_scale=guidance_scale,
                                  num_inference_steps=inference_step,
                                  generator=torch.Generator("cpu").manual_seed(0),
                                  window_size = window_size, # window_size = 8
                                  motion_controller = motion_controller,
                                  save_base_folder = save_folder)
                    total_video = output[1]
                    output = output[0]

                    end_time = datetime.now()
                    take_time = end_time - start_time
                    frames = output.frames
                    #for f, frame in enumerate(frames) :
                    #    export_to_gif(frame[0],
                    #                  os.path.join(savedir, f"{name}_frame_{f}-num_frame_{args.num_frames}_{inference_step}_guidance_scale_{guidance_scale}_ip_adapter_scale_{ip_adapter_scale}.gif"))
                    shutil.copy(image_dir, os.path.join(save_folder, f"{name}_origin.jpg"))
                    # [2] save all frame
                    frames = total_video[0]
                    #export_to_gif(frames, os.path.join())
                    fps = 10
                    frames[0].save(os.path.join(save_folder,f"{name}_full.gif"),
                                   save_all=True,
                                   append_images=frames[1:],
                                   optimize=False,
                                   duration=1000 // fps,
                                   loop=0)

                    wandb.log({"video":
                                   wandb.Video(data_or_path=os.path.join(save_folder, f"{name}_full.gif"),caption='Test', fps=fps)})
                    with open(os.path.join(save_folder, f"{name}_{inference_step}_guidance_scale_{guidance_scale}_ip_adapter_scale_{ip_adapter_scale}.txt"), 'w') as f:
                        f.write(f'prompt: {prompt}\n')
                        f.write(f'negative_prompt: {negative_prompt}\n')
                        f.write(f'take_time: {take_time}\n')

            unet = pipe.unet
            vae = pipe.vae
            text_encoder = pipe.text_encoder
            unet_params_num = sum(p.numel() for p in unet.parameters())
            vae_params_num = sum(p.numel() for p in vae.parameters())
            text_encoder_params_num = sum(p.numel() for p in text_encoder.parameters())
            total_params_num = unet_params_num + vae_params_num + text_encoder_params_num
            print(f'Unet params num: {unet_params_num}')
            print(f'VAE params num: {vae_params_num}')
            print(f'Text encoder params num: {text_encoder_params_num}')
            print(f'Total params num: {total_params_num}')
            with open(os.path.join(save_folder, "params.txt"), 'w') as f:
                f.write(f'Unet params num: {unet_params_num}\n')
                f.write(f'VAE params num: {vae_params_num}\n')
                f.write(f'Text encoder params num: {text_encoder_params_num}\n')
                f.write(f'Total params num: {total_params_num}\n')




if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='t2v_inference')
    parser.add_argument('--project', type=str, default="video_test")
    parser.add_argument("--save_base_dir", type=str, )
    parser.add_argument('--prompt', type=str,
                        default="A space rocket with trails of smoke behind it launching into space from the desert, 4k, high resolution")
    parser.add_argument('--n_prompt', type=str,
                        default="bad quality, worse quality, low resolution")
    parser.add_argument('--image_dir', type=str, default="__assets__/imgs/space_rocket.jpg")
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--full_attention', action='store_true')
    parser.add_argument('--window_attention', action='store_true')
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--motion_control', action='store_true')
    parser.add_argument('--inference_steps', type=int, default=6)
    parser.add_argument('--self_control', action='store_true')
    from utils import arg_as_list
    parser.add_argument('--skip_layers', type=arg_as_list)
    args = parser.parse_args()
    main(args)