import torch
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif, load_image
from PIL import Image
import argparse, datetime, os
import shutil
import yaml
def main(args) :

    print(f'\n step 1. make Motion Base Pipeline with LCM Scheduler')
    adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=torch.float16)
    pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter,
                                               torch_dtype=torch.float16)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

    print(f'\n step 2. LCM Lora')
    pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
    pipe.set_adapters(["lcm-lora"], [0.8])

    print(f' \n step 3. (image condition) IP-Adapter')
    #pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    #pipe.set_ip_adapter_scale(0.6)

    print(f' \n step 4. Inference')
    print(f' (0) save dir')
    # print(f' (1) prompt')
    # prompt = args.prompt
    test_file_dir = r'__assets__/test.txt'
    with open(test_file_dir, 'r') as f:
        datas = f.readlines()

    inference_steps = [6]
    guidance_scales = [1.5]
    ip_adapter_scales = [0.6]

    for inference_step in inference_steps:
        name = os.path.splitext(os.path.split(args.image_dir)[-1])[0]
        time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        savedir = f"result/{time_str}-infsteps_{inference_step}-num_frames_{args.num_frames}"
        os.makedirs(savedir)

        for guidance_scale in guidance_scales:
            for ip_adapter_scale in ip_adapter_scales:
                pipe.set_ip_adapter_scale(ip_adapter_scale)
                print(f'datas: {datas}')
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
                    start_time = datetime.datetime.now()
                    output = pipe(prompt=prompt,
                                  negative_prompt=negative_prompt,
                                  # ip_adapter_image=ip_adapter_image, # ip_adapter_image
                                  num_frames=args.num_frames,
                                  guidance_scale=guidance_scale,
                                  num_inference_steps=inference_step,
                                  window_size = 16, #args.window_size,
                                  generator=torch.Generator("cpu").manual_seed(0),
                                  save_base_folder = savedir,)
                    end_time = datetime.datetime.now()
                    take_time = end_time - start_time
                    frames = output.frames[0]
                    save_name = os.path.join(savedir, f"{name}_infsteos_{inference_step}_guidance_scale_{guidance_scale}_ip_adapter_scale_{ip_adapter_scale}.gif")
                    print(f'save_name: {save_name}')
                    export_to_gif(frames, save_name)
                    shutil.copy(image_dir, os.path.join(savedir, f"{name}_origin.jpg"))
                    with open(os.path.join(savedir, f"{name}_{inference_step}_guidance_scale_{guidance_scale}_ip_adapter_scale_{ip_adapter_scale}.txt"), 'w') as f:
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
    with open(os.path.join(savedir, "params.txt"), 'w') as f:
        f.write(f'Unet params num: {unet_params_num}\n')
        f.write(f'VAE params num: {vae_params_num}\n')
        f.write(f'Text encoder params num: {text_encoder_params_num}\n')
        f.write(f'Total params num: {total_params_num}\n')




if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='t2v_inference')
    parser.add_argument('--prompt', type=str,
                        default="A space rocket with trails of smoke behind it launching into space from the desert, 4k, high resolution")
    parser.add_argument('--n_prompt', type=str,
                        default="bad quality, worse quality, low resolution")
    parser.add_argument('--image_dir', type=str, default="__assets__/imgs/space_rocket.jpg")
    parser.add_argument('--num_frames', type=int, default=16)

    args = parser.parse_args()
    main(args)