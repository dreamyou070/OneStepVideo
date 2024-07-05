import torch
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif, load_image
from PIL import Image
import argparse, datetime, os
import shutil
import yaml
from masactrl.masactrl_utils import (regiter_attention_editor_diffusers,
                                     regiter_motion_attention_editor_diffusers)
from masactrl.masactrl import MutualSelfAttentionControl, MutualMotionAttentionControl
from diffusers import StableDiffusionPipeline

def main(args):

    print(f'\n step 1. make Motion Base Pipeline with LCM Scheduler')
    pipe = StableDiffusionPipeline.from_pretrained(args.model_path_diffusion,
                                                       torch_dtype=torch.float16)
    device = 'cuda'
    unet = pipe.unet

    print(f' \n step 4. Inference')
    print(f' (0) save dir')
    # print(f' (1) prompt')
    # prompt = args.prompt
    test_file_dir = r'__assets__/test.txt'
    with open(test_file_dir, 'r') as f:
        datas = f.readlines()

    inference_steps = [args.inference_steps]  # 6
    guidance_scales = [1.5]
    ip_adapter_scales = [0.6]
    if args.self_control:
        self_controller = MutualSelfAttentionControl(guidance_scale=guidance_scales[0],
                                                     frame_num=16, )
        regiter_attention_editor_diffusers(unet, self_controller)
    #########################################################################################
    motion_controler = None
    window_size = args.window_size

    if args.motion_control:
        motion_controler = MutualMotionAttentionControl(guidance_scale=guidance_scales[0],
                                                        frame_num=16,
                                                        full_attention=args.full_attention,
                                                        window_attention=args.window_attention,
                                                        window_size=window_size,
                                                        total_frame_num=args.num_frames)  # 32
        regiter_motion_attention_editor_diffusers(unet, motion_controler)

    for inference_step in inference_steps:
        time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        savedir = f"result/{time_str}-infsteps_{inference_step}_window_size_{window_size}_group_query"
        os.makedirs(savedir)

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
                    # pipe.enable_model_cpu_offload()
                    pipe.to('cuda')
                    start_time = datetime.datetime.now()
                    image = pipe(prompt=prompt,
                                  negative_prompt=negative_prompt,
                                  guidance_scale=guidance_scale,
                                  num_inference_steps=inference_step,
                                  generator=torch.Generator("cpu").manual_seed(0),).images
                    shutil.copy(image_dir, os.path.join(savedir, f"{name}_origin.jpg"))
                    # [2] save image
                    image = image[0]
                    print(f'type of image : {type(image)}')

                    image.save(os.path.join(savedir,
                                            f"{name}_{inference_step}_guidance_scale_{guidance_scale}_ip_adapter_scale_{ip_adapter_scale}.jpg"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='t2v_inference')
    parser.add_argument('--model_path_diffusion', type=str,)
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
    args = parser.parse_args()
    main(args)