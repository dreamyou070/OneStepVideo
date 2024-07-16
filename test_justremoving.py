import torch
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif, export_to_video
import os
from attn.masactrl import MutualMotionAttentionControl
from attn.masactrl_utils import regiter_motion_attention_editor_diffusers
import time
from utils.layer_dictionary import find_layer_name
import argparse
from utils import arg_as_list

def main(args) :

    print(f' \n step 0. inference condition')
    guidance_scales = [1.5]

    print(f' \n step 1. make Motion Base Pipeline with LCM Scheduler')
    adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=torch.float16)
    pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16)
    unet = pipe.unet
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

    print(f'\n step 2. skip layer applying')
    skip_layers, skip_layers_dot = find_layer_name(args.skip_layers)

    motion_controller = MutualMotionAttentionControl(guidance_scale=guidance_scales[0],
                                                     frame_num=16,
                                                     full_attention=True,
                                                     window_attention=False,
                                                     window_size=16,
                                                     total_frame_num=16,
                                                     is_teacher=args.is_teacher,
                                                     skip_layers=skip_layers)  # 32
    regiter_motion_attention_editor_diffusers(unet, motion_controller)
    pipe.unet = unet

    print(f'\n step 3. LCM Lora')
    pipe.load_lora_weights("wangfuyun/AnimateLCM",
                           weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                           adapter_name="lcm-lora")
    pipe.set_adapters(["lcm-lora"], [0.8])
    pipe.enable_vae_slicing()
    pipe.to('cuda')

    print(f' \n step 4. save_base_dir')
    num_frames = 16
    save_base_dir = 'result'
    os.makedirs(save_base_dir, exist_ok=True)
    save_base_dir = 'result/not_distill_just_removing'
    os.makedirs(save_base_dir, exist_ok=True)
    save_folder = os.path.join(save_base_dir, args.sub_folder_name)
    os.makedirs(save_folder, exist_ok=True)

    print(f' \n step 3. inference test')
    prompt_dir = f'./configs/prompts/filtered_captions_val_{args.start_num}_{args.end_num}.txt'
    with open(prompt_dir, 'r') as f:
        prompts = f.readlines()
    guidance_scales = [1.5]
    num_inference_steps = [6]
    n_prompt = "bad quality, worse quality, low resolution"
    seeds = [0]
    iter = 0
    for p, prompt in enumerate(prompts):
        save_p = str((args.m - 1) * 60 + p).zfill(3)
        prompt_folder = os.path.join(save_base_dir, f'prompt_idx_{save_p}')
        print(f'prompt_folder = {prompt_folder}')
        os.makedirs(prompt_folder, exist_ok=True)
        # prompt setting
        with open(os.path.join(prompt_folder, 'prompt.txt'), 'w') as f:
            f.write(prompt)
        for guidance_scale in guidance_scales :
            for inference_scale in num_inference_steps :
                for seed in seeds :
                    iter += 1
                    start_time = time.time()
                    output = pipe(prompt=prompt,
                                  negative_prompt=n_prompt,
                                  num_frames=num_frames,
                                  guidance_scale=guidance_scale,
                                  num_inference_steps=inference_scale,
                                  generator=torch.Generator("cpu").manual_seed(seed), )
                    end_time = time.time()
                    elapse_time = end_time - start_time
                    # [0] controller reset
                    motion_controller.reset()
                    # [1] save
                    frames = output.frames[0]
                    export_to_gif(frames, os.path.join(save_folder, f'sample_{str(iter).zfill(3)}.gif'))
                    export_to_video(frames, os.path.join(save_folder, f'sample_{str(iter).zfill(3)}.mp4'))
                    # [2] text save
                    text_dir = os.path.join(save_folder, f'sample_{str(iter).zfill(3)}.txt')
                    with open(text_dir, 'w') as f:
                        f.write(f'prompt : {prompt}\n')
                        f.write(f'inference_step : {inference_scale}\n')
                        f.write(f'guidance_scale : {guidance_scale}\n')
                        f.write(f'seed : {seed}\n')
                        f.write(f'elapse_time : {elapse_time}\n')

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='t2v_inference')
    parser.add_argument('--m', type=int,default=1)
    parser.add_argument('--is_teacher', action='store_true')
    parser.add_argument('--start_num', type=int, default=100)
    parser.add_argument('--end_num', type=int, default=140)

    parser.add_argument('--skip_layers', type=arg_as_list)
    parser.add_argument('--sub_folder_name', type=str)
    parser.add_argument('--sample_n_frames', type=int, default=16)
    parser.add_argument('--inference_step', type=int, default=6)
    parser.add_argument('--guidance_scale', type=float, default=1.5)
    parser.add_argument('--motion_control', action='store_true')
    args = parser.parse_args()
    main(args)