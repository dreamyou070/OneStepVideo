import torch
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif, export_to_video
import os
from masactrl.masactrl import MutualMotionAttentionControl
from masactrl.masactrl_utils import regiter_motion_attention_editor_diffusers
import time

layer_dict = {0: 'down_blocks_0_motion_modules_0',
              1: 'down_blocks_0_motion_modules_1',
              2: 'down_blocks_1_motion_modules_0',
              3: 'down_blocks_1_motion_modules_1',
              4: 'down_blocks_2_motion_modules_0',
              5: 'down_blocks_2_motion_modules_1',
              6: 'down_blocks_3_motion_modules_0',
              7: 'down_blocks_3_motion_modules_1',
              8: 'mid_block_motion_modules_0',
              9: 'up_blocks_0_motion_modules_0',
              10: 'up_blocks_0_motion_modules_1',
              11: 'up_blocks_0_motion_modules_2',
              12: 'up_blocks_1_motion_modules_0',
              13: 'up_blocks_1_motion_modules_1',
              14: 'up_blocks_1_motion_modules_2',
              15: 'up_blocks_2_motion_modules_0',
              16: 'up_blocks_2_motion_modules_1',
              17: 'up_blocks_2_motion_modules_2',
              18: 'up_blocks_3_motion_modules_0',
              19: 'up_blocks_3_motion_modules_1',
              20: 'up_blocks_3_motion_modules_2'}


def main(args) :

    print(f' \n step 1. make Motion Base Pipeline with LCM Scheduler')
    adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=torch.float16)
    #pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16)
    pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism",
                                               motion_adapter=adapter,
                                               torch_dtype=torch.float16)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
    print(f'\n step 2. LCM Lora')
    pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                           adapter_name="lcm-lora")
    pipe.set_adapters(["lcm-lora"], [0.8])
    unet = pipe.unet
    pipe.enable_vae_slicing()
    pipe.to('cuda')

    print(f' \n step 2. save_base_dir')
    num_frames = 16
    save_base_dir = f'experiment_20240710_jpg_gif_mp4/general_prompt_num_frames_{num_frames}_0709'
    os.makedirs(save_base_dir, exist_ok=True)

    print(f' \n step 3. inference test')
    prompt_dir = f'./configs/prompts/filtered_captions_val_{args.start_num}_{args.end_num}.txt'
    with open(prompt_dir, 'r') as f:
        prompts = f.readlines()
    guidance_scales = [1.5]
    num_inference_steps = [6]
    n_prompt = "bad quality, worse quality, low resolution"
    seeds = [0]
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


                    base_folder = os.path.join(prompt_folder, f'guidance_{guidance_scale}_inference_{inference_scale}')
                    os.makedirs(base_folder, exist_ok=True)

                    motion_controller = MutualMotionAttentionControl(guidance_scale=guidance_scales[0],
                                                                     frame_num=16,
                                                                     full_attention=True,
                                                                     window_attention=False,
                                                                     window_size=16,
                                                                     total_frame_num=16,
                                                                     is_teacher = args.is_teacher,
                                                                     skip_layers=[])  # 32
                    regiter_motion_attention_editor_diffusers(unet, motion_controller)

                    pipe.unet = unet
                    start_time = time.time()
                    # seed setting

                    output = pipe(prompt=prompt,
                                  negative_prompt=n_prompt,
                                  num_frames=num_frames,
                                  guidance_scale=guidance_scale,
                                  num_inference_steps=inference_scale,
                                  generator=torch.Generator("cpu").manual_seed(seed), )
                    end_time = time.time()
                    elapse_time = end_time - start_time
                    frames = output.frames[0]
                    #
                    save_folder = os.path.join(base_folder, f'origin')
                    #save_folder = os.path.join(base_folder, f'origin_elapse_time_{elapse_time}')
                    os.makedirs(save_folder, exist_ok=True)
                    # [1] frame image save
                    for frame_idx, img in enumerate(frames) :
                        save_frame_idx = str(frame_idx).zfill(2)
                        img.save(os.path.join(save_folder, f'prompt_{save_p}_seed_{seed}_frame_idx_{save_frame_idx}.jpg'))
                    export_to_gif(frames, os.path.join(save_folder, f'prompt_{save_p}_seed_{seed}.gif'))
                    export_to_video(frames, os.path.join(save_folder, f'prompt_{save_p}_seed_{seed}.mp4'))


                    # text recording
                    with open(os.path.join(save_folder, 'elapse_time.txt'), 'w') as f :
                        f.write(f'elapse_time = {elapse_time}')

                    for i, skip_layer in layer_dict.items() :
                        print(f' ** {skip_layer} Test ** ')
                        motion_controller = MutualMotionAttentionControl(guidance_scale=guidance_scales[0],
                                                                         frame_num=16,
                                                                         full_attention=True,
                                                                         window_attention=False,
                                                                         window_size=16,
                                                                         total_frame_num=16,
                                                                         is_teacher = args.is_teacher,
                                                                         is_eval = False,
                                                                         skip_layers=[skip_layer])  # 32
                        regiter_motion_attention_editor_diffusers(unet, motion_controller)
                        pipe.unet = unet
                        output = pipe(prompt=prompt,
                                      negative_prompt="bad quality, worse quality, low resolution",
                                      num_frames=num_frames,
                                      guidance_scale=guidance_scale,
                                      num_inference_steps=inference_scale,
                                      generator=torch.Generator("cpu").manual_seed(seed), )
                        ########
                        frames = output.frames[0]
                        save_folder = os.path.join(base_folder, f'{skip_layer}')
                        os.makedirs(save_folder, exist_ok=True)
                        #save_name = f'prompt_{save_p}_seed_{seed}.mp4'
                        ##export_to_gif(frames, os.path.join(save_folder, save_name))
                        #export_to_video(frames, os.path.join(save_folder, save_name))
                        ##########
                        # [1] frame image save
                        for frame_idx, img in enumerate(frames):
                            save_frame_idx = str(frame_idx).zfill(2)
                            img.save(
                                os.path.join(save_folder, f'prompt_{save_p}_seed_{seed}_frame_idx_{save_frame_idx}.jpg'))
                        export_to_gif(frames, os.path.join(save_folder, f'prompt_{save_p}_seed_{seed}.gif'))
                        export_to_video(frames, os.path.join(save_folder, f'prompt_{save_p}_seed_{seed}.mp4'))

if __name__ == "__main__" :
    import argparse
    parser = argparse.ArgumentParser(description='t2v_inference')
    parser.add_argument('--m', type=int,default=1)
    parser.add_argument('--is_teacher', action='store_true')
    parser.add_argument('--start_num', type=int, default=100)
    parser.add_argument('--end_num', type=int, default=140)
    args = parser.parse_args()
    main(args)