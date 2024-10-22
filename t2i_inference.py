import torch
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
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


def main() :
    from diffusers import StableDiffusion
    print(f' \n step 1. make Motion Base Pipeline with LCM Scheduler')
    pipe = StableDiffusion.from_pretrained("emilianJR/epiCRealism",
                                               torch_dtype=torch.float16)
    unet = pipe.unet
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
    pipe.enable_vae_slicing()
    pipe.to('cuda')

    print(f' \n step 2. save_base_dir')
    num_frames = 16
    save_base_dir = f'experiment_20240708/general_prompt_test_gpt_num_frames_{num_frames}_0709_test_2'
    os.makedirs(save_base_dir, exist_ok=True)

    print(f' \n step 3. inference test')
    prompt_dir = r'configs/prompts/test_prompt_0709.txt'
    with open(prompt_dir, 'r') as f:
        prompts = f.readlines()
    guidance_scales = [6]
    num_inference_steps = [30]
    n_prompt = "bad quality, worse quality, low resolution"
    seeds = [0,42,876,5787,78935]
    for p, prompt in enumerate(prompts):
        for guidance_scale in guidance_scales :
            for inference_scale in num_inference_steps :
                for seed in seeds :

                    base_folder = os.path.join(save_base_dir, f'guidance_{guidance_scale}_inference_{inference_scale}')
                    os.makedirs(base_folder, exist_ok=True)
                    """
                    motion_controller = MutualMotionAttentionControl(guidance_scale=guidance_scales[0],
                                                                     frame_num=16,
                                                                     full_attention=True,
                                                                     window_attention=False,
                                                                     window_size=16,
                                                                     total_frame_num=16,
                                                                     skip_layers=[])  # 32
                    regiter_motion_attention_editor_diffusers(unet, motion_controller)
                    """
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
                    save_name = f'prompt_{p}_seed_{seed}.gif'
                    export_to_gif(frames, os.path.join(save_folder, save_name))
                    """
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
                                                                         is_teacher = False,
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
                        frames = output.frames[0]
                        save_folder = os.path.join(base_folder, f'{skip_layer}')
                        os.makedirs(save_folder, exist_ok=True)
                        save_name = f'prompt_{p}_seed_{seed}.gif'
                        export_to_gif(frames, os.path.join(save_folder, save_name))
                    """
if __name__ == "__main__" :
    main()