import os
import shutil

save_folder = r'./experiment_20240710_jpg_gif_mp4_trimming'
os.makedirs(save_folder, exist_ok=True)

base_folder = r'/share0/dreamyou070/dreamyou070/OneStepVideo/experiment_20240710_jpg_gif_mp4/general_prompt_num_frames_16_0709/'
folders = os.listdir(base_folder)
for folder in folders :

    idx = folder.split('_')[-1]

    folder_dir = os.path.join(base_folder, f'{folder}')

    folder_dir = os.path.join(base_folder, f'{folder}/guidance_1.5_inference_6')
    cases = os.listdir(folder_dir)
    text_dir = os.path.join(base_folder, f'{folder}/prompt.txt')

    # [2]
    save_folder_dir = os.path.join(save_folder, folder)
    os.makedirs(save_folder_dir, exist_ok=True)
    for case in cases :
        case_dir = os.path.join(folder_dir, case)
        target_image_dir = os.path.join(case_dir, f'prompt_{idx}_seed_0_frame_idx_15.jpg')

        new_img_dir = os.path.join(save_folder_dir, f'{case}.jpg')
        shutil.copyfile(target_image_dir, new_img_dir)

    new_prompt_dir = os.path.join(save_folder_dir, 'prompt.txt')
    shutil.copyfile(text_dir, new_prompt_dir)
