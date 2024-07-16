import os, shutil
import glob

csv_dir = '/share0/dreamyou070/dreamyou070/MyData/video/webvid-10M/webvid-10M-csv/0_300.csv'
elems = []
elem = ['videoid','page_dir','name']
elems.append(elem)

def main() :

    base_folder_dir = '/share0/dreamyou070/dreamyou070/OneStepVideo/experiment_20240710_jpg_gif_mp4/general_prompt_num_frames_16_0709'
    prompt_folders = os.listdir(base_folder_dir)
    for prompt_folder in prompt_folders :
        #print(f'prompt_folder : {prompt_folder}')
        p = prompt_folder.split('_')[-1] # 000
        prompt_folder_dir = os.path.join(base_folder_dir, prompt_folder)
        prompt_file = os.path.join(prompt_folder_dir, 'prompt.txt')
        with open(prompt_file, 'r') as f :
            prompt = f.readlines()
        name = prompt[0].strip()
        try :
            original_video_base_dir = os.path.join(prompt_folder_dir, f'guidance_1.5_inference_6/origin')
            #original_video_base_dir_src = glob.glob(f'{original_video_base_dir}/*.gif')[0]
            original_video_base_dir_trg = os.path.join(f'{original_video_base_dir}/prompt_{p}_seed_0.mp4')
            #os.rename(original_video_base_dir_src,original_video_base_dir_trg)
            new_video_base = '/share0/dreamyou070/dreamyou070/MyData/video/webvid-10M/webvid-10M-partial-video'
            os.makedirs(new_video_base, exist_ok=True)
            new_video_dir = os.path.join(new_video_base, f'prompt_{p}_seed_0.mp4')
            # copy
            shutil.copy(original_video_base_dir_trg, new_video_dir)
            videoid = f'prompt_{p}_seed_0'
            page_dir = f'{videoid}.mp4'
            elem = [videoid,page_dir,name]
            #print(f'elem = {elem}')
            elems.append(elem)
        except :
            continue
    # write csv
    print(f'len of elems : {len(elems)}')
    with open(csv_dir, 'w') as f :
        for elem in elems :
            f.write(','.join(elem)+'\n')

if __name__ == "__main__" :
    main()