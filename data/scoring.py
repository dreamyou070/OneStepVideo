import os, torch
import cv2
import glob
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
import numpy as np
from scoring_module.metric import calculate_fid
#transform = transforms.Compose([
#    transforms.Resize((224, 224)),
#    transforms.ToTensor()
#])
def torch_to_scalar(tensor):
    if type(tensor) == torch.Tensor:
        return tensor.cpu().detach().numpy().item()
    else:
        return tensor

def main(args) :

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # [1] CLIP model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    prompt_folder_base_dir = r'/share0/dreamyou070/dreamyou070/OneStepVideo/experiment_20240710_jpg_gif_mp4/general_prompt_num_frames_16_0709'
    prompt_folders = os.listdir(prompt_folder_base_dir)

    # [1] case block calling
    example_folder = os.path.join(prompt_folder_base_dir, f'prompt_idx_000/guidance_1.5_inference_6')
    example_files = os.listdir(example_folder)
    example_files = [ex_file for ex_file in example_files if 'origin' not in ex_file]
    example_files.sort()
    total_csv_elem = []
    e = 'sample_id,'
    for i, example_file in enumerate(example_files) :
        e += f'{example_file}_image_sim,FVD,test_sim,'
    header = e.split(',')

    #csv_file_dir = '/share0/dreamyou070/dreamyou070/OneStepVideo/experiment_20240710_jpg_gif_mp4/scoring.csv'


    prompt_folders.sort()

    for p, prompt_folder in enumerate(prompt_folders) :

        if p > args.start_num and p <= args.end_num :
            str_p = str(p).zfill(3)
            csv_file_dir = f'/share0/dreamyou070/dreamyou070/OneStepVideo/experiment_20240710_jpg_gif_mp4/scoring_{str_p}.csv'
            total_csv_elem = []
            idx = prompt_folder.split('_')[-1]
            prompt_folder_dir = os.path.join(prompt_folder_base_dir, prompt_folder)
            # [0] text feature
            prompt_file = os.path.join(prompt_folder_dir, 'prompt.txt')
            with open(prompt_file, 'r') as f:
                prompt = f.readlines()[0]
            text_embedding = processor(text=prompt, return_tensors="pt")
            input_ides = text_embedding['input_ids'].to(device)
            mask = text_embedding['attention_mask'].to(device)
            text_embedding['input_ids'] = input_ides
            text_embedding['attention_mask'] = mask
            text_features = model.get_text_features(**text_embedding) # tensor of shape (1, 512)

            folder_dir = os.path.join(prompt_folder_dir, 'guidance_1.5_inference_6')
            cases = os.listdir(folder_dir)

            # [1] origin folder
            original_folder = os.path.join(folder_dir, 'origin') #######################################################
            mp4_file_dir = glob.glob(os.path.join(original_folder, '*.mp4'))[0]
            original_files = glob.glob(os.path.join(original_folder, '*.jpg'))
            original_files.sort()
            origin_image_features = {}
            origin_image_features_list = []
            e = 'prompt_folder,'
            origin_features = []
            for i, origin_file in enumerate(original_files) :
                # [1] image feature get
                origin_image = Image.open(origin_file) # [512,512,3]
                origin_feat = torch.tensor(np.array(origin_image))
                origin_features.append(origin_feat)
                inputs = processor(images=origin_image, return_tensors="pt",padding=True)
                image_features = inputs.data['pixel_values']
                image_features = model.get_image_features(pixel_values=image_features.to(device)) #
                origin_image_features[i] = image_features
                origin_image_features_list.append(image_features)

            # ------------------------------------------------------------------------------------------------------------ #
            # [2] case folder
            cases.sort()
            for case in cases :
                if case != 'origin' :
                    case_dir = os.path.join(folder_dir, case) ##########################################################
                    print(f'case_dir = {case_dir}')
                    # up_blocks_1_motion_modules_2
                    mp4_file_case_dir = glob.glob(os.path.join(case_dir, '*.mp4'))[0]
                    case_files = glob.glob(os.path.join(case_dir, '*.jpg'))
                    case_files.sort()
                    case_image_features = {}
                    case_image_features_list = []
                    case_features = []
                    for i, case_file in enumerate(case_files) :
                        # [1] image feature get
                        case_image = Image.open(case_file)
                        case_image_th = torch.tensor(np.array(case_image))
                        case_features.append(case_image_th)
                        inputs = processor(images=case_image, return_tensors="pt", padding=True)
                        image_features = inputs.data['pixel_values']
                        image_features = model.get_image_features(pixel_values=image_features.to(device))  #
                        case_image_features[i] = image_features
                        case_image_features_list.append(image_features)
                    # ------------------------------------------------------------------------------------------------------------ #
                    # [2.1.1] image clip cosine similarity
                    cosine_simes = []
                    text_simes = []
                    for i in range(len(original_files)) :
                        origin_feature = origin_image_features[i]
                        case_feature = case_image_features[i]
                        cos = torch.nn.functional.cosine_similarity(origin_feature, case_feature, dim=-1)
                        text_cos = torch.nn.functional.cosine_similarity(text_features, case_feature, dim=-1)
                        cosine_simes.append(cos)
                        text_simes.append(text_cos)
                    image_similarity = round(torch_to_scalar(sum(cosine_simes) / len(cosine_simes)),3)

                    # [2.1.2] FVD score
                    # fvd_score = torch_to_scalar(calculate_fvd(mp4_file_dir, mp4_file_case_dir))  # low value is good
                    fvd_score = round(calculate_fid(original_folder, case_dir), 2)
                    print(f'fvd_score = {fvd_score}')
                    # [2.1.3] text image similarity
                    text_similarity = round(torch_to_scalar(sum(text_simes) / len(text_simes)), 3)
                    # [3] save
                    e += f'{image_similarity},{fvd_score},{text_similarity},'
            elements = e.split(',')
            total_csv_elem.append(elements)

            # [e1,e2]
            import csv
            with open(csv_file_dir, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)  # 헤더 작성
                writer.writerows(total_csv_elem)  # 데이터 작성





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='scoring')
    parser.add_argument('--start_num', type=int, default=80)
    parser.add_argument('--end_num', type=int, default=160)
    args = parser.parse_args()
    main(args)