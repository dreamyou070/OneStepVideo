import os
from PIL import Image
import matplotlib.pyplot as plt

base_folder = '../experiment_20240708'
cases = os.listdir(base_folder)
for case in cases :
    case_dir = os.path.join(base_folder, case)
    layer_folders = os.listdir(case_dir)
    for layer_folder in layer_folders :
        if 'origin' in layer_folder :
            original_folder = os.path.join(case_dir, layer_folder)
            videos = os.listdir(original_folder)
            for video in videos :
                origin_video_dir = os.path.join(original_folder, video)
                origin_gif = Image.open(origin_video_dir)
                # read gif from python
                for layer_folder in layer_folders :
                    if 'origin' not in layer_folder :
                        layer_dir = os.path.join(case_dir, layer_folder)
                        compare_video_dir = os.path.join(layer_dir, video)
                        compare_gif = Image.open(compare_video_dir)

                        # [1] FID Score

                        # [2] CLIP Similarity

                        # [3] Frame Consistency