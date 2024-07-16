import os, io, csv, math, random
import numpy as np
from einops import rearrange
from decord import VideoReader

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from animatediff.utils.util import zero_rank_print


class DistillWebVid10M(Dataset):
    def __init__(
            self,
            csv_path,
            video_folder,
            sample_size=256,
            sample_stride=4,
            sample_n_frames=16,
            is_image=False,
    ):
        zero_rank_print(f"loading annotations from {csv_path} ...")
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        # ----------------------------------------------- Test ----------------------------------------------- #
        # self.dataset = [self.dataset[0]]
        # ----------------------------------------------- Test ----------------------------------------------- #

        self.length = len(self.dataset)
        zero_rank_print(f"data scale: {self.length}")
        self.video_folder = video_folder
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image = is_image
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                    transforms.Resize(sample_size[0]),
                                                    transforms.CenterCrop(sample_size),
                                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                                                                         inplace=True), ])

    def get_batch(self, idx):
        # self.dataset = csv file
        # self.video_folder = 'base_video_path'
        # self.dataset = [{'videoid' : 1, "name" : 11, "page_dir" : "test_1.gif"},
        #                {'videoid' : 2, "name" : 22, "page_dir" : "test_2.gif"}]

        video_dict = self.dataset[idx]
        # check againn
        key_list = list(video_dict.keys())
        name = video_dict['name']
        if len(key_list) > 3:
            name = video_dict['name']
            none_list = video_dict[None]
            non_str = ','.join(none_list)
            video_dict['name'] = name + non_str
        # video_dict= video_dict = {
        # 'videoid': 'prompt_83_seed_0',
        # 'page_dir': 'prompt_83_seed_0.gif',
        # 'name': 'Man actor wearing a tiger clothe preparing his acting with a tambourine'}
        videoid, name, page_dir = video_dict['videoid'], video_dict['name'], video_dict['page_dir']
        video_dir = os.path.join(self.video_folder, f"{videoid}.mp4")
        # /share0/dreamyou070/dreamyou070/MyData/video/webvid-10M/webvid-10M-partial-video/prompt_69_seed_0.gif
        # cannot read gif ?
        video_reader = VideoReader(video_dir)
        video_length = len(video_reader)
        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]  # why only one frame ?
        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader

        if self.is_image:
            pixel_values = pixel_values[0]

        return pixel_values, name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        pixel_values, name = self.get_batch(idx)
        # try:

        #    break

        # except Exception as e:
        #    idx = random.randint(0, self.length-1)

        pixel_values = self.pixel_transforms(pixel_values)
        sample = dict(pixel_values=pixel_values, text=name)
        return sample

