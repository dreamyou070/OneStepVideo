import os
import cv2, torch
from PIL import Image

mp4_dir = 'prompt_0_seed_0.mp4'
video = cv2.VideoCapture(mp4_dir)
print(f'video = {video}')
ret, frame = video.read()
print()
frame_pil = Image.fromarray(frame)
print(type(frame_pil))
frame_pil_np = torch.tensor(frame)
#print(f'frame_pil = {frame_pil_np.shape}')