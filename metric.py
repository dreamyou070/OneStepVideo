# pip install pytorch-fid
# pip install git+https://github.com/openai/CLIP.git

import os
import subprocess
import torch
import clip
from PIL import Image

def calculate_fid(real_images_path, generated_images_path):
    # Ensure the paths exist
    if not os.path.exists(real_images_path):
        raise FileNotFoundError(f"Real images path '{real_images_path}' does not exist.")
    if not os.path.exists(generated_images_path):
        raise FileNotFoundError(f"Generated images path '{generated_images_path}' does not exist.")
    
    # Run the FID calculation
    result = subprocess.run(['python', '-m', 'pytorch_fid', real_images_path, generated_images_path], 
                            capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"FID calculation failed with the following error:\n{result.stderr}")
    
    print(result.stdout)


def calculate_clip_score(image_path, text):
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Load and preprocess the image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    # Encode image and text
    text_inputs = clip.tokenize([text]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)
    
    # Calculate cosine similarity
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (image_features @ text_features.T).item()
    
    return similarity


if __name__ == '__main__' :
    # FID
    real_images_folder = '/home/jisoo6687/OneStepVideo/example/1'
    generated_images_folder = '/home/jisoo6687/OneStepVideo/example/2'
    calculate_fid(real_images_path, generated_images_path)
    # CLIP
    image_path = '/home/jisoo6687/OneStepVideo/example/1/Figure_1.png'
    text = "A description of the image"
    clip_score = calculate_clip_score(image_path, text)
    print(f"CLIP Score: {clip_score}")
