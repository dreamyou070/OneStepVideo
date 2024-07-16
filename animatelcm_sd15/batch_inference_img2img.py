import argparse
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.utils import load_image
from diffusers import DiffusionPipeline, LCMScheduler
import torch
from diffusers.utils import load_image
import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf
import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
from animatelcm.utils.util import save_videos_grid
from animatelcm.utils.util import load_weights
from animatelcm.scheduler.lcm_scheduler import LCMScheduler
from animatelcm.utils.lcm_utils import convert_lcm_lora
from pathlib import Path
from diffusers.utils import load_image

def main(args):

    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"result/{time_str}-original_scheduling"
    os.makedirs(savedir)

    print(f'\n step 1. make base pipeline')
    #model_id = "sd-dreambooth-library/herge-style" # sd_1.5
    model_id = r'/share0/dreamyou070/dreamyou070/SD/stable-diffusion-webui/models/Stable-diffusion/stable-diffusion-v1-5'
    pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    print(f'\n step 2. loading ip adapter')
    pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    pipeline.set_ip_adapter_scale(0.6)

    print(f'\n step 3. loading lcm lora')
    lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"
    pipeline.load_lora_weights(lcm_lora_id)

    print(f'\n step 4. LCM Scheduler')
    pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)

    print(f'\n step 5. image generating')
    prompt = "herge_style woman in armor, best quality, high quality"
    #generator = torch.Generator(device="cpu").manual_seed(0)
    ip_adapter_image = load_image(
        "https://user-images.githubusercontent.com/24734142/266492875-2d50d223-8475-44f0-a7c6-08b51cb53572.png")
    pipeline.to('cuda')
    image = pipeline(
        prompt=prompt,
        ip_adapter_image=ip_adapter_image,
        num_inference_steps=4,
        guidance_scale=1,
    ).images[0]
    image.save(os.path.join(savedir ,'test_lcm_img.png'))

    # ---------------------------------------------------------------------------------------------------------------------------------- #
    # video
    print(f'\n step 5. video pipeline')
    # v3_sd15_mm.ckpt
    #adaoter_dir = r'/share0/dreamyou070/dreamyou070/SD/pretrained/Animatediff/motion_module/v3_sd15_mm.ckpt'
    #adapter = MotionAdapter.from_pretrained(adaoter_dir, torch_dtype=torch.float16)
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    i2v_pipeline = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)
    i2v_pipeline.vae = vae.to(device = adapter.device, dtype=adapter.dtype)
    i2v_pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    i2v_pipeline.set_ip_adapter_scale(0.6)
    i2v_pipeline.load_lora_weights(lcm_lora_id) # maybe right ...
    #original_scheduler_config = i2v_pipeline.scheduler.config
    #print(f'original_scheduler_config : {original_scheduler_config}')

    #i2v_pipeline.scheduler = LCMScheduler.from_config(i2v_pipeline.scheduler.config)
    i2v_pipeline.to('cuda')

    print(f'\n step 6. video generation')
    # Scale Ref Image and VAE Encode
    # UpscaleAndVaeEncode
    prompt = "herge_style woman in armor, best quality, high quality"
    # generator = torch.Generator(device="cpu").manual_seed(0)
    ip_adapter_image = load_image(
        "https://user-images.githubusercontent.com/24734142/266492875-2d50d223-8475-44f0-a7c6-08b51cb53572.png")
    # vae_model_path
    n_prompt = "worst quality, low quality,nudity,simple background,border,mouth closed,text, patreon,bed,bedroom,white background,((monochrome)),sketch,pink body:1.4,7 arms,8 arms,4 arms"

    sample = i2v_pipeline(prompt,
                          n_prompt=n_prompt,
                          ip_adapter_image=ip_adapter_image,
                          num_inference_steps=40,
                          guidance_scale=20,
                          width=512,
                          height=512,
                          video_length=16,
                          save_folder=os.path.join(savedir, 'referece_img_not_lcm.png')).frames[0]
    video_dir = os.path.join(savedir, 'test_lcm_video.gif')
    export_to_gif(sample,video_dir)
    print(f"save to {video_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path",
                        type=str,
                        default="models/StableDiffusion/stable-diffusion-v1-5",)
    parser.add_argument("--inference_config", type=str, default="configs/inference-t2v.yaml")
    #parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--adapter_scale", type=float, default=1.0)
    parser.add_argument("--inference_steps", type=int, default=100)
    parser.add_argument("--cfg", type=float, default=1.0)
    args = parser.parse_args()
    main(args)
