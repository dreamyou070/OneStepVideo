import torch
from diffusers import StableDiffusionSAGPipeline
from accelerate.utils import set_seed


def main() :

    pipe = StableDiffusionSAGPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    seed = 8978
    prompt = "."
    guidance_scale = 7.5
    num_images_per_prompt = 1

    sag_scale = 1.0

    set_seed(seed)
    images = pipe(
        prompt, num_images_per_prompt=num_images_per_prompt, guidance_scale=guidance_scale, sag_scale=sag_scale
    ).images
    images[0].save("example.png")

if __name__ == '__main__':
    main()