import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel

from transformers import CLIPTextModel, CLIPTokenizer

from animatelcm.models.unet import UNet3DConditionModel

from animatelcm.pipelines.pipeline_animation import AnimationPipeline
from animatelcm.utils.util import save_videos_grid
from animatelcm.utils.util import load_weights
from animatelcm.scheduler.lcm_scheduler import LCMScheduler
from animatelcm.utils.lcm_utils import convert_lcm_lora
from pathlib import Path


def main(args):
    print(f' \n step 1. check config')
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)




    config = OmegaConf.load(args.config)
    samples = []
    sample_idx = 0

    print(f' \n step 3. start')
    for model_idx, (config_key, model_config) in enumerate(list(config.items())):

        print(f'model_config: {model_config}')

        print(f' (3.1) motion module')
        motion_modules = model_config.motion_module # ckpt
        motion_modules = ([motion_modules] if isinstance(motion_modules, str) else list(motion_modules))
        # "/share0/dreamyou070/dreamyou070/SD/pretrained/AnimateLCM/AnimateLCM_sd15_t2v.ckpt"

        print(f' (3.2) motion lora')
        lcm_lora = model_config.lcm_lora
        lcm_lora = [lcm_lora] if isinstance(lcm_lora, str) else list(lcm_lora)
        lcm_lora = lcm_lora * len(motion_modules) if len(lcm_lora) == 1 else lcm_lora

        for motion_module, lcm_lora in zip(motion_modules, lcm_lora):

            inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))

            print(f' (3.3) base model and aniomation pipeline')
            tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
            unet2d = UNet2DConditionModel.from_pretrained(args.pretrained_model_path,subfolder="unet",)
            vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
            unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path,subfolder="unet",
                                                           unet_additional_kwargs=
                                                           OmegaConf.to_container(inference_config.unet_additional_kwargs),)

            print(f' (3.4) LCM schedule')
            schedule = LCMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs))
            pipeline = AnimationPipeline(vae=vae,text_encoder=text_encoder,tokenizer=tokenizer,
                                         unet=unet, scheduler=schedule).to("cuda")
            print(f' (3.4) loading motion module and personalized model')
            pipeline = load_weights(pipeline,
                                    motion_module_path=motion_module,
                                    motion_module_lora_configs=model_config.get("motion_module_lora_configs", []),
                                    dreambooth_model_path=model_config.get("dreambooth_path", ""),
                                    lora_model_path=model_config.get("lora_model_path", ""),
                                    lora_alpha=model_config.get("lora_alpha", 0.8),).to("cuda")

            print(f' (3.5) convert to lcm lora')
            pipeline.unet = convert_lcm_lora(pipeline.unet, lcm_lora, 1.0)

            print(f' (3.5.1) prompt')
            prompts = model_config.prompt
            print(f' (3.5.2) input imgage')
            image_paths = (model_config.image_paths if hasattr(model_config, "image_paths")
                            else [None for _ in range(len(prompts))])
            control_paths = (model_config.control_paths if hasattr(model_config, "control_paths")
                             else [None for _ in range(len(prompts))])
            print(f' (3.5.3) n prompt')
            n_prompts = (list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1
                         else model_config.n_prompt)
            print(f' (3.5.4) random seed')
            random_seeds = model_config.get("seed", [-1])
            random_seeds = ([random_seeds] if isinstance(random_seeds, int) else list(random_seeds))
            random_seeds = (random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds)
            config[config_key].random_seed = []
            for prompt_idx, (prompt,n_prompt,random_seed,image_path,control_path,) in enumerate(
                zip(prompts, n_prompts, random_seeds, image_paths, control_paths)):
                _, name = os.path.split(image_path)
                name = name.split(".")[0]

                print(f' \n step 2. result dir')
                time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
                savedir = f"result/{Path(args.config).stem}-{time_str}_base_general_model_img_{name}"
                os.makedirs(savedir)

                if random_seed != -1:
                    torch.manual_seed(random_seed)
                else:
                    torch.seed()
                config[config_key].random_seed.append(torch.initial_seed())
                print(f"current seed: {torch.initial_seed()}")
                print(f"sampling {prompt} ...")
                os.makedirs(f'{savedir}/sample', exist_ok=True)
                save_name = f"{savedir}/sample/{sample_idx}-{prompt}_input_img.png"
                sample = pipeline(prompt,
                                  negative_prompt=n_prompt,
                                  control_path=control_path,
                                  image_path=image_path, ###########
                                  num_inference_steps=model_config.steps,
                                  guidance_scale=model_config.guidance_scale,
                                  width=model_config.W,
                                  height=model_config.H,
                                  video_length=model_config.L,
                                  do_classifier_free_guidance=model_config.get("do_classifier_free_guidance", False),
                                  save_folder = save_name).videos
                samples.append(sample)
                prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
                save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{prompt}.gif")
                print(f"save to {savedir}/sample/{prompt}.gif")

                sample_idx += 1

    samples = torch.concat(samples)
    save_videos_grid(samples, f"{savedir}/sample.gif", n_rows=4)

    OmegaConf.save(config, f"{savedir}/config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path",
                        type=str,
                        default="models/StableDiffusion/stable-diffusion-v1-5",)
    parser.add_argument("--inference_config", type=str, default="configs/inference-t2v.yaml")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
