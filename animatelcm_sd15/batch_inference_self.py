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

    print(f' \n step 1. check config')
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    config = OmegaConf.load(args.config)
    samples = []
    sample_idx = 0

    print(f' \n step 2. loading modeltart')
    for model_idx, (config_key, model_config) in enumerate(list(config.items())):

        # ------------------------------------------------------------------------------------------------------------------------------------- #
        motion_modules = model_config.motion_module # ckpt
        motion_modules = ([motion_modules] if isinstance(motion_modules, str) else list(motion_modules))
        # ------------------------------------------------------------------------------------------------------------------------------------- #
        lcm_lora = model_config.lcm_lora
        lcm_lora = [lcm_lora] if isinstance(lcm_lora, str) else list(lcm_lora)
        lcm_lora = lcm_lora * len(motion_modules) if len(lcm_lora) == 1 else lcm_lora
        # ------------------------------------------------------------------------------------------------------------------------------------- #

        for motion_module, lcm_lora in zip(motion_modules, lcm_lora):
            inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))

            print(f'\n (3.1) motion adapter')
            adapter = MotionAdapter.from_pretrained(model_config.motion_adopter[0], torch_dtype=torch.float16)

            print(f'\n (3.5) base model')
            pipeline = AnimateDiffPipeline.from_pretrained(model_config.base[0], motion_adapter=adapter, torch_dtype=torch.float16)

            print(f'\n (3.2) lcm lora')
            pipeline.load_lora_weights(lcm_lora, adapter_name="lcm") # weight
            lcm_lora_strength = [0.5, 1, 1.2, 2.0]
            for i, strength in enumerate(lcm_lora_strength):
                pipeline.set_adapters(["lcm"], [strength]) # set lcm lora to base model

                print(f'\n (3.3) IP Adapter')
                # ip_adapter what do am i using ?
                pipeline.load_ip_adapter(model_config.ip_adapter[0],
                                         subfolder="models",
                                         weight_name="ip-adapter-full-face_sd15.bin") # setting ip adapter
                adapter_scales = [0.5, 1, 1.2, 2.0]
                for adapter_scale in adapter_scales:
                    pipeline.set_ip_adapter_scale(adapter_scale)  # one scale for each image-mask pair

                    print(f'\n (3.4) LCM schedule')
                    schedule = LCMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs))
                    pipeline.schedule = schedule
                    pipeline.to("cuda")

                    print(f' \n step 4. Start Inference')
                    print(f' (4.1) prompt')
                    prompts = model_config.prompt
                    print(f' (4.2) input imgage')
                    image_paths = (model_config.image_paths if hasattr(model_config, "image_paths")
                                    else [None for _ in range(len(prompts))])
                    control_paths = (model_config.control_paths if hasattr(model_config, "control_paths")
                                     else [None for _ in range(len(prompts))])
                    print(f' (4.3) n prompt')
                    n_prompts = (list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1
                                 else model_config.n_prompt)
                    print(f' (4.4) random seed')
                    random_seeds = model_config.get("seed", [-1])
                    random_seeds = ([random_seeds] if isinstance(random_seeds, int) else list(random_seeds))
                    random_seeds = (random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds)
                    config[config_key].random_seed = []
                    for prompt_idx, (prompt,n_prompt,random_seed,image_path,control_path,) in enumerate(
                        zip(prompts, n_prompts, random_seeds, image_paths, control_paths)):
                        _, name = os.path.split(image_path)
                        name = name.split(".")[0]

                        print(f' \n step 5. result dir')
                        time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
                        savedir = f"result/{Path(args.config).stem}-{time_str}"
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
                        # image_path copy to save_name
                        import shutil
                        shutil.copy(image_path, save_name)
                        sample = pipeline(prompt,
                                          negative_prompt=n_prompt,
                                          control_path=control_path,
                                          ip_adapter_image=load_image(image_path), ###########
                                          num_inference_steps=args.inference_steps,
                                          guidance_scale=args.cfg,
                                          width=model_config.W,
                                          height=model_config.H,
                                          video_length=model_config.L,
                                          do_classifier_free_guidance=model_config.get("do_classifier_free_guidance", False),
                                          save_folder = save_name).frames[0]
                        samples.append(sample)
                        prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
                        export_to_gif(sample, f"{savedir}/sample/lcm_lora_{strength}_adapter_scale_{adapter_scale}_infer_step_{args.inference_steps}_cfg_{args.cfg}.gif")
                        print(f"save to {savedir}/sample/{prompt}.gif")

                        sample_idx += 1

    #samples = torch.concat(samples)
    #save_videos_grid(samples, f"{savedir}/sample.gif", n_rows=4)

    OmegaConf.save(config, f"{savedir}/config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path",
                        type=str,
                        default="models/StableDiffusion/stable-diffusion-v1-5",)
    parser.add_argument("--inference_config", type=str, default="configs/inference-t2v.yaml")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--adapter_scale", type=float, default=1.0)
    parser.add_argument("--inference_steps", type=int, default=100)
    parser.add_argument("--cfg", type=float, default=1.0)
    args = parser.parse_args()
    main(args)
