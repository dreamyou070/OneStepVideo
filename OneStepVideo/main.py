
from diffusers import AutoPipelineForText2Image, LCMScheduler
import torch

pipe = AutoPipelineForText2Image.from_pretrained('lykon/dreamshaper-8-lcm', torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
