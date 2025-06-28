from diffusers import StableDiffusionPipeline
import torch

model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.cfg_type = 'original' 
# pipe.cfg_param = xxxx

# Define the text prompt and generate image
prompt = "A scenic view of mountains during sunset"
image = pipe(prompt, guidance_scale = 7.0).images[0]