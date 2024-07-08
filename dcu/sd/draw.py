import torch
import time
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "runwayml/stable-diffusion-v1-5"
prompts = [  # try your own prompts
    "moon",
    "a boat in the sea",
    "a photograph of a squirrel holding an arrow above its head and holding a longbow in its left hand"
]
height = 512
width = 512
num_inference_steps = 30

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
base_count = 0

for prompt in prompts:
    start = time.time()
    image = pipe(prompt, height=height, width=width, num_inference_steps=num_inference_steps, num_images_per_prompt=1).images[0]
    print(f"the {base_count} text-to-image use time {time.time()-start}")
    base_count += 1
    image.save(f"{base_count:03}.png")
    
print(f"Your samples are ready and waiting for you here.\nEnjoy.")
