import torch
from diffusers import StableDiffusionXLPipeline

lora_path = "./LoRAs/pytorch_lora_weights.safetensors"
model_base = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = StableDiffusionXLPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
pipe.unet.load_attn_procs(lora_path)
pipe.to("cuda")

image = pipe(prompt="There’s a fast new code editor in town").images[0]
image.save("test.png")
