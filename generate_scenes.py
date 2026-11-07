"""U4: generate one scene image per role with a diffusion model (PyTorch + diffusers).
Run once on Colab (free T4) or locally, then commit assets/*.png."""
import os, torch
from diffusers import AutoPipelineForText2Image
import core

dev = "cuda" if torch.cuda.is_available() else "cpu"
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo", torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev)
os.makedirs("assets", exist_ok=True)
for role, cfg in core.ROLES.items():
    img = pipe(prompt=cfg["scene"], num_inference_steps=2, guidance_scale=0.0).images[0]
    img.save(f"assets/scene_{role.lower()}.png")
    print("saved", role)
