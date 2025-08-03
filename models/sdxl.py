import logging
import os

import torch
from diffusers import StableDiffusionXLPipeline, UniPCMultistepScheduler
from utils.device import get_device

logger = logging.getLogger(__name__)

class SDXLModel:
    def __init__(self, model_name: str | None = None):
        if model_name is None:
            model_name = os.getenv(
                "MODEL_NAME", "stabilityai/stable-diffusion-xl-base-1.0"
            )

        self.device = get_device()
        torch_dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            use_safetensors=True
        ).to(self.device)
        upc = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler = upc
        self.pipe.enable_attention_slicing()
        self.pipe.enable_vae_slicing()
        if self.device.type == 'cuda':
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except ModuleNotFoundError:
                logger.warning(
                    "xformers not installed; skipping memory-efficient attention. "
                    "Install via 'pip install xformers' for better VRAM usage."
                )
        else:
            logger.info("CPU mode: using float32 precision for compatibility.")
        try:
            self.pipe.enable_model_cpu_offload()
        except Exception as e:
            logger.warning(f"Skipping CPU offload: {e}")

    def generate(self, prompt: str, num_images: int = 1, **kwargs):
        """
        Generate images from the given prompt.
        Returns a list of PIL.Image objects.
        """
        # Pass kwargs directly to the pipe. This will include 'callback' and 'callback_steps'.
        result = self.pipe(prompt, num_images_per_prompt=num_images, **kwargs)
        return result.images
