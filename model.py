import torch
import numpy as np
from cog import BasePredictor, Input, Path
from PIL import Image

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

from controlnet_aux import HEDdetector


class Predictor(BasePredictor):
    def setup(self):
        # ControlNet: HED (edge detector) -> best for clean, faithful outlines
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-hed",
            torch_dtype=torch.float16,
        )

        # SD1.5 + ControlNet (no NSFW safety checker to avoid false blocks)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )

        # Stable, sharp sampler
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        # Memory efficiency if available
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        self.pipe.to("cuda")

        # Load HED once
        self.hed = HEDdetector.from_pretrained("lllyasviel/ControlNet")

        # Hard-coded prompt (one source of truth)
        self.prompt = (
            "Ultra-sharp black and white LINE ART of the full scene (subject AND background). "
            "Clean, closed outlines only; no tone, no shading, no gradients, no color. "
            "Accurate structure and proportions; preserve scene layout and details. "
            "Adult coloring book page style; crisp, printable, high-contrast lines."
        )

        # Negative prompt to clamp style
        self.negative = (
            "color, colours, grayscale shading, grey, gray, tone, gradients, halftone, "
            "sketchy lines, pencil shading, cross-hatching, stippling, watercolor, oil paint, "
            "blur, soft focus, noise, artifacts, text, watermark, logo, distortion, abstraction, "
            "extra objects, extra limbs, incorrect anatomy, low detail"
        )

        # Default generation knobs – tuned for faithful outlines
        self.default_steps = 32
        self.default_guidance = 13.0
        self.default_seed = 42

    def predict(
        self,
        input_image: Path = Input(description="Photo to convert into printable line art"),
    ) -> Path:
        # Deterministic generator on GPU
        generator = torch.Generator(device="cuda").manual_seed(self.default_seed)

        # Load image
        image = Image.open(input_image).convert("RGB")

        # HED edge map (keeps real edges, less noise than Canny)
        edge = self.hed(image).resize(image.size)

        # Generate
        result = self.pipe(
            prompt=self.prompt,
            negative_prompt=self.negative,
            image=edge,
            num_inference_steps=self.default_steps,
            guidance_scale=self.default_guidance,
            generator=generator,
        ).images[0]

        out_path = "/tmp/output.png"
        result.save(out_path)
        return Path(out_path)
