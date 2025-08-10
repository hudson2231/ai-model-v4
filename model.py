import os
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


# --- helpers ---
def _limit_size(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        # snap to multiple of 8 anyway
        return img.resize(((w // 8) * 8 or 8, (h // 8) * 8 or 8), Image.BILINEAR)
    scale = max_side / float(max(w, h))
    nw, nh = int(w * scale), int(h * scale)
    nw, nh = (nw // 8) * 8, (nh // 8) * 8
    nw = max(nw, 8)
    nh = max(nh, 8)
    return img.resize((nw, nh), Image.BILINEAR)


class Predictor(BasePredictor):
    def setup(self):
        # Stable cache + faster downloads
        os.environ["HF_HOME"] = "/root/.cache/huggingface"
        os.environ["TRANSFORMERS_CACHE"] = "/root/.cache/huggingface"
        os.environ["HF_DATASETS_CACHE"] = "/root/.cache/huggingface"
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        # ControlNet: HED (fp16, safetensors)
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-hed",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )

        # SD1.5 + ControlNet
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            use_safetensors=True,
            safety_checker=None,
            variant="fp16",
        )

        # Sharp, stable sampler
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        # Perf toggles (safe if not available)
        try:
            self.pipe.enable_vae_slicing()
            self.pipe.enable_vae_tiling()
            self.pipe.enable_attention_slicing("max")
            if hasattr(self.pipe, "enable_sdpa"):
                self.pipe.enable_sdpa()
        except Exception:
            pass

        # Low VRAM safety: disable NSFW checker already done above (safety_checker=None)
        try:
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        # Move to GPU
        self.pipe.to("cuda")

        # Load HED once (pin to same cache)
        self.hed = HEDdetector.from_pretrained("lllyasviel/ControlNet", cache_dir=os.environ["HF_HOME"])

        # Prompt/negative (single source of truth)
        self.prompt = (
            "Ultra-sharp black and white LINE ART of the full scene (subject AND background). "
            "Clean, closed outlines only; no tone, no shading, no gradients, no color. "
            "Accurate structure and proportions; preserve scene layout and details. "
            "Adult coloring book page style; crisp, printable, high-contrast lines."
        )
        self.negative = (
            "color, colours, grayscale shading, grey, gray, tone, gradients, halftone, "
            "sketchy lines, pencil shading, cross-hatching, stippling, watercolor, oil paint, "
            "blur, soft focus, noise, artifacts, text, watermark, logo, distortion, abstraction, "
            "extra objects, extra limbs, incorrect anatomy, low detail"
        )

    def predict(
        self,
        input_image: Path = Input(description="Photo to convert into printable line art"),
        steps: int = Input(description="Diffusion steps", default=20, ge=8, le=50),
        guidance: float = Input(description="CFG (higher = more prompt adherence)", default=9.0, ge=1.0, le=20.0),
        max_side: int = Input(description="Max image side in px (multiple of 8)", default=768, ge=512, le=1024),
        seed: int = Input(description="Seed for repeatability", default=42),
    ) -> Path:
        # Deterministic generator on GPU
        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Load & size
        image = Image.open(input_image).convert("RGB")
        image = _limit_size(image, max_side)

        # HED edge map at working size (faster than full-res)
        edge = self.hed(image).resize(image.size)

        # Generate
        result = self.pipe(
            prompt=self.prompt,
            negative_prompt=self.negative,
            image=edge,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        ).images[0]

        out_path = "/tmp/output.png"
        result.save(out_path)
        return Path(out_path)

