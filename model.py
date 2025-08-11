# model.py — robust against HF proxy timeouts
# - Loads SD first (so we can still generate without ControlNet)
# - Tries ControlNet CANNY by default; HED optional
# - Increases HF Hub HTTP timeout/retries
# - If ControlNet fails to download, we still produce output (edge map or plain SD)
# - Forces GPU and logs progress

import os, time
import torch
from cog import BasePredictor, Input, Path
from PIL import Image

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from controlnet_aux import CannyDetector
try:
    from controlnet_aux import HEDdetector
except Exception:
    HEDdetector = None


def _limit_size(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        W, H = (w // 8) * 8, (h // 8) * 8
        return img.resize((max(W, 8), max(H, 8)), Image.BILINEAR)
    s = max_side / float(max(w, h))
    W, H = int(w * s), int(h * s)
    W, H = (W // 8) * 8, (H // 8) * 8
    return img.resize((max(W, 8), max(H, 8)), Image.BILINEAR)


class Predictor(BasePredictor):
    def setup(self):
        # GPU or bust
        if not torch.cuda.is_available():
            raise RuntimeError("GPU not available. Choose L40S or A100 on Replicate.")
        print(f"[setup] CUDA: {torch.cuda.get_device_name(0)}", flush=True)

        # HuggingFace cache + robust networking
        os.environ["HF_HOME"] = "/root/.cache/huggingface"
        os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
        os.environ["HF_DATASETS_CACHE"] = os.environ["HF_HOME"]
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        os.environ["HF_HUB_HTTP_TIMEOUT"] = "60"     # <-- default is 10s; bump to 60s
        os.environ["HF_HUB_MAX_RETRIES"] = "20"      # <-- retry more before failing
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

        # --- Load SD base first (so we can still run without ControlNet) ---
        print("[setup] load SD1.5 base …", flush=True)
        self.pipe_sd = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            use_safetensors=True,
            safety_checker=None,
        )
        self.pipe_sd.scheduler = UniPCMultistepScheduler.from_config(self.pipe_sd.scheduler.config)
        try:
            if hasattr(self.pipe_sd, "enable_sdpa"): self.pipe_sd.enable_sdpa()
            self.pipe_sd.enable_vae_slicing()
            self.pipe_sd.enable_attention_slicing("max")
        except Exception: pass
        self.pipe_sd.to("cuda")

        # Warm little kernel so first call is quicker
        try:
            print("[setup] warmup SD …", flush=True)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                _ = self.pipe_sd("line art", num_inference_steps=1, guidance_scale=1.0).images[0]
            print("[setup] warmup SD done", flush=True)
        except Exception as e:
            print(f"[setup] warmup SD skipped: {e}", flush=True)

        # --- Try to load ControlNet (CANNY default; HED optional later) ---
        self.cn_pipe = None
        self.canny = CannyDetector()
        self.hed = None
        try:
            print("[setup] load ControlNet (CANNY) …", flush=True)
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny",
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
            self.cn_pipe = StableDiffusionControlNetPipeline(
                vae=self.pipe_sd.vae,
                text_encoder=self.pipe_sd.text_encoder,
                tokenizer=self.pipe_sd.tokenizer,
                unet=self.pipe_sd.unet,
                scheduler=self.pipe_sd.scheduler,
                safety_checker=None,
                feature_extractor=self.pipe_sd.feature_extractor,
                controlnet=controlnet,
                requires_safety_checker=False,
            )
            try:
                if hasattr(self.cn_pipe, "enable_sdpa"): self.cn_pipe.enable_sdpa()
                self.cn_pipe.enable_vae_slicing()
                self.cn_pipe.enable_attention_slicing("max")
            except Exception: pass
            self.cn_pipe.to("cuda")
            print("[setup] ControlNet(CANNY) ready", flush=True)
        except Exception as e:
            print(f"[setup] ControlNet(CANNY) load failed → will use plain SD if needed: {e}", flush=True)

        # Optional HED edge detector (NOT required)
        if HEDdetector is not None:
            try:
                print("[setup] try load HED detector …", flush=True)
                self.hed = HEDdetector.from_pretrained("lllyasviel/Annotators", cache_dir=os.environ["HF_HOME"])
                try:
                    self.hed.to("cuda")
                    print("[setup] HED on CUDA", flush=True)
                except Exception:
                    print("[setup] HED on CPU", flush=True)
            except Exception as e:
                print(f"[setup] HED detector unavailable (ok): {e}", flush=True)
        else:
            print("[setup] HED package missing (ok)", flush=True)

    def predict(
        self,
        input_image: Path = Input(description="Photo to convert into printable line art"),
        steps: int = Input(default=18, ge=8, le=50),
        guidance: float = Input(default=8.5, ge=1.0, le=20.0),
        max_side: int = Input(default=704, ge=512, le=1024),
        seed: int = Input(default=42),
        edge_detector: str = Input(default="canny", choices=["canny", "hed"]),
        canny_low: int = Input(default=50, ge=1, le=255),
        canny_high: int = Input(default=150, ge=1, le=255),
        fallback_return_edges: bool = Input(default=True),
    ) -> Path:
        t0 = time.time()
        img = Image.open(input_image).convert("RGB")
        img = _limit_size(img, max_side)

        # Build edges
        use_hed = (edge_detector == "hed") and (self.hed is not None)
        print(f"[predict] edges via {'hed' if use_hed else 'canny'} at {img.size}", flush=True)
        try:
            if use_hed:
                edge = self.hed(img).resize(img.size)
            else:
                edge = self.canny(img, low_threshold=canny_low, high_threshold=canny_high)
        except Exception as e:
            print(f"[predict] edge failed ({e}) → fallback to canny", flush=True)
            edge = self.canny(img, low_threshold=canny_low, high_threshold=canny_high)

        gen = torch.Generator(device="cuda").manual_seed(seed)
        out_path = "/tmp/output.png"

        # Prefer ControlNet if available
        pipe = self.cn_pipe or self.pipe_sd

        def on_step(pipe, i, t, **kw):
            if i % 5 == 0:
                print(f"[predict] diffusion step {i}", flush=True)

        try:
            print(f"[predict] generate (steps={steps}, cfg={guidance}) with {'ControlNet' if self.cn_pipe else 'plain SD'}", flush=True)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                if self.cn_pipe:
                    image = pipe(
                        prompt=self.prompt,
                        negative_prompt=self.negative,
                        image=edge,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        generator=gen,
                        callback_on_step_end=on_step,
                        callback_on_step_end_tensor_inputs=["latents"],
                    ).images[0]
                else:
                    # Plain SD fallback (no conditioning)
                    image = self.pipe_sd(
                        prompt=self.prompt,
                        negative_prompt=self.negative,
                        num_inference_steps=max(steps - 4, 8),
                        guidance_scale=guidance,
                        generator=gen,
                        height=img.height, width=img.width,
                    ).images[0]
            image.save(out_path, format="PNG", optimize=False)
            print(f"[predict] done in {time.time()-t0:.1f}s -> {out_path}", flush=True)
            return Path(out_path)
        except Exception as e:
            print(f"[predict] generation error: {e}", flush=True)
            if fallback_return_edges:
                print("[predict] returning edge map instead", flush=True)
                (edge if isinstance(edge, Image.Image) else Image.fromarray(edge)).save(out_path, format="PNG", optimize=False)
                print(f"[predict] done in {time.time()-t0:.1f}s -> {out_path}", flush=True)
                return Path(out_path)
            raise

