# model.py — Stable, output-or-bust
# - Default: ControlNet Canny (reliable)
# - Optional: HED if available; if HED fails, auto-fallback to Canny
# - Forces GPU, logs clearly, warmup pulls weights
# - If generation fails for any reason, returns the edge map so you still get an output file

import os, time, io
import torch
from cog import BasePredictor, Input, Path
from PIL import Image

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from controlnet_aux import CannyDetector
try:
    from controlnet_aux import HEDdetector
except Exception:
    HEDdetector = None  # optional


# ---------- helpers ----------
def _limit_size(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        W, H = (w // 8) * 8, (h // 8) * 8
        return img.resize((max(W, 8), max(H, 8)), Image.BILINEAR)
    s = max_side / float(max(w, h))
    W, H = int(w * s), int(h * s)
    W, H = (W // 8) * 8, (H // 8) * 8
    return img.resize((max(W, 8), max(H, 8)), Image.BILINEAR)

def _ensure_pil(x) -> Image.Image:
    if isinstance(x, Image.Image):
        return x
    try:
        return Image.fromarray(x)
    except Exception:
        # last resort: make a white square to avoid crashes
        return Image.new("RGB", (512, 512), (255, 255, 255))


class Predictor(BasePredictor):
    def setup(self):
        # 1) Hard GPU check (fail fast if Replicate schedules CPU)
        if not torch.cuda.is_available():
            raise RuntimeError("GPU not available. On Replicate, pick L40S or A100.")
        print(f"[setup] CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

        # 2) Stable HF caching + faster transfers
        os.environ["HF_HOME"] = "/root/.cache/huggingface"
        os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
        os.environ["HF_DATASETS_CACHE"] = os.environ["HF_HOME"]
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # 3) Load ControlNet Canny + SD1.5
        print("[setup] load ControlNet(CANNY) + SD1.5 …", flush=True)
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            use_safetensors=True,
            safety_checker=None,
            variant="fp16",
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        # 4) Perf toggles (safe if not present)
        try:
            if hasattr(self.pipe, "enable_sdpa"): self.pipe.enable_sdpa()
            self.pipe.enable_vae_slicing()
            self.pipe.enable_attention_slicing("max")
        except Exception:
            pass
        try:
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass

        # 5) Move to GPU
        print("[setup] move pipeline to CUDA …", flush=True)
        self.pipe.to("cuda")
        try:
            self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe.vae.to(memory_format=torch.channels_last)
        except Exception:
            pass

        # 6) Edge detectors (Canny always; HED optional + correct repo path)
        print("[setup] load edge detectors …", flush=True)
        self.canny = CannyDetector()
        self.hed = None
        if HEDdetector is not None:
            try:
                self.hed = HEDdetector.from_pretrained("lllyasviel/Annotators", cache_dir=os.environ["HF_HOME"])
                try:
                    self.hed.to("cuda")
                    print("[setup] HED on CUDA", flush=True)
                except Exception:
                    print("[setup] HED stays on CPU", flush=True)
            except Exception as e:
                print(f"[setup] HED load failed, will fallback to Canny: {e}", flush=True)
        else:
            print("[setup] HED package not present; using Canny only", flush=True)

        # 7) Prompts
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

        # 8) Warmup: pulls weights, compiles kernels
        try:
            print("[setup] warmup …", flush=True)
            dummy = Image.new("RGB", (96, 96), (255, 255, 255))
            edge = self.canny(dummy, low_threshold=50, high_threshold=150)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                _ = self.pipe(
                    prompt="line art", negative_prompt=self.negative, image=edge,
                    num_inference_steps=1, guidance_scale=1.0
                ).images[0]
            print("[setup] warmup done", flush=True)
        except Exception as e:
            print(f"[setup] warmup skipped: {e}", flush=True)

    def predict(
        self,
        input_image: Path = Input(description="Photo to convert into printable line art"),
        steps: int = Input(description="Diffusion steps", default=18, ge=8, le=50),
        guidance: float = Input(description="CFG", default=8.5, ge=1.0, le=20.0),
        max_side: int = Input(description="Max image side px", default=704, ge=512, le=1024),
        seed: int = Input(description="Seed", default=42),
        edge_detector: str = Input(description="Edge detector", default="canny", choices=["canny", "hed"]),
        canny_low: int = Input(description="Canny low threshold", default=50, ge=1, le=255),
        canny_high: int = Input(description="Canny high threshold", default=150, ge=1, le=255),
        fallback_return_edges: bool = Input(description="If generation fails, return edge map", default=True),
    ) -> Path:
        t0 = time.time()
        print("[predict] load image …", flush=True)
        img = Image.open(input_image).convert("RGB")
        img = _limit_size(img, max_side)

        # Edges (stable + safe)
        use_hed = (edge_detector == "hed") and (self.hed is not None)
        print(f"[predict] edges via {'hed' if use_hed else 'canny'} at {img.size} …", flush=True)
        try:
            if use_hed:
                edge = self.hed(img).resize(img.size)
            else:
                edge = self.canny(img, low_threshold=canny_low, high_threshold=canny_high)
        except Exception as e:
            print(f"[predict] edge failed ({e}) → fallback to Canny", flush=True)
            edge = self.canny(img, low_threshold=canny_low, high_threshold=canny_high)
        edge = _ensure_pil(edge)

        # Progress callback so logs prove liveness
        def _on_step_end(pipe, i, t, **kw):
            if i % 5 == 0:
                print(f"[predict] diffusion step {i}", flush=True)

        gen = torch.Generator(device="cuda").manual_seed(seed)

        # Main generation with robust fallbacks
        try:
            print(f"[predict] generate (steps={steps}, guidance={guidance}) …", flush=True)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                out_img = self.pipe(
                    prompt=self.prompt,
                    negative_prompt=self.negative,
                    image=edge,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=gen,
                    callback_on_step_end=_on_step_end,
                    callback_on_step_end_tensor_inputs=["latents"],
                ).images[0]
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "cublas" in str(e):
                print("[predict] OOM → retry smaller …", flush=True)
                img_small = _limit_size(img, max(512, int(max_side * 0.8)))
                try:
                    edge = self.canny(img_small, low_threshold=canny_low, high_threshold=canny_high)
                except Exception:
                    pass
                steps = max(12, int(steps * 0.8))
                guidance = max(6.0, min(guidance, 10.0))
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    out_img = self.pipe(
                        prompt=self.prompt,
                        negative_prompt=self.negative,
                        image=_ensure_pil(edge),
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        generator=gen,
                        callback_on_step_end=_on_step_end,
                        callback_on_step_end_tensor_inputs=["latents"],
                    ).images[0]
            else:
                print(f"[predict] generation error: {e}", flush=True)
                if fallback_return_edges:
                    print("[predict] returning edge map as final output", flush=True)
                    out = "/tmp/output.png"
                    _ensure_pil(edge).save(out, format="PNG", optimize=False)
                    print(f"[predict] done in {time.time()-t0:.1f}s -> {out}", flush=True)
                    return Path(out)
                raise

        out = "/tmp/output.png"
        out_img.save(out, format="PNG", optimize=False)
        print(f"[predict] done in {time.time()-t0:.1f}s -> {out}", flush=True)
        return Path(out)

