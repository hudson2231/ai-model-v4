import os, time, math
from typing import Optional
import torch
from cog import BasePredictor, Input, Path
from PIL import Image

from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from controlnet_aux import CannyDetector
try:
    # HED is optional; if it fails we keep going
    from controlnet_aux import HEDdetector
    HAVE_HED = True
except Exception:
    HAVE_HED = False

from huggingface_hub import snapshot_download


# ---------------- utils ----------------
def _limit_size(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        W, H = (max((w // 8) * 8, 8), max((h // 8) * 8, 8))
        return img.resize((W, H), Image.BILINEAR)
    s = max_side / float(max(w, h))
    W, H = int(w * s), int(h * s)
    W, H = (max((W // 8) * 8, 8), max((H // 8) * 8, 8))
    return img.resize((W, H), Image.BILINEAR)


def _log(msg: str):
    print(msg, flush=True)


# --------------- predictor --------------
class Predictor(BasePredictor):
    def setup(self):
        # Stable caches + more robust downloads
        os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
        os.environ.setdefault("TRANSFORMERS_CACHE", "/root/.cache/huggingface")
        os.environ.setdefault("HF_DATASETS_CACHE", "/root/.cache/huggingface")
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"     # faster pulls when available
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        _log(f"[setup] device: {self.device} | cuda_available={torch.cuda.is_available()}")

        # Pre-download models with retries (avoids proxy hiccups)
        def _dl(repo: str, allow_patterns=None, optional=False):
            tries, last_err = 3, None
            for k in range(tries):
                try:
                    path = snapshot_download(
                        repo_id=repo,
                        cache_dir=os.environ["HF_HOME"],
                        local_files_only=False,
                        allow_patterns=allow_patterns,
                        resume_download=True,
                        revision=None,
                        max_workers=4,
                        tqdm_class=None,
                    )
                    _log(f"[setup] downloaded: {repo} -> {path}")
                    return path
                except Exception as e:
                    last_err = e
                    _log(f"[setup] download fail {k+1}/{tries} for {repo}: {e}")
                    time.sleep(2.5 * (k + 1))
            if optional:
                _log(f"[setup] optional model skipped: {repo} ({last_err})")
                return None
            raise last_err

        # Core models
        sd_dir = _dl("runwayml/stable-diffusion-v1-5")
        cn_canny_dir = _dl("lllyasviel/sd-controlnet-canny", optional=True)
        # HED controlnet is optional; we'll primarily use Canny
        cn_hed_dir = _dl("lllyasviel/sd-controlnet-hed", optional=True)
        # HED aux .pth (optional)
        hed_aux_dir = _dl("lllyasviel/ControlNet", allow_patterns=["ControlNetHED.pth"], optional=True)

        # Build pipeline: prefer ControlNet(CANNY) -> else plain SD
        self.pipe: Optional[StableDiffusionControlNetPipeline | StableDiffusionPipeline] = None
        controlnet = None

        if cn_canny_dir:
            try:
                _log("[setup] loading ControlNet(CANNY)…")
                controlnet = ControlNetModel.from_pretrained(
                    cn_canny_dir,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                )
            except Exception as e:
                _log(f"[setup] ControlNet(CANNY) load failed: {e}")
                controlnet = None

        if controlnet is not None:
            _log("[setup] building SD1.5 + ControlNet pipeline…")
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                sd_dir,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                use_safetensors=True,
                safety_checker=None,
                variant="fp16",
            )
        else:
            _log("[setup] ControlNet unavailable -> using plain SD1.5")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                sd_dir,
                torch_dtype=torch.float16,
                use_safetensors=True,
                safety_checker=None,
                variant="fp16",
            )

        # Scheduler + perf flags
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        try:
            if hasattr(self.pipe, "enable_sdpa"):
                self.pipe.enable_sdpa()
            self.pipe.enable_vae_slicing()
            self.pipe.enable_vae_tiling()
            if hasattr(self.pipe, "enable_attention_slicing"):
                self.pipe.enable_attention_slicing("max")
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass

        # Move to device
        self.pipe.to(self.device)
        try:
            # micro-opt
            if hasattr(self.pipe, "unet"):
                self.pipe.unet.to(memory_format=torch.channels_last)
            if hasattr(self.pipe, "vae"):
                self.pipe.vae.to(memory_format=torch.channels_last)
        except Exception:
            pass

        # Edge detectors
        _log("[setup] init edge detectors…")
        self.canny = CannyDetector()
        self.hed = None
        if HAVE_HED and hed_aux_dir:
            try:
                self.hed = HEDdetector.from_pretrained("lllyasviel/ControlNet", cache_dir=os.environ["HF_HOME"])
                _log("[setup] HED ready")
            except Exception as e:
                _log(f"[setup] HED unavailable: {e}")

        # Prompts
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

        # Warmup (ensures first predict doesn’t stall forever)
        try:
            _log("[setup] warmup …")
            dummy = Image.new("RGB", (96, 96), (255, 255, 255))
            edge = self.canny(dummy, low_threshold=50, high_threshold=150)
            gen = torch.Generator(device=self.device).manual_seed(123)
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.float16):
                _ = self.pipe(
                    prompt="line art",
                    negative_prompt=self.negative,
                    image=edge if isinstance(self.pipe, StableDiffusionControlNetPipeline) else None,
                    num_inference_steps=1,
                    guidance_scale=1.0,
                    generator=gen,
                ).images[0]
            _log("[setup] warmup done")
        except Exception as e:
            _log(f"[setup] warmup skipped: {e}")

    def predict(
        self,
        input_image: Path = Input(description="Photo to convert into printable line art"),
        steps: int = Input(description="Diffusion steps", default=18, ge=8, le=50),
        guidance: float = Input(description="CFG", default=8.5, ge=1.0, le=20.0),
        max_side: int = Input(description="Max image side px", default=704, ge=512, le=1024),
        seed: int = Input(description="Seed", default=42),
        edge_detector: str = Input(
            description="Which edges to use",
            default="canny",
            choices=["canny", "hed"],
        ),
        canny_low: int = Input(description="Canny low threshold", default=50, ge=1, le=255),
        canny_high: int = Input(description="Canny high threshold", default=150, ge=1, le=255),
        fallback_return_edges: bool = Input(
            description="If generation fails, return the edge map PNG instead of erroring",
            default=True,
        ),
    ) -> Path:
        t0 = time.time()
        image = Image.open(input_image).convert("RGB")
        image = _limit_size(image, max_side)
        _log(f"[predict] image size -> {image.size}")

        # Compute edges
        try:
            if edge_detector == "hed" and self.hed is not None:
                _log("[predict] edges: HED")
                edges = self.hed(image).resize(image.size)
            else:
                _log("[predict] edges: Canny")
                edges = self.canny(image, low_threshold=canny_low, high_threshold=canny_high)
        except Exception as e:
            _log(f"[predict] edge fail ({e}) -> fallback to Canny")
            edges = self.canny(image, low_threshold=canny_low, high_threshold=canny_high)

        # Progress logs
        def _on_step_end(pipe, i, t, **kwargs):
            if i % 5 == 0:
                _log(f"[predict] diffusion step {i}")

        gen = torch.Generator(device=self.device).manual_seed(seed)

        def _generate(sd_steps, sd_guidance, cond_image):
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.float16):
                if isinstance(self.pipe, StableDiffusionControlNetPipeline):
                    out = self.pipe(
                        prompt=self.prompt,
                        negative_prompt=self.negative,
                        image=cond_image,
                        num_inference_steps=sd_steps,
                        guidance_scale=sd_guidance,
                        generator=gen,
                        callback_on_step_end=_on_step_end,
                        callback_on_step_end_tensor_inputs=["latents"],
                    ).images[0]
                else:
                    # plain SD fallback (ignore cond)
                    out = self.pipe(
                        prompt=self.prompt,
                        negative_prompt=self.negative,
                        num_inference_steps=sd_steps,
                        guidance_scale=sd_guidance,
                        generator=gen,
                    ).images[0]
            if self.device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            return out

        # Try full settings, then shrink on failure
        try:
            _log(f"[predict] generate (steps={steps}, guidance={guidance}) …")
            result = _generate(steps, guidance, edges)
        except RuntimeError as e:
            msg = str(e)
            _log(f"[predict] generation error: {msg}")
            # OOM or weird kernel -> shrink and retry once
            new_side = max(512, int(max_side * 0.8))
            image = _limit_size(image, new_side)
            try:
                edges = self.canny(image, low_threshold=canny_low, high_threshold=canny_high)
            except Exception:
                pass
            steps = max(12, int(steps * 0.8))
            guidance = max(6.0, min(guidance, 10.0))
            _log(f"[predict] retry smaller (steps={steps}, guidance={guidance}, side<= {new_side}) …")
            result = _generate(steps, guidance, edges)
        except Exception as e:
            _log(f"[predict] fatal generation error: {e}")
            if fallback_return_edges:
                _log("[predict] returning edges PNG as fallback")
                out_path = "/tmp/output_edges.png"
                edges.save(out_path, format="PNG", optimize=False)
                return Path(out_path)
            raise

        out_path = "/tmp/output.png"
        result.save(out_path, format="PNG", optimize=False)
        _log(f"[predict] done in {time.time() - t0:.1f}s -> {out_path}")
        return Path(out_path)


