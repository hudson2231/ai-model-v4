import os, time
import torch
from cog import BasePredictor, Input, Path
from PIL import Image

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from controlnet_aux import CannyDetector

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

class Predictor(BasePredictor):
    def setup(self):
        # Fail fast if GPU missing
        if not torch.cuda.is_available():
            raise RuntimeError("GPU not available. On Replicate, pick L40S or A100.")
        print(f"[setup] CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

        # Caches + faster downloads
        os.environ["HF_HOME"] = "/root/.cache/huggingface"
        os.environ["TRANSFORMERS_CACHE"] = "/root/.cache/huggingface"
        os.environ["HF_DATASETS_CACHE"] = "/root/.cache/huggingface"
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        print("[setup] load ControlNet Canny + SD1.5 …", flush=True)
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

        # Perf toggles
        try:
            if hasattr(self.pipe, "enable_sdpa"):
                self.pipe.enable_sdpa()
            self.pipe.enable_vae_slicing()
            self.pipe.enable_vae_tiling()
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

        print("[setup] move pipeline to CUDA …", flush=True)
        self.pipe.to("cuda")
        try:
            self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe.vae.to(memory_format=torch.channels_last)
        except Exception:
            pass

        print("[setup] load Canny detector …", flush=True)
        self.canny = CannyDetector()

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

        # Warmup so first predict returns quickly
        try:
            print("[setup] warmup …", flush=True)
            dummy = Image.new("RGB", (96, 96), (255, 255, 255))
            dummy_edge = self.canny(dummy, low_threshold=50, high_threshold=150)
            gen = torch.Generator(device="cuda").manual_seed(123)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                _ = self.pipe(
                    prompt="line art",
                    negative_prompt=self.negative,
                    image=dummy_edge,
                    num_inference_steps=1,
                    guidance_scale=1.0,
                    generator=gen,
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
        canny_low: int = Input(description="Canny low threshold", default=50, ge=1, le=255),
        canny_high: int = Input(description="Canny high threshold", default=150, ge=1, le=255),
    ) -> Path:
        t0 = time.time()
        print("[predict] load image …", flush=True)
        image = Image.open(input_image).convert("RGB")
        image = _limit_size(image, max_side)

        print(f"[predict] edges via canny at {image.size} …", flush=True)
        edge = self.canny(image, low_threshold=canny_low, high_threshold=canny_high)

        def _on_step_end(pipe, i, t, **kwargs):
            if i % 5 == 0:
                print(f"[predict] diffusion step {i}", flush=True)

        gen = torch.Generator(device="cuda").manual_seed(seed)

        def run(sd_steps, sd_guidance):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                img = self.pipe(
                    prompt=self.prompt,
                    negative_prompt=self.negative,
                    image=edge,
                    num_inference_steps=sd_steps,
                    guidance_scale=sd_guidance,
                    generator=gen,
                    callback_on_step_end=_on_step_end,
                    callback_on_step_end_tensor_inputs=["latents"],
                ).images[0]
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            return img

        try:
            print(f"[predict] generate (steps={steps}, guidance={guidance}) …", flush=True)
            result = run(steps, guidance)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "cublas" in str(e):
                print("[predict] OOM → retry smaller …", flush=True)
                image = _limit_size(image, max(512, int(max_side * 0.8)))
                edge = self.canny(image, low_threshold=canny_low, high_threshold=canny_high)
                steps = max(12, int(steps * 0.8))
                guidance = max(6.0, min(guidance, 10.0))
                result = run(steps, guidance)
            else:
                print(f"[predict] error: {e} → retry smaller …", flush=True)
                image = _limit_size(image, max(512, int(max_side * 0.8)))
                edge = self.canny(image, low_threshold=canny_low, high_threshold=canny_high)
                steps = max(12, int(steps * 0.8))
                guidance = max(6.0, min(guidance, 10.0))
                result = run(steps, guidance)

        out_path = "/tmp/output.png"
        result.save(out_path, format="PNG", optimize=False)
        print(f"[predict] done in {time.time() - t0:.1f}s -> {out_path}", flush=True)
        return Path(out_path)
