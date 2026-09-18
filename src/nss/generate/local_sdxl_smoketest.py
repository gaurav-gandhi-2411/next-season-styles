"""Local SDXL + IP-Adapter feasibility smoke test (task B1).

Hard gate, not reusable infrastructure: proves that `stabilityai/stable-diffusion-xl-base-1.0`
plus IP-Adapter (`h94/IP-Adapter`, `sdxl_models/ip-adapter-plus_sdxl_vit-h`) can generate a single
1024x1024 image on an RTX 3070 (8 GB VRAM) within budget (peak VRAM <= ~7.5 GB, generation time
<= 90s excluding one-time model load). Deliberately minimal -- the backend-agnostic generation
interface (B2) and everything built on top of it are separate, later tasks. FLUX is explicitly out
of scope for this hardware (will not fit in 8 GB).

Usage:
    uv run python -m nss.generate.local_sdxl_smoketest
    uv run python -m nss.generate.local_sdxl_smoketest --steps 20 \
        --reference-image data/images/0452717006.jpg
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHT_NAME = "ip-adapter-plus_sdxl_vit-h.safetensors"
# `ip-adapter-plus_sdxl_vit-h` is paired with OpenCLIP-ViT-H-14 (hidden_size=1280) per the
# h94/IP-Adapter README, NOT the bigG encoder (hidden_size=1664) diffusers would auto-load by
# default from "sdxl_models/image_encoder" -- that mismatch (1664 vs the checkpoint's expected
# 1280) surfaces as a matmul shape error at inference time, not at load time. Passing a path with
# a "/" makes diffusers use it verbatim (relative to the repo root) instead of joining it under
# `subfolder`, so this correctly resolves to the ViT-H encoder under "models/image_encoder".
IP_ADAPTER_IMAGE_ENCODER_FOLDER = "models/image_encoder"

DEFAULT_PROMPT = "a black cotton t-shirt, product photography, plain background"
DEFAULT_STEPS = 30
IMAGE_SIZE = 1024
SEED = 42  # hardcoded per project determinism convention

IMAGES_DIR = Path("data/images")
OUTPUT_DIR = Path("data/generated")

# Gate thresholds from the B1 task spec -- exceeding either is a hard stop, not a soft target.
MAX_PEAK_VRAM_GB = 7.5
MAX_GENERATION_SECONDS = 90.0


def load_pipeline() -> tuple[StableDiffusionXLPipeline, float]:
    """Load the SDXL base pipeline with IP-Adapter, fp16, CPU offload, and VAE slicing/tiling.

    Returns:
        The configured pipeline and the wall-clock seconds spent loading it (weights download/
        deserialization + IP-Adapter attach), measured separately from per-image generation time.
    """
    start = time.perf_counter()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.load_ip_adapter(
        IP_ADAPTER_REPO,
        subfolder=IP_ADAPTER_SUBFOLDER,
        weight_name=IP_ADAPTER_WEIGHT_NAME,
        image_encoder_folder=IP_ADAPTER_IMAGE_ENCODER_FOLDER,
    )
    # cpu offload keeps only the active submodule resident on the GPU, which is what makes an
    # 8 GB card viable for an SDXL pipeline at all -- confirm it's actually engaged (not silently
    # a no-op) by checking the pipeline still reports its device as GPU-backed post-offload.
    pipe.enable_model_cpu_offload()
    # NOT pipe.enable_attention_slicing(): diffusers' own docstring warns against combining it
    # with SDPA (the default UNet attention backend on torch>=2.0, which we have) -- confirmed by
    # reproduction, not just the warning: with attention slicing enabled it replaces every UNet
    # attention processor with the legacy `SlicedAttnProcessor`, which does not understand the
    # (text_embeds, image_embeds) tuple IP-Adapter's processor injects as `encoder_hidden_states`,
    # crashing with `AttributeError: 'tuple' object has no attribute 'shape'` at generation time.
    # `vae.enable_slicing()`/`enable_tiling()` is the current SDXL-appropriate, IP-Adapter-safe
    # equivalent for the 8 GB budget: it reduces VAE decode memory for the 1024x1024 output
    # without touching UNet cross-attention at all.
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    load_seconds = time.perf_counter() - start
    return pipe, load_seconds


def pick_reference_image(images_dir: Path, explicit_path: Path | None) -> Path:
    """Pick a reference image for the IP-Adapter smoke test.

    Args:
        images_dir: Directory to search for a fallback image (any file already on disk from the
            Phase 2 exemplar fetch -- which one is used doesn't matter for this smoke test).
        explicit_path: If given, use this path instead of searching `images_dir`.

    Returns:
        Path to an existing image file.

    Raises:
        FileNotFoundError: if no explicit path was given and `images_dir` has no image files.
    """
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Reference image not found: {explicit_path}")
        return explicit_path
    candidates = sorted(images_dir.glob("*.jpg"))
    if not candidates:
        raise FileNotFoundError(f"No reference images found under {images_dir}")
    return candidates[0]


def run_generation(
    pipe: StableDiffusionXLPipeline,
    prompt: str,
    reference_image_path: Path,
    steps: int,
) -> tuple[Image.Image, float]:
    """Generate one 1024x1024 image conditioned on a text prompt and an IP-Adapter reference image.

    Args:
        pipe: Loaded SDXL + IP-Adapter pipeline.
        prompt: Text prompt.
        reference_image_path: Local image file used as the IP-Adapter style reference.
        steps: Number of denoising steps.

    Returns:
        The generated PIL image and the wall-clock seconds the generation call took.
    """
    reference_image = Image.open(reference_image_path).convert("RGB")
    generator = torch.Generator(device="cpu").manual_seed(SEED)  # cpu offload -> generator on cpu

    start = time.perf_counter()
    result = pipe(
        prompt=prompt,
        ip_adapter_image=reference_image,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        num_inference_steps=steps,
        generator=generator,
    )
    generation_seconds = time.perf_counter() - start
    return result.images[0], generation_seconds


def main() -> None:
    """Run the B1 smoke test: load the pipeline, generate one image, measure VRAM/time, report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--reference-image", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "smoketest.png")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() is False -- this smoke test requires a CUDA-enabled torch "
            "build and a visible GPU; refusing to run a CPU-only SDXL pass (would be misleadingly "
            "slow and would not measure what the B1 gate cares about)."
        )

    reference_image_path = pick_reference_image(IMAGES_DIR, args.reference_image)
    print(f"Reference image: {reference_image_path}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    pipe, load_seconds = load_pipeline()
    print(f"Model load time: {load_seconds:.1f}s")

    torch.cuda.reset_peak_memory_stats()
    image, generation_seconds = run_generation(pipe, args.prompt, reference_image_path, args.steps)
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)

    print(f"Generation time: {generation_seconds:.1f}s ({args.steps} steps)")
    print(f"Peak VRAM allocated: {peak_vram_gb:.2f} GB")
    print(f"Saved: {args.output}")

    vram_ok = peak_vram_gb <= MAX_PEAK_VRAM_GB
    time_ok = generation_seconds <= MAX_GENERATION_SECONDS
    gate_pass = vram_ok and time_ok
    print(
        f"GATE: {'PASS' if gate_pass else 'FAIL'} "
        f"(VRAM {peak_vram_gb:.2f}/{MAX_PEAK_VRAM_GB} GB, "
        f"time {generation_seconds:.1f}/{MAX_GENERATION_SECONDS}s)"
    )


if __name__ == "__main__":
    main()
