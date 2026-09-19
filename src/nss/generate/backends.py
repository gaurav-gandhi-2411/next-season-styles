"""Backend-agnostic concept-image generation interface (task B2).

`generate_concept()` dispatches to one of two generation backends behind a single signature, so
downstream code (the B3/B4 generation + evaluation sweeps) never branches on which backend
produced an image:

- ``"local_sdxl"``: SDXL + IP-Adapter Plus running locally on-GPU. Reuses the exact configuration
  the B1 smoke test (`nss.generate.local_sdxl_smoketest`) proved viable on an RTX 3070 (8 GB VRAM):
  fp16 weights, `enable_model_cpu_offload()`, the ViT-H `image_encoder_folder` fix, and VAE-level
  slicing/tiling instead of `enable_attention_slicing()` (the latter crashes IP-Adapter -- see that
  module's docstring for the mechanism).
- ``"gemini"``: Gemini 2.5 Flash Image (`gemini-2.5-flash-image`) via the `google-genai` SDK,
  called over the network. Requires `GEMINI_API_KEY` in a repo-root `.env` file.

API-shape notes (verified against the installed package sources, not assumed):

1. IP-Adapter multi-reference handling: `diffusers==0.40.0`'s
   `StableDiffusionXLPipeline.prepare_ip_adapter_image_embeds` requires
   ``len(ip_adapter_image) == len(unet.encoder_hid_proj.image_projection_layers)`` -- i.e. exactly
   one image per *loaded* IP-Adapter, not one embedding averaged/pooled across images. Since B2
   loads a single IP-Adapter, only the first `reference_images` entry is usable per call; passing
   more than one would need to `load_ip_adapter()` with one adapter per reference image, which is
   out of scope here. Documented, not silently ignored -- see `_generate_local_sdxl`.
2. `google-genai==2.24.0`'s content transformers (`_transformers.t_part`) accept a `PIL.Image.Image`
   directly inside a `contents=[...]` list (auto-converted to an inline-data `Part`), so reference
   images are passed as ``[prompt_text, *PIL_images]`` -- no manual base64/Part wrapping needed.
   The `gemini-2.5-flash-image` model id and the `response_modalities=["IMAGE"]` config field were
   confirmed against the SDK's own bundled test fixtures
   (`google/genai/tests/models/test_generate_content_image_generation.py`), since neither is
   documented in the public docstrings.
"""

from __future__ import annotations

import io
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from nss.generate.local_sdxl_smoketest import (
    IP_ADAPTER_IMAGE_ENCODER_FOLDER,
    IP_ADAPTER_REPO,
    IP_ADAPTER_SUBFOLDER,
    IP_ADAPTER_WEIGHT_NAME,
    MODEL_ID,
)

LOCAL_SDXL = "local_sdxl"
GEMINI = "gemini"
SUPPORTED_BACKENDS = (LOCAL_SDXL, GEMINI)

GEMINI_MODEL_ID = "gemini-2.5-flash-image"

OUTPUT_ROOT = Path("data/generated")
IMAGE_SIZE = 1024  # SDXL native resolution; matches the B1 smoke test
LOCAL_SDXL_STEPS = 30  # matches DEFAULT_STEPS in the B1 smoke test

# Loading SDXL + IP-Adapter takes tens of seconds; cache the pipeline across repeated
# generate_concept() calls in the same process (e.g. a B4 parameter sweep) instead of reloading it
# every call. Keyed by backend name purely so the pattern extends cleanly if a second local backend
# is ever added -- only "local_sdxl" populates it today.
_PIPELINE_CACHE: dict[str, Any] = {}


def _local_sdxl_pipeline() -> Any:
    """Load (once, cached) the SDXL + IP-Adapter Plus pipeline used by the ``local_sdxl`` backend.

    Returns:
        A `diffusers.StableDiffusionXLPipeline` with IP-Adapter Plus attached, fp16 weights, CPU
        offload, and VAE slicing/tiling enabled -- the exact B1-smoke-test-proven configuration.

    Raises:
        RuntimeError: if no CUDA GPU is visible (this backend has not been validated on CPU and
            would be misleadingly slow).
    """
    if LOCAL_SDXL in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[LOCAL_SDXL]

    import torch
    from diffusers import StableDiffusionXLPipeline

    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() is False -- the local_sdxl backend requires a CUDA-enabled "
            "torch build and a visible GPU."
        )

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
    pipe.enable_model_cpu_offload()
    # NOT enable_attention_slicing() -- crashes IP-Adapter's tuple encoder_hidden_states. See the
    # B1 smoke test's `load_pipeline()` docstring for the full mechanism.
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    _PIPELINE_CACHE[LOCAL_SDXL] = pipe
    return pipe


def _generate_local_sdxl(
    prompt: str,
    reference_images: list[Path],
    ip_adapter_scale: float | None,
    seed: int,
    n: int,
    negative_prompt: str | None = None,
) -> tuple[list[Image.Image], float]:
    """Generate `n` images with the local SDXL + IP-Adapter Plus backend.

    Args:
        prompt: Text prompt.
        reference_images: IP-Adapter reference image paths. Only the first is used -- see module
            docstring note 1 for why (a single loaded IP-Adapter accepts exactly one image).
        ip_adapter_scale: IP-Adapter conditioning strength. `None` leaves diffusers' own default
            (1.0, set implicitly by `load_ip_adapter`) rather than calling `set_ip_adapter_scale`.
        seed: Seed for a single `torch.Generator` shared across the `n`-image batch.
        n: Number of images to generate (one `pipe()` call with `num_images_per_prompt=n`).
        negative_prompt: Optional negative prompt passed straight through to the SDXL pipeline's
            own `negative_prompt` argument (standard classifier-free-guidance negative
            conditioning). `None` (the default) leaves diffusers' own default of no negative
            conditioning -- behavior-identical to callers written before this parameter existed.

    Returns:
        The `n` generated PIL images and the wall-clock seconds the batched generation call took.
    """
    import torch

    pipe = _local_sdxl_pipeline()
    if ip_adapter_scale is not None:
        pipe.set_ip_adapter_scale(ip_adapter_scale)

    reference_image = Image.open(reference_images[0]).convert("RGB")
    # cpu offload keeps the generator on cpu, matching the B1 smoke test.
    generator = torch.Generator(device="cpu").manual_seed(seed)

    start = time.perf_counter()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        ip_adapter_image=reference_image,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        num_inference_steps=LOCAL_SDXL_STEPS,
        num_images_per_prompt=n,
        generator=generator,
    )
    elapsed = time.perf_counter() - start
    return list(result.images), elapsed


def _require_gemini_api_key() -> str:
    """Load `.env` (if present) and return `GEMINI_API_KEY`, or raise with an actionable message.

    Returns:
        The API key.

    Raises:
        RuntimeError: if `GEMINI_API_KEY` is not set after attempting to load a repo-root `.env`.
    """
    from dotenv import load_dotenv

    load_dotenv()  # no-op if no .env is found; never overrides an already-set env var
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Create a .env file at the repo root containing "
            "GEMINI_API_KEY=<your key> (get a key from Google AI Studio: "
            "https://aistudio.google.com/apikey). Never commit this file."
        )
    return api_key


def _extract_image(response: Any) -> Image.Image:
    """Pull the first inline image out of a `generate_content` response.

    Args:
        response: A `google.genai.types.GenerateContentResponse`.

    Returns:
        The decoded PIL image.

    Raises:
        RuntimeError: if the response has no candidates/parts, or no part is an image -- Gemini
            can return a text-only response (e.g. a refusal), which must not be mistaken for an
            image and silently produce an empty/garbage output file.
    """
    candidates = response.candidates or []
    if not candidates or not candidates[0].content or not candidates[0].content.parts:
        finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
        raise RuntimeError(
            "Gemini response contained no candidates/parts -- cannot extract an image "
            f"(finish_reason={finish_reason})."
        )
    for part in candidates[0].content.parts:
        generated = part.as_image()
        if generated is not None and generated.image_bytes is not None:
            return Image.open(io.BytesIO(generated.image_bytes)).convert("RGB")
    raise RuntimeError("Gemini response contained no image part (text-only reply or refusal?).")


def _generate_gemini(
    prompt: str,
    reference_images: list[Path],
    seed: int,
    n: int,
) -> tuple[list[Image.Image], float]:
    """Generate `n` images with the Gemini 2.5 Flash Image backend.

    Args:
        prompt: Text prompt.
        reference_images: Images passed as additional multimodal `contents` alongside the prompt
            (see module docstring note 2) -- all of them, unlike `local_sdxl`'s single-image limit.
        seed: Base seed; `seed + i` is passed as `GenerateContentConfig.seed` for image `i`. Google
            does not document/guarantee reproducibility for image-generation outputs even with
            `seed` set, unlike the `local_sdxl` backend's `torch.Generator` -- best-effort only.
        n: Number of images to generate (one `generate_content` call per image; the SDK's
            `candidate_count` is a single-call alternative but was not used here to keep each
            image's generation time independently measurable).

    Returns:
        The `n` generated PIL images and the wall-clock seconds all `n` calls took combined.
    """
    from google import genai
    from google.genai import types as genai_types

    api_key = _require_gemini_api_key()
    client = genai.Client(api_key=api_key)
    reference_pil = [Image.open(p).convert("RGB") for p in reference_images]
    contents: list[Any] = [prompt, *reference_pil]

    images: list[Image.Image] = []
    start = time.perf_counter()
    for i in range(n):
        response = client.models.generate_content(
            model=GEMINI_MODEL_ID,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                seed=seed + i,
            ),
        )
        images.append(_extract_image(response))
    elapsed = time.perf_counter() - start
    return images, elapsed


def _write_metadata(path: Path, **fields: Any) -> None:
    """Write a JSON metadata sidecar for one generated image.

    Args:
        path: Output `.json` path.
        **fields: Arbitrary JSON-serializable fields to record (backend, prompt, seed, etc.). A
            `generated_at` UTC timestamp is added automatically.
    """
    payload = {**fields, "generated_at": datetime.now(UTC).isoformat()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def generate_concept(
    prompt: str,
    reference_images: list[Path],
    backend: str,
    ip_adapter_scale: float | None,
    seed: int,
    n: int,
    negative_prompt: str | None = None,
) -> list[Path]:
    """Generate `n` fashion concept images from `prompt` (+ `reference_images`) via `backend`.

    Args:
        prompt: Text prompt describing the fashion concept to generate.
        reference_images: Local image paths used as style/content references. Must be non-empty.
            See module docstring notes 1-2 for how each backend handles more than one reference.
        backend: One of `SUPPORTED_BACKENDS` (`"local_sdxl"`, `"gemini"`).
        ip_adapter_scale: IP-Adapter conditioning strength, applies to `"local_sdxl"` only. Must
            be `None` for `"gemini"` (raises `ValueError` otherwise) -- Gemini's image API has no
            equivalent knob, and a non-None value would otherwise look honoured when it silently
            wasn't.
        seed: Base random seed (deterministic for `local_sdxl`; best-effort for `gemini`).
        n: Number of images to generate.
        negative_prompt: Optional negative prompt, `"local_sdxl"` only (passed straight through to
            the SDXL pipeline's own `negative_prompt` argument). Must be `None` for `"gemini"`
            (raises `ValueError` otherwise) -- Gemini's image API has no negative-prompt
            equivalent, same fail-loud convention as `ip_adapter_scale` above.

    Returns:
        Paths to the `n` saved PNG files under `data/generated/<backend>/`. A JSON metadata
        sidecar (same stem, `.json`) is written next to each image recording backend, prompt,
        negative_prompt, reference image paths, seed, the `ip_adapter_scale` actually applied
        (`null` for `"gemini"`), and the batch generation time.

    Raises:
        ValueError: unknown `backend`, empty `reference_images`, `n < 1`, a non-`None`
            `ip_adapter_scale` passed with `backend="gemini"`, or a non-`None` `negative_prompt`
            passed with `backend="gemini"`.
        RuntimeError: missing `GEMINI_API_KEY` (`backend="gemini"`) or no CUDA GPU visible
            (`backend="local_sdxl"`).
    """
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; supported: {SUPPORTED_BACKENDS}")
    if not reference_images:
        raise ValueError("reference_images must be non-empty.")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if backend == GEMINI and ip_adapter_scale is not None:
        raise ValueError(
            "ip_adapter_scale is not applicable to the gemini backend and must be None -- "
            f"got {ip_adapter_scale!r}."
        )
    if backend == GEMINI and negative_prompt is not None:
        raise ValueError(
            "negative_prompt is not applicable to the gemini backend and must be None -- "
            f"got {negative_prompt!r}."
        )

    if backend == LOCAL_SDXL:
        images, elapsed = _generate_local_sdxl(
            prompt, reference_images, ip_adapter_scale, seed, n, negative_prompt
        )
        effective_scale = ip_adapter_scale
    else:
        images, elapsed = _generate_gemini(prompt, reference_images, seed, n)
        effective_scale = None

    output_dir = OUTPUT_ROOT / backend
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for i, image in enumerate(images):
        stem = f"seed{seed}_{i:02d}"
        image_path = output_dir / f"{stem}.png"
        image.save(image_path)
        _write_metadata(
            output_dir / f"{stem}.json",
            backend=backend,
            prompt=prompt,
            negative_prompt=negative_prompt,
            reference_images=[str(p) for p in reference_images],
            ip_adapter_scale=effective_scale,
            seed=seed,
            index=i,
            batch_generation_seconds=elapsed,
        )
        paths.append(image_path)
    return paths
