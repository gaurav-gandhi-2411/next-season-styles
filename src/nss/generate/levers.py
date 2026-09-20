"""Reference-conditioning levers for SDXL + IP-Adapter Plus (task N1).

ROOT CAUSE this module addresses: generation conditioned on `references[0]` alone through
IP-Adapter, whose image tokens dominate the text prompt. Two rounds x 4 seeds x 3 styles gave zero
briefed design changes. IP-Adapter is a style-transfer tool; the task is design variation, so the
reference must CONSTRAIN (archetype, category, colour) and the text must DRIVE (the change).

API behaviour, verified against the installed `diffusers==0.40.0` sources (not assumed):

- Multi-reference (native): `ip_adapter_image=[[img_1 .. img_N]]` -- one nested list per loaded
  adapter. `encode_image` encodes all N with the CLIP-ViT-H encoder (`hidden_states[-2]`, 257
  tokens each), the UNet's `encoder_hid_proj` projects each image separately, and
  `IPAdapterAttnProcessor2_0` (no masks) runs `to_k_ip`/`to_v_ip` on a 4-D
  `(batch, N, tokens, dim)` tensor and views the keys as `(batch, N*tokens, heads, head_dim)`:
  the N images' tokens are CONCATENATED into one attention, so text-conditioned queries attend over
  the union (an archetype mixture) rather than one garment. Mode ``concat``.
- Multi-reference (mean): average the N images' encoder hidden states BEFORE the resampler and pass
  the result through `ip_adapter_image_embeds` (shape `(2, 1, 257, 1280)`: negative then positive,
  since `prepare_ip_adapter_image_embeds` chunks CFG halves on dim 0). One synthetic
  "average garment" image. Mode ``mean``.
- Per-block scale: `pipe.set_ip_adapter_scale(scale)` accepts a float, or a dict keyed by
  "down"/"mid"/"up" -> {"block_i": [per-attention-layer scales]} (InstantStyle: style lives in
  `up.block_0[1]`, layout in `down.block_2[1]`). Supported by the installed version.
- Prompt weighting: diffusers itself has no `(clause)1.3` syntax; `compel` provides it for SDXL
  via explicit `prompt_embeds`/`pooled_prompt_embeds`.

Usage:
    uv run python -m nss.generate.levers run <experiment>
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from nss.generate import backends

OUT_ROOT = Path("data/generated/n1_levers")
# InstantStyle-style block scales for SDXL: style block only / style + layout.
STYLE_ONLY = "style_only"
STYLE_AND_LAYOUT = "style_layout"


def block_scale(kind: str, strength: float) -> dict[str, dict[str, list[float]]]:
    """Per-block IP-Adapter scale dict for `kind` at `strength` (see module docstring)."""
    if kind == STYLE_ONLY:
        return {"up": {"block_0": [0.0, strength, 0.0]}}
    if kind == STYLE_AND_LAYOUT:
        return {"down": {"block_2": [0.0, strength]}, "up": {"block_0": [0.0, strength, 0.0]}}
    raise ValueError(f"unknown block-scale kind {kind!r}")


def _mean_embeds(pipe: Any, images: Sequence[Image.Image]) -> list[Any]:
    """`ip_adapter_image_embeds` for ONE adapter from the mean of the N images' hidden states."""
    import torch

    device = pipe._execution_device
    pos, neg = pipe.encode_image(list(images), device, 1, True)  # each (N, 257, 1280)
    pos_mean = pos.mean(dim=0, keepdim=True)[None]  # (1, 1, 257, 1280)
    neg_mean = neg.mean(dim=0, keepdim=True)[None]
    return [torch.cat([neg_mean, pos_mean], dim=0)]


def generate_variant(
    prompt: str,
    negative_prompt: str,
    prompt_2: str | None,
    references: Sequence[Path],
    *,
    mode: str,
    scale: float | dict[str, Any],
    seed: int,
    weighted_prompt_embeds: dict[str, Any] | None = None,
) -> tuple[Image.Image, float]:
    """One image. `mode` in {"single", "concat", "mean"}; `scale` float or per-block dict.

    `weighted_prompt_embeds` (from `compel_embeds`) replaces the plain text prompt when given.
    """
    import torch

    pipe = backends._local_sdxl_pipeline()
    pipe.set_ip_adapter_scale(scale)
    images = [Image.open(p).convert("RGB") for p in references]
    kwargs: dict[str, Any] = {}
    if mode == "single":
        kwargs["ip_adapter_image"] = images[0]
    elif mode == "concat":
        kwargs["ip_adapter_image"] = [images]
    elif mode == "mean":
        kwargs["ip_adapter_image_embeds"] = _mean_embeds(pipe, images)
        # `encode_image` outside a pipeline call leaves the ViT-H encoder resident on the GPU next
        # to the UNet; on 8 GB that spills into shared memory (measured: 17.7 s/step vs 0.6).
        # The offload chain never moves it back (embeds are precomputed, so its hook never fires
        # again): send it to CPU by hand before the real call.
        pipe.image_encoder.to("cpu")
        torch.cuda.empty_cache()
    else:
        raise ValueError(f"unknown mode {mode!r}")
    if weighted_prompt_embeds is None:
        kwargs.update(
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
        )
    else:
        kwargs.update(weighted_prompt_embeds)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    start = time.perf_counter()
    result = pipe(
        height=backends.IMAGE_SIZE,
        width=backends.IMAGE_SIZE,
        num_inference_steps=backends.LOCAL_SDXL_STEPS,
        num_images_per_prompt=1,
        generator=generator,
        **kwargs,
    )
    return result.images[0], time.perf_counter() - start


def compel_embeds(prompt: str, negative_prompt: str) -> dict[str, Any]:
    """SDXL prompt/pooled embeddings with `compel` weighting syntax, e.g. `(funnel neck)1.5`."""
    from compel import Compel, ReturnedEmbeddingsType

    pipe = backends._local_sdxl_pipeline()
    # compel calls the encoders' internals directly, bypassing accelerate's offload hooks (which
    # only fire on `forward`): move them to the GPU by hand for the encode, then back.
    pipe.text_encoder.to("cuda")
    pipe.text_encoder_2.to("cuda")
    compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
        device="cuda",
        truncate_long_prompts=False,
    )
    cond, pooled = compel(prompt)
    neg, neg_pooled = compel(negative_prompt)
    cond, neg = compel.pad_conditioning_tensors_to_same_length([cond, neg])
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    return {
        "prompt_embeds": cond,
        "pooled_prompt_embeds": pooled,
        "negative_prompt_embeds": neg,
        "negative_pooled_prompt_embeds": neg_pooled,
    }
