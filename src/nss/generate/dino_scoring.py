"""DINOv2-based image embeddings, the second embedding space for margin scoring (task C2).

Companion to `nss.generate.clip_scoring`: same `embed_image(path) -> L2-normalized np.ndarray`
shape, different backbone (`facebook/dinov2-base` via `transformers`' `AutoModel`/
`AutoImageProcessor`, not `CLIPModel`/`CLIPProcessor`). `transformers>=5.17.0` is already a
`pyproject.toml` dependency (see `clip_scoring`'s docstring), so this module needs zero new
dependencies beyond downloading the `dinov2-base` weights themselves (~330MB) at first use.

WHY DINOv2 ALONGSIDE CLIP (the actual reason this module exists): CLIP is trained contrastively
against natural-language captions, so its embedding space is comparatively sensitive to
whole-image "gestalt" cues that correlate with how a caption would describe a photo -- including
shared product-photography conventions (white background, flat-lay framing, centered garment)
that inflate raw cosine similarity between genuinely different H&M styles regardless of actual
style relatedness (the B3 confound `nss.generate.margin_scoring`'s docstring describes: mean 0.799
CLIP cosine similarity across styles that share nothing but photography convention). DINOv2 is
trained with a self-supervised, purely visual objective (no paired-caption supervision) that
explicitly targets object/part-level structure -- shape, texture, local geometry -- over
image-level "what would this be captioned as" semantics. Because it doesn't key on captionable
gestalt the way CLIP does, DINOv2's embedding space is EXPECTED to be comparatively less inflated
by shared photographic convention alone, which should translate into WIDER separation between
genuine own-style margin and cross-style/control margin than CLIP shows for the identical image
set. This is a stated expectation, not an assumed conclusion -- `nss.generate.derive_margin_band`
measures both spaces' anchor distributions independently and reports explicitly whether DINOv2
actually shows wider separation, rather than assuming it.

VRAM/process isolation (same hard constraint as `clip_scoring.py`, copied verbatim because it
applies identically here): this module must NEVER be imported into, or run in the same process as,
`nss.generate.backends.generate_concept` / `_local_sdxl_pipeline`. This machine has 8GB VRAM and
SDXL alone peaks at ~6.5GB; run DINOv2 scoring as a fully separate process/script invocation from
any SDXL generation. The model is loaded CPU-only here (see `_load_dino`), so scoring never
contends for VRAM even if accidentally run alongside a generation process.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

DINO_MODEL_ID = "facebook/dinov2-base"


@lru_cache(maxsize=1)
def _load_dino() -> tuple[Any, Any]:
    """Load (once, process-cached) the DINOv2-base model + image processor, CPU-only.

    Returns:
        `(model, processor)` -- the model in eval mode.

    Note:
        Deliberately CPU-only, not "cuda if available" -- see module docstring VRAM/process
        isolation constraint, identical reasoning to `clip_scoring._load_clip`.
    """
    from transformers import AutoImageProcessor, AutoModel

    model = AutoModel.from_pretrained(DINO_MODEL_ID)
    model.eval()
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
    return model, processor


def embed_image(image_path: Path) -> np.ndarray:
    """Compute the L2-normalized DINOv2 image embedding for one image file.

    Args:
        image_path: Path to a local image file (JPEG/PNG).

    Returns:
        1D `float32` array (768-dim for `dinov2-base`), L2-normalized.

    Note:
        `Dinov2Model.forward` returns `pooler_output = layernorm(sequence_output)[:, 0, :]` --
        the CLS token after the model's final layernorm (confirmed against the installed
        `transformers==5.17.0` package source, `modeling_dinov2.py`'s `Dinov2Model.forward`, not
        assumed from memory) -- the standard whole-image embedding for DINOv2, analogous to
        CLIP's projected pooled image feature used in `clip_scoring.embed_image`.
    """
    import torch

    model, processor = _load_dino()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model(**inputs).pooler_output
    normalized = features / features.norm(p=2, dim=-1, keepdim=True)
    return normalized.squeeze(0).numpy()
