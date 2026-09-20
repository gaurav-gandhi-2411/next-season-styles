"""CLIP-based scoring of generated concept images against real reference images.

Scores a generated concept image's CLIP cosine similarity against (a) its target style's real
reference images and (b) an unrelated "control" style's real images, then combines the two into
an "in-band" novel-but-related determination: similar enough to its own style to be recognizably
on-trend, dissimilar enough from a near-duplicate reference (and from an unrelated style) to be
a genuinely novel generation rather than a copy.

Library choice: `transformers`' `CLIPModel`/`CLIPProcessor` (`openai/clip-vit-large-patch14`),
not `open_clip`. `transformers>=5.17.0` is already a `pyproject.toml` dependency (needed for the
SDXL/IP-Adapter generation stack in `nss.generate.backends`), so this module needs zero new
dependencies; `open_clip` is not installed, and adding a new dependency requires an explicit ask
per project policy. `transformers` loads the same OpenAI-released ViT-L/14 checkpoint `open_clip`
would (`openai/clip-vit-large-patch14`), so there is no scoring-quality difference for this task.

VRAM/process isolation (critical): this module must NEVER be imported into, or run in the same
process as, `nss.generate.backends.generate_concept` / `_local_sdxl_pipeline`. This machine has
8GB VRAM and SDXL alone peaks at ~6.5GB; run CLIP scoring as a fully separate process/script
invocation from any SDXL generation. The CLIP model itself is loaded CPU-only here (see
`_load_clip`), which also means scoring never contends for VRAM even if accidentally run
alongside a generation process.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"


@lru_cache(maxsize=1)
def _load_clip() -> tuple[Any, Any]:
    """Load (once, process-cached) the CLIP ViT-L/14 model + processor, CPU-only.

    Returns:
        `(model, processor)` -- the model in eval mode.

    Note:
        Deliberately CPU-only, not "cuda if available": scoring already-generated images on disk
        is not latency-critical (~1s/image on CPU), and this module must never claim VRAM that a
        concurrently-running `local_sdxl` generation process might need (see module docstring).
    """
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
    model.eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    return model, processor


def embed_image(image_path: Path) -> np.ndarray:
    """Compute the L2-normalized CLIP image embedding for one image file.

    Args:
        image_path: Path to a local image file (JPEG/PNG).

    Returns:
        1D `float32` array (768-dim for ViT-L/14), L2-normalized.
    """
    import torch

    model, processor = _load_clip()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        # transformers==5.17.0's CLIPModel.get_image_features returns a BaseModelOutputWithPooling
        # (or tuple), not a raw tensor as older docstrings/examples show -- confirmed against the
        # installed package source: the projected image embedding is written into `.pooler_output`
        # (`vision_outputs.pooler_output = self.visual_projection(pooled_output)`).
        features = model.get_image_features(**inputs).pooler_output
    normalized = features / features.norm(p=2, dim=-1, keepdim=True)
    return normalized.squeeze(0).numpy()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors (not assumed pre-normalized).

    Args:
        a: First vector.
        b: Second vector, same dimensionality as `a`.

    Returns:
        Cosine similarity in `[-1, 1]`.

    Raises:
        ValueError: if either vector has zero norm (cosine similarity is undefined).
    """
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        raise ValueError("cannot compute cosine similarity against a zero-norm vector")
    return float(np.dot(a, b) / denom)


def _mean_max_cosine_similarity(
    query: np.ndarray, references: Sequence[np.ndarray]
) -> dict[str, float]:
    """Cosine similarity between `query` and each of `references`; mean + max across the set.

    Pure numeric core of `clip_similarity`/`control_similarity`, factored out so it is unit
    testable on small hand-computable synthetic vectors without loading a real CLIP model.

    Args:
        query: 1D embedding vector.
        references: Non-empty sequence of 1D embedding vectors, same dimensionality as `query`.

    Returns:
        `{"mean": float, "max": float}` -- mean and max cosine similarity across `references`.

    Raises:
        ValueError: if `references` is empty.
    """
    if not references:
        raise ValueError("references must be non-empty")
    sims = [_cosine(query, ref) for ref in references]
    return {"mean": float(np.mean(sims)), "max": float(np.max(sims))}


def clip_similarity(concept_img: Path, reference_imgs: list[Path]) -> dict[str, float]:
    """CLIP cosine similarity between a generated concept image and its style's real references.

    Args:
        concept_img: Path to the generated concept image.
        reference_imgs: Paths to real reference images for the concept's target style (non-empty).

    Returns:
        `{"mean": float, "max": float}` cosine similarity across `reference_imgs`.
    """
    query = embed_image(concept_img)
    refs = [embed_image(p) for p in reference_imgs]
    return _mean_max_cosine_similarity(query, refs)


def control_similarity(concept_img: Path, control_style_imgs: list[Path]) -> dict[str, float]:
    """CLIP cosine similarity between a generated concept image and an unrelated control style.

    Computationally identical to `clip_similarity`, kept as a separate named function because it
    plays a distinct role in `is_in_band`: it establishes the noise floor a concept image's
    own-style similarity must clear by at least `margin` to count as "materially" related rather
    than nominally so.

    Args:
        concept_img: Path to the generated concept image.
        control_style_imgs: Paths to real images of a style unrelated to the concept (the
            `role == "control"` rows of `reports/tables/exemplar_images*.csv`).

    Returns:
        `{"mean": float, "max": float}` cosine similarity across `control_style_imgs`.
    """
    return clip_similarity(concept_img, control_style_imgs)


def is_in_band(
    concept_img: Path,
    reference_imgs: list[Path],
    control_imgs: list[Path],
    lower: float,
    upper: float,
    margin: float,
) -> bool:
    """Decide whether a generated concept image is a "novel-but-related" in-band result.

    Both conditions below must hold:

    1. **Own-style separation**: mean similarity to `reference_imgs` exceeds mean similarity to
       `control_imgs` by at least `margin`. A fixed margin -- rather than "any positive
       difference counts" -- is used because with only a handful of reference/control images per
       style, a near-zero gap is well within sampling noise and would make the check vacuous.
       `margin` is a required argument (no default baked into this module) so that the empirical
       distributions computed by `derive_similarity_band.py` remain the single source of truth
       for every numeric threshold this project uses -- e.g. passing half the across-style
       pairwise std as a data-driven margin.
    2. **Band membership**: the own-style mean similarity falls within `[lower, upper]` -- above
       `lower` (statistically indistinguishable from an unrelated style, i.e. noise) and below
       `upper` (statistically indistinguishable from a near-duplicate of an existing reference,
       i.e. not novel).

    Args:
        concept_img: Path to the generated concept image.
        reference_imgs: Real reference images for the concept's target style.
        control_imgs: Real images of an unrelated control style.
        lower: Lower band bound (empirically derived -- see `derive_similarity_band.py`).
        upper: Upper band bound (empirically derived -- see `derive_similarity_band.py`).
        margin: Minimum required `(own_mean - control_mean)` similarity gap.

    Returns:
        True iff both the separation and band-membership conditions hold.
    """
    own_mean = clip_similarity(concept_img, reference_imgs)["mean"]
    control_mean = control_similarity(concept_img, control_imgs)["mean"]
    return (own_mean - control_mean >= margin) and (lower <= own_mean <= upper)
