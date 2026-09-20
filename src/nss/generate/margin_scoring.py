"""Margin-based novelty scoring primitive, embedding-space-agnostic.

Replaces the earlier absolute-cosine "in-band" approach (`nss.generate.clip_scoring.is_in_band` +
`nss.generate.derive_similarity_band`). The earlier band `[0.8906, 0.9401]` was derived from
catalogue-photo-vs-catalogue-photo pairs (within/across *real* styles) but applied to
generated-image-vs-catalogue-photo pairs -- a different distribution. Across-style catalogue
photos already share H&M's flat-lay, white-background product-photography conventions, which
inflates raw cosine similarity between genuinely unrelated styles (measured mean 0.799 across
styles in the CLIP space that fed the earlier band) -- an absolute cosine threshold calibrated on
that inflated floor is not a trustworthy novelty signal for a generated image, whose photographic
style may differ from catalogue conventions in ways that have nothing to do with style relatedness.

MARGIN construction: for a query embedding (generated concept OR, for anchor derivation, a
held-out real image), subtract its mean similarity to an unrelated "control" pool from its mean
similarity to its own style's references:

    margin = mean_sim(concept, style_references) - mean_sim(concept, control_pool_references)

This is robust to the shared-photography-style confound by construction: any inflation from
shared photographic conventions raises BOTH terms roughly equally (the query is equally "generic
product photo"-like to its own style and to the control pool on that axis alone), so it mostly
cancels in the subtraction. What survives is the incremental similarity attributable to genuine
style relatedness. See `nss.generate.derive_margin_band` for how the margin distribution is used
to derive the actual `[lower, upper]` in-band thresholds (the equivalent of the earlier band, now
margin-based).

Embedding-space agnostic by design: `margin()` operates on already-computed embedding vectors, not
image paths -- the caller supplies embeddings from whichever model (CLIP via
`nss.generate.clip_scoring.embed_image`, DINOv2 via `nss.generate.dino_scoring.embed_image`, or any
future embedder producing L2-normalizable vectors). This lets the same margin math be reused
verbatim for both embedding spaces, rather than duplicating it per model.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from nss.generate.clip_scoring import _cosine


def margin(
    concept_embedding: np.ndarray,
    style_reference_embeddings: Sequence[np.ndarray],
    control_pool_embeddings: Sequence[np.ndarray],
) -> float:
    """Margin = mean cosine similarity to style references minus mean cosine similarity to control.

    Args:
        concept_embedding: 1D embedding vector for the query image (generated concept, or a
            held-out real image when deriving anchor distributions).
        style_reference_embeddings: Non-empty sequence of 1D embedding vectors for the query's
            target style's real reference images.
        control_pool_embeddings: Non-empty sequence of 1D embedding vectors for an unrelated
            control pool's real images.

    Returns:
        `mean_sim(concept, style_references) - mean_sim(concept, control_pool)`. Positive means
        the query is more similar to its claimed style than to the control pool; near zero means
        statistically indistinguishable from an unrelated pairing (see
        `nss.generate.derive_margin_band`'s lower-anchor validation of this exact expectation).

    Raises:
        ValueError: if either reference sequence is empty (cosine similarity against an empty set
            is undefined, not silently 0.0 -- same fail-loud convention as
            `nss.generate.clip_scoring._mean_max_cosine_similarity`).
    """
    if not style_reference_embeddings:
        raise ValueError("style_reference_embeddings must be non-empty")
    if not control_pool_embeddings:
        raise ValueError("control_pool_embeddings must be non-empty")

    style_mean = float(
        np.mean([_cosine(concept_embedding, ref) for ref in style_reference_embeddings])
    )
    control_mean = float(
        np.mean([_cosine(concept_embedding, ref) for ref in control_pool_embeddings])
    )
    return style_mean - control_mean
