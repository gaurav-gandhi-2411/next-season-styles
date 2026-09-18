from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from nss.generate import clip_scoring
from nss.generate.clip_scoring import (
    _cosine,
    _mean_max_cosine_similarity,
    clip_similarity,
    control_similarity,
    is_in_band,
)


def test_cosine_identical_vectors_is_one() -> None:
    """Cosine similarity of a vector with itself is 1.0."""
    a = np.array([1.0, 2.0, 3.0])
    assert _cosine(a, a) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors_is_zero() -> None:
    """Perpendicular vectors have cosine similarity 0."""
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert _cosine(a, b) == pytest.approx(0.0)


def test_cosine_opposite_vectors_is_negative_one() -> None:
    """Antiparallel vectors have cosine similarity -1."""
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert _cosine(a, b) == pytest.approx(-1.0)


def test_cosine_is_scale_invariant() -> None:
    """Cosine similarity depends only on direction, not magnitude."""
    a = np.array([1.0, 1.0])
    b = np.array([10.0, 10.0])
    assert _cosine(a, b) == pytest.approx(1.0)


def test_cosine_zero_vector_raises() -> None:
    """A zero-norm vector makes cosine similarity undefined -- must raise, not divide by zero."""
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    with pytest.raises(ValueError, match="zero-norm"):
        _cosine(a, b)


def test_mean_max_cosine_similarity_hand_computed() -> None:
    """query=[1,0] vs refs [1,0] (cos=1) and [0,1] (cos=0) -> mean=0.5, max=1.0."""
    query = np.array([1.0, 0.0])
    refs = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    result = _mean_max_cosine_similarity(query, refs)
    assert result == {"mean": pytest.approx(0.5), "max": pytest.approx(1.0)}


def test_mean_max_cosine_similarity_single_reference() -> None:
    """With one reference, mean == max == that single cosine similarity."""
    query = np.array([1.0, 0.0])
    refs = [np.array([0.0, 1.0])]
    result = _mean_max_cosine_similarity(query, refs)
    assert result["mean"] == pytest.approx(0.0)
    assert result["max"] == pytest.approx(0.0)


def test_mean_max_cosine_similarity_empty_references_raises() -> None:
    """An empty reference set is a caller error, not a silent 0.0/NaN."""
    with pytest.raises(ValueError, match="non-empty"):
        _mean_max_cosine_similarity(np.array([1.0, 0.0]), [])


def _fake_embed_image(vectors: dict[Path, np.ndarray]):
    """Build an `embed_image`-shaped stand-in that returns fixed vectors, no real CLIP model."""

    def _embed(path: Path) -> np.ndarray:
        return vectors[path]

    return _embed


def test_clip_similarity_computes_mean_max_over_embedded_references() -> None:
    """clip_similarity embeds the concept + each reference and reduces via mean/max cosine."""
    concept, ref1, ref2 = Path("concept.png"), Path("ref1.jpg"), Path("ref2.jpg")
    vectors = {
        concept: np.array([1.0, 0.0]),
        ref1: np.array([1.0, 0.0]),
        ref2: np.array([0.0, 1.0]),
    }
    with patch.object(clip_scoring, "embed_image", side_effect=_fake_embed_image(vectors)):
        result = clip_similarity(concept, [ref1, ref2])
    assert result == {"mean": pytest.approx(0.5), "max": pytest.approx(1.0)}


def test_control_similarity_same_shape_as_clip_similarity() -> None:
    """control_similarity is computationally identical to clip_similarity, given control images."""
    concept, control = Path("concept.png"), Path("control.jpg")
    vectors = {concept: np.array([1.0, 0.0]), control: np.array([0.0, 1.0])}
    with patch.object(clip_scoring, "embed_image", side_effect=_fake_embed_image(vectors)):
        result = control_similarity(concept, [control])
    assert result == {"mean": pytest.approx(0.0), "max": pytest.approx(0.0)}


def test_is_in_band_true_when_separated_and_within_band() -> None:
    """Own-style mean clears the control by >= margin and sits inside [lower, upper] -> True."""
    with (
        patch.object(clip_scoring, "clip_similarity", return_value={"mean": 0.7, "max": 0.8}),
        patch.object(clip_scoring, "control_similarity", return_value={"mean": 0.5, "max": 0.6}),
    ):
        assert (
            is_in_band(
                Path("c.png"),
                [Path("r.jpg")],
                [Path("ctrl.jpg")],
                lower=0.6,
                upper=0.8,
                margin=0.1,
            )
            is True
        )


def test_is_in_band_false_when_margin_not_met() -> None:
    """Inside the band but the gap over the control set is smaller than margin -> False."""
    with (
        patch.object(clip_scoring, "clip_similarity", return_value={"mean": 0.65, "max": 0.8}),
        patch.object(clip_scoring, "control_similarity", return_value={"mean": 0.6, "max": 0.6}),
    ):
        assert (
            is_in_band(
                Path("c.png"),
                [Path("r.jpg")],
                [Path("ctrl.jpg")],
                lower=0.5,
                upper=0.9,
                margin=0.1,
            )
            is False
        )


def test_is_in_band_false_when_below_lower_bound() -> None:
    """Margin is cleared but own-style mean sits below `lower` -> False."""
    with (
        patch.object(clip_scoring, "clip_similarity", return_value={"mean": 0.4, "max": 0.5}),
        patch.object(clip_scoring, "control_similarity", return_value={"mean": 0.1, "max": 0.2}),
    ):
        assert (
            is_in_band(
                Path("c.png"),
                [Path("r.jpg")],
                [Path("ctrl.jpg")],
                lower=0.5,
                upper=0.9,
                margin=0.1,
            )
            is False
        )


def test_is_in_band_false_when_above_upper_bound() -> None:
    """Margin is cleared but own-style mean exceeds `upper` (near-duplicate regime) -> False."""
    with (
        patch.object(clip_scoring, "clip_similarity", return_value={"mean": 0.95, "max": 0.99}),
        patch.object(clip_scoring, "control_similarity", return_value={"mean": 0.5, "max": 0.5}),
    ):
        assert (
            is_in_band(
                Path("c.png"),
                [Path("r.jpg")],
                [Path("ctrl.jpg")],
                lower=0.5,
                upper=0.9,
                margin=0.1,
            )
            is False
        )


def test_is_in_band_boundary_values_are_inclusive() -> None:
    """own_mean exactly == lower and (own - control) exactly == margin still counts -> True.

    Uses 0.75/0.5/0.25 (all exactly representable in binary floating point) so the boundary
    comparison is exact, not incidentally failing on float rounding.
    """
    with (
        patch.object(clip_scoring, "clip_similarity", return_value={"mean": 0.75, "max": 0.75}),
        patch.object(clip_scoring, "control_similarity", return_value={"mean": 0.5, "max": 0.5}),
    ):
        assert (
            is_in_band(
                Path("c.png"),
                [Path("r.jpg")],
                [Path("ctrl.jpg")],
                lower=0.75,
                upper=0.9,
                margin=0.25,
            )
            is True
        )
