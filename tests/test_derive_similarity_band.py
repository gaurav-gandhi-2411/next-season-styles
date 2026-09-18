from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nss.generate.derive_similarity_band import (
    across_style_similarities,
    derive_band,
    group_by_style,
    load_exemplar_manifest,
    summarize,
    within_style_similarities,
)

_STYLE_A_IMG1 = Path("a1.jpg")
_STYLE_A_IMG2 = Path("a2.jpg")
_STYLE_B_IMG1 = Path("b1.jpg")
_STYLE_B_IMG2 = Path("b2.jpg")


def _style_to_images() -> dict[str, list[Path]]:
    return {"style_a": [_STYLE_A_IMG1, _STYLE_A_IMG2], "style_b": [_STYLE_B_IMG1, _STYLE_B_IMG2]}


def _embeddings() -> dict[Path, np.ndarray]:
    return {
        _STYLE_A_IMG1: np.array([1.0, 0.0]),
        _STYLE_A_IMG2: np.array([1.0, 0.0]),  # identical to A1 -> within-style cos = 1.0
        _STYLE_B_IMG1: np.array([0.0, 1.0]),
        _STYLE_B_IMG2: np.array([0.0, 1.0]),  # identical to B1 -> within-style cos = 1.0
    }


def test_within_style_similarities_pairs_only_same_style() -> None:
    """2 images per style, 2 styles -> exactly 1 within-style pair per style (2 total)."""
    sims = within_style_similarities(_style_to_images(), _embeddings())
    assert len(sims) == 2
    assert all(s == pytest.approx(1.0) for s in sims)


def test_within_style_similarities_skips_singleton_styles() -> None:
    """A style with only 1 image contributes zero within-style pairs."""
    style_to_images = {"solo": [_STYLE_A_IMG1]}
    embeddings = {_STYLE_A_IMG1: np.array([1.0, 0.0])}
    assert within_style_similarities(style_to_images, embeddings) == []


def test_across_style_similarities_pairs_only_different_styles() -> None:
    """2 images per style x 2 styles -> 2*2 = 4 across-style pairs, A orthogonal to B -> cos 0."""
    sims = across_style_similarities(_style_to_images(), _embeddings())
    assert len(sims) == 4
    assert all(s == pytest.approx(0.0) for s in sims)


def test_summarize_hand_computed() -> None:
    """[0.0, 0.5, 1.0] -> mean=0.5, median=0.5, min=0.0, max=1.0, n_pairs=3."""
    result = summarize([0.0, 0.5, 1.0])
    assert result["n_pairs"] == 3
    assert result["mean"] == pytest.approx(0.5)
    assert result["median"] == pytest.approx(0.5)
    assert result["min"] == pytest.approx(0.0)
    assert result["max"] == pytest.approx(1.0)


def test_summarize_empty_raises() -> None:
    """An empty distribution has no defined summary stats -- must raise, not fabricate zeros."""
    with pytest.raises(ValueError, match="non-empty"):
        summarize([])


def test_derive_band_uses_across_percentile_and_within_median() -> None:
    """lower = across-style p90, upper = within-style median -- read straight off the inputs."""
    within_summary = summarize([0.6, 0.7, 0.8])  # median = 0.7
    across_summary = summarize([0.1, 0.2, 0.9])  # p90 close to the top of a 3-point sample
    band = derive_band(within_summary, across_summary, lower_percentile=90.0)
    assert band["upper"] == pytest.approx(0.7)
    assert band["lower"] == pytest.approx(across_summary["p90"])
    assert band["lower_percentile_used"] == 90.0


def test_load_exemplar_manifest_dedupes_and_drops_fetch_failures(tmp_path: Path) -> None:
    """De-duplicates by article_id across manifests and drops fetch_success=False rows."""
    manifest_a = pl.DataFrame(
        [
            {
                "style_key": "style_a",
                "article_id": 1,
                "local_image_path": "img1.jpg",
                "fetch_success": True,
            },
            {
                "style_key": "style_a",
                "article_id": 2,
                "local_image_path": "img2.jpg",
                "fetch_success": False,  # never actually fetched -- must be dropped
            },
        ]
    )
    manifest_b = pl.DataFrame(
        [
            {
                "style_key": "style_a",
                "article_id": 1,  # duplicate of manifest_a's row -- must be deduped
                "local_image_path": "img1.jpg",
                "fetch_success": True,
            },
            {
                "style_key": "style_b",
                "article_id": 3,
                "local_image_path": "img3.jpg",
                "fetch_success": True,
            },
        ]
    )
    path_a = tmp_path / "manifest_a.csv"
    path_b = tmp_path / "manifest_b.csv"
    manifest_a.write_csv(path_a)
    manifest_b.write_csv(path_b)

    combined = load_exemplar_manifest((path_a, path_b))

    assert sorted(combined["article_id"].to_list()) == [1, 3]


def test_group_by_style_only_includes_existing_files(tmp_path: Path) -> None:
    """Rows whose local_image_path doesn't actually exist on disk are excluded defensively."""
    real_image = tmp_path / "real.jpg"
    real_image.write_bytes(b"fake-image-bytes")
    manifest = pl.DataFrame(
        [
            {"style_key": "style_a", "article_id": 1, "local_image_path": str(real_image)},
            {
                "style_key": "style_a",
                "article_id": 2,
                "local_image_path": str(tmp_path / "missing.jpg"),
            },
        ]
    )
    grouped = group_by_style(manifest)
    assert grouped == {"style_a": [real_image]}
