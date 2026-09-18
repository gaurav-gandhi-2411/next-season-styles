from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nss.generate.derive_margin_band import (
    derive_target_band,
    embed_all,
    load_control_pool,
    load_manifest,
    load_non_control_styles,
    lower_anchor_margins,
    summarize,
    upper_anchor_margins,
)

# 3D axis-aligned vectors so every cosine similarity is hand-computable exactly (0 or 1) --
# same fixture-design convention as test_derive_similarity_band.py.
_STYLE_A_IMG1 = Path("a1.jpg")
_STYLE_A_IMG2 = Path("a2.jpg")
_STYLE_B_IMG1 = Path("b1.jpg")
_STYLE_B_IMG2 = Path("b2.jpg")
_CONTROL_IMG = Path("c1.jpg")


def _style_to_images() -> dict[str, list[Path]]:
    return {"style_a": [_STYLE_A_IMG1, _STYLE_A_IMG2], "style_b": [_STYLE_B_IMG1, _STYLE_B_IMG2]}


def _embeddings() -> dict[Path, np.ndarray]:
    return {
        _STYLE_A_IMG1: np.array([1.0, 0.0, 0.0]),
        _STYLE_A_IMG2: np.array([1.0, 0.0, 0.0]),  # identical to A1 -> leave-one-out cos = 1.0
        _STYLE_B_IMG1: np.array([0.0, 1.0, 0.0]),
        _STYLE_B_IMG2: np.array([0.0, 1.0, 0.0]),  # identical to B1
        _CONTROL_IMG: np.array([0.0, 0.0, 1.0]),  # orthogonal to both A and B
    }


def test_upper_anchor_margins_leave_one_out_hand_computed() -> None:
    """Each image's leave-one-out own-style cos = 1.0 (identical sibling), control cos = 0.0
    (orthogonal) -> margin = 1.0 for every one of the 4 images (2 styles x 2 images)."""
    embeddings = _embeddings()
    control_embeddings = [embeddings[_CONTROL_IMG]]

    margins = upper_anchor_margins(_style_to_images(), embeddings, control_embeddings)

    assert len(margins) == 4
    assert all(m == pytest.approx(1.0) for m in margins)


def test_upper_anchor_margins_skips_singleton_styles() -> None:
    """A style with only 1 image has no leave-one-out reference and contributes no margin."""
    embeddings = _embeddings()
    control_embeddings = [embeddings[_CONTROL_IMG]]
    style_to_images = {"solo": [_STYLE_A_IMG1]}

    assert upper_anchor_margins(style_to_images, embeddings, control_embeddings) == []


def test_lower_anchor_margins_cross_style_is_zero_in_orthogonal_fixture() -> None:
    """style_a vs style_b (orthogonal, cos=0) minus control (also orthogonal, cos=0) -> margin 0,
    for whichever of the two styles the seeded draw picks as style_a."""
    embeddings = _embeddings()
    control_embeddings = [embeddings[_CONTROL_IMG]]

    margins, style_a, style_b = lower_anchor_margins(
        _style_to_images(), embeddings, control_embeddings, seed=42
    )

    assert {style_a, style_b} == {"style_a", "style_b"}
    assert style_a != style_b
    assert len(margins) == 2  # whichever style was drawn as style_a has 2 images
    assert all(m == pytest.approx(0.0) for m in margins)


def test_lower_anchor_margins_raises_with_fewer_than_two_styles() -> None:
    embeddings = _embeddings()
    control_embeddings = [embeddings[_CONTROL_IMG]]
    with pytest.raises(ValueError, match="at least 2"):
        lower_anchor_margins({"style_a": [_STYLE_A_IMG1]}, embeddings, control_embeddings)


def test_summarize_hand_computed() -> None:
    result = summarize([0.0, 0.5, 1.0])
    assert result == {
        "n": 3,
        "mean": pytest.approx(0.5),
        "median": pytest.approx(0.5),
        "std": pytest.approx(0.5),
    }


def test_summarize_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        summarize([])


def test_derive_target_band_is_a_fraction_of_upper_anchor_mean() -> None:
    """band = [lower_fraction, upper_fraction] * upper_anchor_mean -- read straight off the
    formula with round, hand-checkable numbers."""
    upper_summary = {"n": 4, "mean": 0.8, "median": 0.8, "std": 0.0}

    band = derive_target_band(upper_summary, lower_fraction=0.25, upper_fraction=0.75)

    assert band["lower"] == pytest.approx(0.2)
    assert band["upper"] == pytest.approx(0.6)
    assert band["lower_fraction"] == 0.25
    assert band["upper_fraction"] == 0.75


def test_embed_all_embeds_each_distinct_path_once() -> None:
    calls: list[Path] = []

    def _fake_embed(path: Path) -> np.ndarray:
        calls.append(path)
        return np.array([1.0, 0.0])

    result = embed_all([_STYLE_A_IMG1, _STYLE_A_IMG1, _STYLE_B_IMG1], _fake_embed)

    assert calls == [_STYLE_A_IMG1, _STYLE_B_IMG1]  # A1 embedded once despite 2 occurrences
    assert set(result) == {_STYLE_A_IMG1, _STYLE_B_IMG1}


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    pl.DataFrame(rows).write_csv(path)


def test_load_manifest_dedupes_and_drops_fetch_failures(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(
        manifest_path,
        [
            {
                "role": "control",
                "style_key": "style_a",
                "article_id": 1,
                "local_image_path": "img1.jpg",
                "fetch_success": True,
            },
            {
                "role": "control",
                "style_key": "style_a",
                "article_id": 1,  # duplicate article_id -- must be deduped
                "local_image_path": "img1.jpg",
                "fetch_success": True,
            },
            {
                "role": "control",
                "style_key": "style_a",
                "article_id": 2,
                "local_image_path": "img2.jpg",
                "fetch_success": False,  # never fetched -- must be dropped
            },
        ],
    )

    result = load_manifest(manifest_path)

    assert result["article_id"].to_list() == [1]


def test_load_control_pool_only_keeps_control_role(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    control_img = tmp_path / "control.jpg"
    control_img.write_bytes(b"fake")
    winner_img = tmp_path / "winner.jpg"
    winner_img.write_bytes(b"fake")
    _write_manifest(
        manifest_path,
        [
            {
                "role": "control",
                "style_key": "control_style",
                "article_id": 1,
                "local_image_path": str(control_img),
                "fetch_success": True,
            },
            {
                "role": "winner_rank_1",
                "style_key": "winner_style",
                "article_id": 2,
                "local_image_path": str(winner_img),
                "fetch_success": True,
            },
        ],
    )

    result = load_control_pool(manifest_path)

    assert result == [control_img]


def test_load_non_control_styles_combines_final_three_and_margin_reference(
    tmp_path: Path,
) -> None:
    final_three_path = tmp_path / "final_three.csv"
    margin_ref_path = tmp_path / "margin_ref.csv"
    img1 = tmp_path / "img1.jpg"
    img1.write_bytes(b"fake")
    img2 = tmp_path / "img2.jpg"
    img2.write_bytes(b"fake")

    _write_manifest(
        final_three_path,
        [
            {
                "role": "final_rank_1",
                "style_key": "style_x",
                "article_id": 1,
                "local_image_path": str(img1),
                "fetch_success": True,
            },
            {
                "role": "control",  # must be excluded -- only final_rank_* rows are kept
                "style_key": "control_style",
                "article_id": 99,
                "local_image_path": str(img1),
                "fetch_success": True,
            },
        ],
    )
    _write_manifest(
        margin_ref_path,
        [
            {
                "role": "margin_ref_01",
                "style_key": "style_y",
                "article_id": 2,
                "local_image_path": str(img2),
                "fetch_success": True,
            },
        ],
    )

    result = load_non_control_styles(final_three_path, margin_ref_path)

    assert result == {"style_x": [img1], "style_y": [img2]}
