"""Tests for `nss.generate.derive_unrelated_anchor_fix`."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nss.generate.derive_generated_space_anchors import ALL_STYLES_POOLED_KEY
from nss.generate.derive_unrelated_anchor_fix import (
    NON_DISCRIMINATIVE_GAP_THRESHOLD,
    is_discriminative,
    merge_corrected_unrelated_rows,
)

# ---------------------------------------------------------------------------
# is_discriminative -- the metric-dropping decision
# ---------------------------------------------------------------------------


def test_is_discriminative_hand_verified_threshold() -> None:
    """Default threshold 0.05: at or above passes, below fails -- hand-verifiable boundary."""
    assert is_discriminative(0.05) is True
    assert is_discriminative(0.0500001) is True
    assert is_discriminative(0.0499999) is False
    assert is_discriminative(0.0) is False


def test_is_discriminative_uses_absolute_value() -> None:
    """A large NEGATIVE gap is just as discriminative as a large positive one -- what matters is
    separation, not direction."""
    assert is_discriminative(-0.2) is True
    assert is_discriminative(-0.01) is False


def test_is_discriminative_reproduces_e1_contaminated_gaps_as_non_discriminative() -> None:
    """The original contaminated gaps (the motivating evidence) both fail this bar -- CLIP
    0.0029, DINOv2 0.0251, both far below 0.05."""
    assert is_discriminative(0.0029) is False
    assert is_discriminative(0.0251) is False


def test_is_discriminative_custom_threshold() -> None:
    assert is_discriminative(0.02, threshold=0.01) is True
    assert is_discriminative(0.02, threshold=0.03) is False


def test_non_discriminative_gap_threshold_is_order_of_magnitude_above_e1_contaminated_gaps() -> (
    None
):
    """Sanity check on the constant itself: it sits well above both of the original contaminated
    gaps, so the contamination this module fixes would have been correctly flagged
    non-discriminative under the OLD (uncorrected) anchors too -- confirming the bar isn't fit to
    this run's own numbers."""
    assert NON_DISCRIMINATIVE_GAP_THRESHOLD > 0.0251
    assert NON_DISCRIMINATIVE_GAP_THRESHOLD > 0.0029


# ---------------------------------------------------------------------------
# merge_corrected_unrelated_rows -- copy_anchor_gen rows preserved verbatim
# ---------------------------------------------------------------------------


def test_merge_corrected_unrelated_rows_keeps_copy_rows_verbatim(tmp_path: Path) -> None:
    existing_path = tmp_path / "margin_anchors_generated_space.csv"
    pl.DataFrame(
        [
            {
                "style_key": "style_a",
                "anchor_type": "copy_anchor_gen",
                "embedding_space": "clip",
                "n": 3,
                "mean": 0.1359,
                "median": 0.14,
                "std": 0.01,
            },
            {
                "style_key": "style_a",
                "anchor_type": "unrelated_anchor_gen",
                "embedding_space": "clip",
                "n": 3,
                "mean": 0.093,  # the OLD, contaminated value -- must be replaced, not kept
                "median": 0.09,
                "std": 0.01,
            },
            {
                "style_key": ALL_STYLES_POOLED_KEY,
                "anchor_type": "copy_anchor_gen",
                "embedding_space": "clip",
                "n": 9,
                "mean": 0.096,
                "median": 0.10,
                "std": 0.05,
            },
        ]
    ).write_csv(existing_path)

    corrected_summaries = [
        {
            "style_key": "style_a",
            "anchor_type": "unrelated_anchor_gen",
            "embedding_space": "clip",
            "n": 3,
            "mean": -0.02,  # the NEW, corrected value
            "median": -0.02,
            "std": 0.01,
        }
    ]

    merged = merge_corrected_unrelated_rows(
        corrected_summaries, existing_anchors_path=existing_path
    )
    by_key = {(r["style_key"], r["anchor_type"]): r for r in merged}

    # copy_anchor_gen rows are untouched, byte-identical to the existing file.
    assert by_key[("style_a", "copy_anchor_gen")]["mean"] == pytest.approx(0.1359)
    assert by_key[(ALL_STYLES_POOLED_KEY, "copy_anchor_gen")]["mean"] == pytest.approx(0.096)

    # unrelated_anchor_gen is REPLACED with the corrected value, not the old contaminated one.
    assert by_key[("style_a", "unrelated_anchor_gen")]["mean"] == pytest.approx(-0.02)
    assert by_key[("style_a", "unrelated_anchor_gen")]["mean"] != pytest.approx(0.093)


def test_merge_corrected_unrelated_rows_drops_stale_pooled_unrelated_row(tmp_path: Path) -> None:
    """If the existing file's OLD pooled unrelated_anchor_gen row isn't in the fresh corrected
    summaries, it must not survive the merge (never silently mix a stale pooled row with fresh
    per-style rows)."""
    existing_path = tmp_path / "margin_anchors_generated_space.csv"
    pl.DataFrame(
        [
            {
                "style_key": ALL_STYLES_POOLED_KEY,
                "anchor_type": "unrelated_anchor_gen",
                "embedding_space": "clip",
                "n": 9,
                "mean": 0.093,  # stale
                "median": 0.09,
                "std": 0.05,
            },
        ]
    ).write_csv(existing_path)

    corrected_summaries = [
        {
            "style_key": ALL_STYLES_POOLED_KEY,
            "anchor_type": "unrelated_anchor_gen",
            "embedding_space": "clip",
            "n": 9,
            "mean": -0.03,
            "median": -0.03,
            "std": 0.05,
        }
    ]

    merged = merge_corrected_unrelated_rows(
        corrected_summaries, existing_anchors_path=existing_path
    )
    assert len(merged) == 1
    assert merged[0]["mean"] == pytest.approx(-0.03)
