from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nss.generate.scale_sweep import (
    STYLE_KEY,
    aggregate_by_scale,
    build_prompt,
    is_monotonic_increasing,
    load_margin_band,
    select_operating_scale,
    style_description,
)


def test_style_description_splits_final_rank_1_style_key() -> None:
    """The chosen sweep style_key parses into its 5 attribute parts in the expected order."""
    assert style_description(STYLE_KEY) == "black solid jersey basic t-shirt"


def test_style_description_rejects_malformed_style_key() -> None:
    """A style_key without exactly 5 ' || '-separated parts fails loudly, not silently."""
    with pytest.raises(ValueError, match="Expected 5"):
        style_description("Ladieswear || T-shirt")


def test_build_prompt_includes_department_and_description() -> None:
    """The generation prompt is deterministically derived from the style_key's attributes."""
    prompt = build_prompt(STYLE_KEY)
    assert prompt == (
        "a new black solid jersey basic t-shirt fashion concept for ladieswear, "
        "product photography, plain background"
    )


def test_is_monotonic_increasing_true_for_nondecreasing_sequence() -> None:
    """A strictly increasing sequence counts as (non-strictly) monotonic."""
    assert is_monotonic_increasing([0.1, 0.2, 0.2, 0.5])


def test_is_monotonic_increasing_false_when_a_later_value_drops() -> None:
    """A single decrease anywhere in the sequence breaks monotonicity."""
    assert not is_monotonic_increasing([0.1, 0.5, 0.3, 0.6])


def test_is_monotonic_increasing_true_for_single_value() -> None:
    """A single-element (or empty) sequence is trivially monotonic."""
    assert is_monotonic_increasing([0.42])
    assert is_monotonic_increasing([])


def test_load_margin_band_reads_band_lower_upper(tmp_path: Path) -> None:
    """`load_margin_band` reads (band_lower, band_upper) off either row of the anchors CSV."""
    path = tmp_path / "margin_anchors_clip.csv"
    pl.DataFrame(
        [
            {"row_type": "upper_anchor", "band_lower": 0.0325, "band_upper": 0.0975},
            {"row_type": "lower_anchor", "band_lower": 0.0325, "band_upper": 0.0975},
        ]
    ).write_csv(path)

    lower, upper = load_margin_band(path)

    assert lower == pytest.approx(0.0325)
    assert upper == pytest.approx(0.0975)


def _raw_fixture() -> pl.DataFrame:
    """3 seeds x 2 scales of hand-picked margin values, round enough to hand-check aggregation."""
    return pl.DataFrame(
        [
            {"ip_adapter_scale": 0.2, "seed": 42, "clip_margin": 0.02, "dino_margin": 0.10},
            {"ip_adapter_scale": 0.2, "seed": 43, "clip_margin": 0.04, "dino_margin": 0.12},
            {"ip_adapter_scale": 0.2, "seed": 44, "clip_margin": 0.03, "dino_margin": 0.11},
            {"ip_adapter_scale": 0.5, "seed": 42, "clip_margin": 0.05, "dino_margin": 0.20},
            {"ip_adapter_scale": 0.5, "seed": 43, "clip_margin": 0.06, "dino_margin": 0.22},
            {"ip_adapter_scale": 0.5, "seed": 44, "clip_margin": 0.07, "dino_margin": 0.21},
        ]
    )


def test_aggregate_by_scale_computes_mean_and_sample_std() -> None:
    """Per-scale mean/std (ddof=1) match numpy's hand-computed values for the fixture."""
    raw = _raw_fixture()

    result = aggregate_by_scale(raw, clip_band=(0.0, 1.0), dino_band=(0.0, 1.0))

    row_02 = result.filter(pl.col("ip_adapter_scale") == 0.2).row(0, named=True)
    assert row_02["clip_margin_mean"] == pytest.approx(0.03)
    assert row_02["clip_margin_std"] == pytest.approx(0.01)  # std([.02,.04,.03], ddof=1)
    assert row_02["n_seeds"] == 3


def test_aggregate_by_scale_flags_in_band_from_mean_only() -> None:
    """`clip_in_band`/`dino_in_band` reflect whether the per-scale MEAN falls in the given band."""
    raw = _raw_fixture()

    result = aggregate_by_scale(raw, clip_band=(0.025, 0.045), dino_band=(0.0, 0.15))

    row_02 = result.filter(pl.col("ip_adapter_scale") == 0.2).row(0, named=True)
    row_05 = result.filter(pl.col("ip_adapter_scale") == 0.5).row(0, named=True)
    assert row_02["clip_in_band"] is True  # mean 0.03 in [0.025, 0.045]
    assert row_05["clip_in_band"] is False  # mean 0.06 not in [0.025, 0.045]
    assert row_02["dino_in_band"] is True  # mean 0.11 in [0.0, 0.15]
    assert row_05["dino_in_band"] is False  # mean 0.21 not in [0.0, 0.15]


def test_select_operating_scale_recommends_lowest_scale_in_band_for_both() -> None:
    aggregated = pl.DataFrame(
        [
            {"ip_adapter_scale": 0.2, "clip_in_band": False, "dino_in_band": False},
            {"ip_adapter_scale": 0.3, "clip_in_band": True, "dino_in_band": True},
            {"ip_adapter_scale": 0.4, "clip_in_band": True, "dino_in_band": True},
        ]
    )

    result = select_operating_scale(aggregated)

    assert result["clip_in_band_scales"] == [0.3, 0.4]
    assert result["dino_in_band_scales"] == [0.3, 0.4]
    assert result["both_in_band_scales"] == [0.3, 0.4]
    assert result["recommended_scale"] == pytest.approx(0.3)


def test_select_operating_scale_reports_disagreement_without_forcing_a_number() -> None:
    """When CLIP and DINOv2 never agree on a scale, `recommended_scale` is None -- not guessed."""
    aggregated = pl.DataFrame(
        [
            {"ip_adapter_scale": 0.2, "clip_in_band": True, "dino_in_band": False},
            {"ip_adapter_scale": 0.6, "clip_in_band": False, "dino_in_band": True},
        ]
    )

    result = select_operating_scale(aggregated)

    assert result["clip_in_band_scales"] == [0.2]
    assert result["dino_in_band_scales"] == [0.6]
    assert result["both_in_band_scales"] == []
    assert result["recommended_scale"] is None
