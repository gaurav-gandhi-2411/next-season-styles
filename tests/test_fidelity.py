from __future__ import annotations

import polars as pl
import pytest

from nss.generate.fidelity import (
    NON_VISUAL_GRAPHICAL_VALUES,
    applicable_dimensions,
    fidelity_both,
    summarise_readings,
)


def test_exclusion_list_is_explicit_and_exact() -> None:
    assert {
        "Other structure",
        "Other pattern",
        "Unknown",
        "Treatment",
    } == NON_VISUAL_GRAPHICAL_VALUES


def test_visual_patterns_are_never_excluded() -> None:
    """Melange, Solid, Lace etc. are describable: a judge miss on them is a real error."""
    for value in ("Solid", "Melange", "Lace", "Stripe", "Mesh", "Contrast", "All over pattern"):
        assert "graphical_treatment" in applicable_dimensions(value)


def test_non_visual_value_drops_only_the_graphical_dimension() -> None:
    assert applicable_dimensions("Other structure") == ("product_type", "colour_family")


def test_fidelity_both_reports_both_figures() -> None:
    scores = {"product_type": 0.85, "colour_family": 0.85, "graphical_treatment": 0.0}
    out = fidelity_both(scores, "Other structure")
    assert out["all_attributes"] == pytest.approx(1.7 / 3)
    assert out["visual_only"] == pytest.approx(0.85)
    assert out["excluded"] == ["graphical_treatment"]
    kept = fidelity_both(scores, "Melange")  # the sweater case: melange=0 STAYS in the score
    assert kept["visual_only"] == kept["all_attributes"] == pytest.approx(1.7 / 3)
    assert kept["excluded"] == []


def test_summarise_readings_takes_median_of_each_figure() -> None:
    readings = [
        {"product_type": 0.85, "colour_family": 0.85, "graphical_treatment": 0.0},
        {"product_type": 0.1, "colour_family": 0.85, "graphical_treatment": 0.0},
        {"product_type": 0.125, "colour_family": 1.0, "graphical_treatment": 0.0},
    ]
    s = summarise_readings(readings, "Other structure")
    assert s["median_visual_only"] == pytest.approx(0.5625)
    assert s["median_all_attributes"] == pytest.approx(0.375)


@pytest.mark.needs_data("data/raw/articles.csv")
def test_every_real_pattern_value_is_classified_without_error() -> None:
    values = pl.read_csv("data/raw/articles.csv")["graphical_appearance_name"].unique().to_list()
    for v in values:
        assert applicable_dimensions(v)  # non-empty for all 30 real values
