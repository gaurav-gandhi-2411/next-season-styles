from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from nss.models.phase_b import guardrail_failures, parent_panel, reconcile


def _styles(y: list[float], w: list[float], ig: list[str], pt: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "style_key": [f"s{i}" for i in range(len(y))],
            "index_group_name": ig,
            "product_type_name": pt,
            "weight": w,
            "y_pred": np.log1p(y),
        }
    )


def test_reconcile_leaves_coherent_forecasts_unchanged() -> None:
    """Parents equal to the weighted mean of their children: OLS projection is the identity."""
    s = _styles([2.0, 4.0, 6.0], [1.0, 1.0, 2.0], ["A", "A", "A"], ["x", "x", "y"])
    pt = pl.DataFrame(
        {
            "index_group_name": ["A", "A"],
            "product_type_name": ["x", "y"],
            "y_pred": np.log1p([3.0, 6.0]),
        }
    )
    ig = pl.DataFrame({"index_group_name": ["A"], "y_pred": np.log1p([(2 + 4 + 12) / 4])})
    out = reconcile(s, pt, ig)
    assert np.expm1(out["y_pred"].to_numpy()) == pytest.approx([2.0, 4.0, 6.0])


def test_reconcile_moves_children_toward_parent() -> None:
    """Two equal-weight children forecast 2 and 4; parent forecasts 6 -> both children rise.

    OLS: minimise (b1-2)^2 + (b2-4)^2 + ((b1+b2)/2 - 6)^2 -> b1 = 3, b2 = 5 (hand-solved).
    """
    s = _styles([2.0, 4.0], [1.0, 1.0], ["A", "A"], ["x", "x"])
    pt = pl.DataFrame(
        {"index_group_name": ["A"], "product_type_name": ["x"], "y_pred": [np.log1p(6.0)]}
    )
    ig = pl.DataFrame(
        {"index_group_name": [], "y_pred": []},
        schema={"index_group_name": pl.String, "y_pred": pl.Float64},
    )
    out = reconcile(s, pt, ig)
    assert np.expm1(out["y_pred"].to_numpy()) == pytest.approx([3.0, 5.0])


def test_reconcile_skips_parent_with_zero_weight() -> None:
    s = _styles([2.0, 4.0], [0.0, 0.0], ["A", "A"], ["x", "x"])
    pt = pl.DataFrame(
        {"index_group_name": ["A"], "product_type_name": ["x"], "y_pred": [np.log1p(100.0)]}
    )
    ig = pl.DataFrame(
        {"index_group_name": [], "y_pred": []},
        schema={"index_group_name": pl.String, "y_pred": pl.Float64},
    )
    assert np.expm1(reconcile(s, pt, ig)["y_pred"].to_numpy()) == pytest.approx([2.0, 4.0])


def test_parent_panel_aggregates_and_densifies() -> None:
    wk = [date(2020, 1, 6), date(2020, 1, 13), date(2020, 1, 20)]
    panel = pl.DataFrame(
        {
            "style_key": ["a", "b", "a"],
            "index_group_name": ["G", "G", "G"],
            "product_type_name": ["T", "T", "T"],
            "garment_group_name": ["g", "g", "g"],
            "perceived_colour_master_name": ["c", "c", "c"],
            "graphical_appearance_name": ["s", "s", "s"],
            "week_start": [wk[0], wk[0], wk[2]],  # nothing in week 2: must be densified
            "units": [10, 30, 5],
            "revenue": [1.0, 3.0, 0.5],
            "n_active_articles": [1, 3, 5],
            "price_index": [1.0, 2.0, None],
            "first_week_seen": [wk[0], wk[0], wk[0]],
        }
    )
    pp = parent_panel(panel, ["index_group_name"], "IG")
    assert pp["week_start"].to_list() == wk
    assert pp["units"].to_list() == [40, 0, 5]
    assert pp["units_per_active_article"].to_list() == pytest.approx([10.0, 0.0, 1.0])
    assert pp["price_index"][0] == pytest.approx((1.0 * 1 + 2.0 * 3) / 4)
    assert pp["product_type_name"].unique().to_list() == ["ALL"]


def test_guardrail_failure_direction() -> None:
    pairs = pl.DataFrame(
        {
            "challenger": ["c"] * 4,
            "metric": ["hit_at_3_in_top20", "ndcg_at_10", "spearman_rho", "wmape"],
            "ci_lo": [-0.2, -0.05, 0.01, 0.001],
            "ci_hi": [0.1, -0.01, 0.05, 0.02],
        }
    )
    # NDCG wholly below zero fails; WMAPE wholly above zero fails (lower WMAPE is better)
    assert guardrail_failures(pairs, "c") == ["ndcg_at_10", "wmape"]
