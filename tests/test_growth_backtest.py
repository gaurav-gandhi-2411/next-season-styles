from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from nss.models import growth_backtest as gb

T = date(2020, 1, 6)


def _frame(n: int = 10) -> pl.DataFrame:
    """One origin; the model's predicted intensity rises with the style index."""
    idx = np.arange(n)
    return pl.DataFrame(
        {
            "style_key": [f"s{i}" for i in idx],
            "origin_week": [T] * n,
            "y_true": np.log1p(2.0 + idx).tolist(),
            "n_active_articles_level": [10.0] * n,
            "trailing_13w_mean_intensity": [2.0] * n,
            "guard1_pass": [True] * n,
            "guard2_pass": [True] * n,
            "guard3_pass": [i != 0 for i in idx],  # style 0 fails a guard
            **{f"y_pred_{m}": np.log1p(1.0 + idx).tolist() for m in gb.METHODS},
        }
    )


def test_p1_drops_guard_failures_and_nonpositive_trailing() -> None:
    f = _frame().with_columns(
        pl.when(pl.col("style_key") == "s1")
        .then(0.0)
        .otherwise(2.0)
        .alias("trailing_13w_mean_intensity")
    )
    assert sorted(gb.population(f, "P1")["style_key"].to_list()) == [f"s{i}" for i in range(2, 10)]


def test_p2_keeps_only_styles_at_or_above_the_models_median_floor() -> None:
    f = _frame()
    p2 = gb.population(f, "P2")
    guarded = f.filter(pl.col("guard3_pass"))
    floor = float(np.median(np.expm1(guarded["y_pred_lightgbm"].to_numpy())))
    assert (np.expm1(p2["y_pred_lightgbm"].to_numpy()) >= floor).all()
    assert 0 < p2.height < guarded.height


def test_growth_is_the_deployed_ratio_on_both_sides() -> None:
    g = gb.growth_columns(_frame(4))
    assert g["g_true"].to_list() == pytest.approx([(2.0 + i) / 2.0 for i in range(4)])
    assert g["g_pred_lightgbm"].to_list() == pytest.approx([(1.0 + i) / 2.0 for i in range(4)])


def test_population_choice_is_validated() -> None:
    with pytest.raises(ValueError):
        gb.population(_frame(), "P3")


def _paired(hit_lo: float, ndcg_lo: float, rho_lo: float, hit_mean: float = 0.1) -> pl.DataFrame:
    rows = []
    for comp in (*gb.BASELINE_METHODS, "random_floor"):
        for metric, lo, mean in (
            ("hit_at_3_in_top20", hit_lo, hit_mean),
            ("ndcg_at_10", ndcg_lo, 0.1),
            ("spearman_rho", rho_lo, 0.1),
        ):
            rows.append(
                {
                    "method_b": comp,
                    "split": "pooled",
                    "metric": metric,
                    "mean_diff": mean,
                    "diff_ci_low": lo,
                    "diff_ci_high": lo + 0.2,
                    "n_origins": 12,
                }
            )
    return pl.DataFrame(rows)


def test_categories_follow_the_preregistered_rule() -> None:
    assert gb.categorise(_paired(0.05, -1, -1))[0] == "VALIDATED"  # rule A vs all
    assert gb.categorise(_paired(-0.05, 0.02, 0.02))[0] == "VALIDATED"  # rule B vs all
    assert gb.categorise(_paired(-0.05, 0.02, 0.02, hit_mean=-0.1))[0] == "NOT_VALIDATED"
    assert gb.categorise(_paired(-0.05, -0.02, 0.02))[0] == "NOT_VALIDATED"


def test_partial_when_the_floor_is_beaten_but_a_baseline_is_not() -> None:
    p = _paired(0.05, -1, -1).with_columns(
        pl.when(
            (pl.col("method_b") == "seasonal_naive") & (pl.col("metric") == "hit_at_3_in_top20")
        )
        .then(-0.05)
        .otherwise(pl.col("diff_ci_low"))
        .alias("diff_ci_low")
    )
    category, table = gb.categorise(p)
    assert category == "PARTIAL"
    assert table.filter(pl.col("comparator") == "seasonal_naive")["beats"][0] is False
