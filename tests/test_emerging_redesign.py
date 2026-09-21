from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from nss.models import emerging_redesign as er
from nss.models import growth_backtest as gb

T = date(2020, 1, 6)


def _pop(n: int = 30) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    trailing = rng.uniform(1.0, 20.0, n)
    base = {
        "style_key": [f"s{i}" for i in range(n)],
        "origin_week": [T] * n,
        "y_true": np.log1p(trailing * rng.uniform(0.5, 2.0, n)).tolist(),
        "trailing_13w_mean_intensity": trailing.tolist(),
    }
    for m in gb.METHODS:
        base[f"y_pred_{m}"] = np.log1p(trailing * rng.uniform(0.6, 1.8, n)).tolist()
    return pl.DataFrame(base)


def test_excess_is_model_minus_seasonal_naive_and_degenerate_for_seasonal_naive() -> None:
    pop = _pop()
    v = er.score_vector(pop, "excess", "lightgbm")
    assert np.allclose(
        v, pop["y_pred_lightgbm"].to_numpy() - pop["y_pred_seasonal_naive"].to_numpy()
    )
    assert er.score_vector(pop, "excess", "seasonal_naive") is None


def test_excess_has_no_trailing_mean_denominator() -> None:
    pop = _pop()
    scaled = pop.with_columns(
        (pl.col("trailing_13w_mean_intensity") * 7.0).alias("trailing_13w_mean_intensity")
    )
    assert np.allclose(
        er.score_vector(pop, "excess", "lightgbm"), er.score_vector(scaled, "excess", "lightgbm")
    )
    assert not np.allclose(
        er.score_vector(pop, "ratio", "lightgbm"), er.score_vector(scaled, "ratio", "lightgbm")
    )


def test_residualised_ratio_is_uncorrelated_with_the_trailing_mean() -> None:
    pop = _pop()
    r = er.score_vector(pop, "residual", "lightgbm")
    assert abs(np.corrcoef(r, pop["trailing_13w_mean_intensity"].to_numpy())[0, 1]) < 1e-9


def test_p3_requires_a_seasonal_naive_forecast() -> None:
    pop = _pop().with_columns(
        pl.lit(True).alias("guard1_pass"),
        pl.lit(True).alias("guard2_pass"),
        pl.lit(True).alias("guard3_pass"),
    )
    with_null = pop.with_columns(
        pl.when(pl.col("style_key") == "s0")
        .then(None)
        .otherwise(pl.col("y_pred_seasonal_naive"))
        .alias("y_pred_seasonal_naive")
    )
    assert "s0" not in er.p3(with_null)["style_key"].to_list()
    assert er.p3(with_null).height < er.p3(pop).height + 1


def _table(
    hit_model: float, hit_sn: float, hit_gm: float, hit_floor: float, n: int = 12
) -> pl.DataFrame:
    rows = []
    noise = np.random.default_rng(3).normal(0, 0.08, n)
    noise = noise - noise.mean()
    for i in range(n):
        wk = T + timedelta(weeks=i)
        for c, m, h in (
            ("excess", "lightgbm", hit_model),
            ("ratio", "seasonal_naive", hit_sn),
            ("excess", "global_mean", hit_gm),
            ("random_floor", "random_floor", hit_floor),
        ):
            h = h + noise[i] if m == "global_mean" else h
            rows.append({"origin_week": wk, "construction": c, "method": m, "hit_at_3_in_top20": h})
    return pl.DataFrame(rows).unique(["origin_week", "construction", "method"])


def test_adoption_needs_both_primary_and_diagnostic() -> None:
    good = _table(0.9, 0.5, 0.18, 0.18, n=40)
    assert bool(er.decide(good, good)["ADOPT_REDESIGNED_SCORE"][0])
    base_effect_left = _table(0.9, 0.5, 0.75, 0.18, n=40)
    d = er.decide(base_effect_left, base_effect_left)
    assert bool(d["primary_passes_on_both"][0]) and not bool(d["diagnostic_passes_on_both"][0])
    assert not bool(d["ADOPT_REDESIGNED_SCORE"][0])
    no_edge = _table(0.5, 0.5, 0.2, 0.18, n=40)
    assert not bool(er.decide(no_edge, no_edge)["ADOPT_REDESIGNED_SCORE"][0])
