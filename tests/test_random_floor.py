from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from nss.models.backtest import Origin
from nss.models.metrics import METRIC_KEYS
from nss.models.random_floor import (
    RANDOM_FLOOR_METHOD,
    aggregate_random_floor_over_seeds,
    permute_predictions,
    score_random_floor_per_origin,
)

_WEEK0 = date(2018, 1, 1)


def _weeks(n: int, start: date = _WEEK0) -> list[date]:
    return [start + timedelta(weeks=i) for i in range(n)]


# ---------------------------------------------------------------------------
# permute_predictions
# ---------------------------------------------------------------------------


def test_permute_predictions_reproducible_under_same_seed() -> None:
    y_true = [float(i) for i in range(20)]

    out1 = permute_predictions(y_true, seed=7)
    out2 = permute_predictions(y_true, seed=7)

    assert np.array_equal(out1, out2)


def test_permute_predictions_different_seeds_give_different_orderings() -> None:
    """20 distinct values -- a coincidental identical shuffle across two different seeds is
    astronomically unlikely (1 in 20! if truly independent), so inequality is a safe assertion."""
    y_true = [float(i) for i in range(20)]

    out_seed0 = permute_predictions(y_true, seed=0)
    out_seed1 = permute_predictions(y_true, seed=1)

    assert not np.array_equal(out_seed0, out_seed1)


def test_permute_predictions_is_a_true_permutation_same_multiset_of_values() -> None:
    y_true = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]

    out = permute_predictions(y_true, seed=42)

    assert sorted(out.tolist()) == sorted(y_true)


# ---------------------------------------------------------------------------
# score_random_floor_per_origin / aggregate_random_floor_over_seeds
# ---------------------------------------------------------------------------


def _predictions_frame(origin_weeks: list[date], n_styles: int = 10) -> pl.DataFrame:
    rows = []
    for ow in origin_weeks:
        for i in range(n_styles):
            rows.append(
                {
                    "style_key": f"S{i}",
                    "origin_week": ow,
                    "y_true": float(n_styles - i),  # distinct, descending: S0 highest
                    "weight": 1.0,
                }
            )
    return pl.DataFrame(rows)


def test_score_random_floor_per_origin_one_row_per_origin_and_seed() -> None:
    weeks = _weeks(2)
    origins = [Origin(origin_week=w, has_52w_lag=False, is_covid=False) for w in weeks]
    predictions = _predictions_frame(weeks)

    out = score_random_floor_per_origin(predictions, origins, seeds=(0, 1, 2))

    assert out.height == len(origins) * 3
    assert set(out["seed"].to_list()) == {0, 1, 2}
    for metric in METRIC_KEYS:
        assert metric in out.columns
    assert (out["n_eval_set"] == 10).all()


def test_score_random_floor_per_origin_uses_distinct_seed_per_row() -> None:
    """Different seeds should (almost always) produce different metric values for a non-trivial
    eval set -- sanity that the seed argument actually reaches the permutation."""
    weeks = _weeks(1)
    origins = [Origin(origin_week=weeks[0], has_52w_lag=False, is_covid=False)]
    predictions = _predictions_frame(weeks, n_styles=20)

    out = score_random_floor_per_origin(predictions, origins, seeds=(0, 1))
    wmape_values = out.sort("seed")["wmape"].to_list()

    assert wmape_values[0] != wmape_values[1]


def test_aggregate_random_floor_over_seeds_averages_correctly() -> None:
    """2 origins x 2 seeds, hand-set precision_at_3 values -> mean per origin must match."""
    weeks = _weeks(2)
    per_origin_per_seed = pl.DataFrame(
        {
            "origin_week": [weeks[0], weeks[0], weeks[1], weeks[1]],
            "seed": [0, 1, 0, 1],
            "has_52w_lag": [False, False, True, True],
            "is_covid": [False, False, True, True],
            "n_eval_set": [10, 10, 10, 10],
            "n_eval": [10, 10, 10, 10],
            **{metric: [0.2, 0.4, 0.6, 1.0] for metric in METRIC_KEYS},
        }
    )

    out = aggregate_random_floor_over_seeds(per_origin_per_seed)

    assert out.height == 2
    assert (out["method"] == RANDOM_FLOOR_METHOD).all()
    row0 = out.filter(pl.col("origin_week") == weeks[0]).row(0, named=True)
    row1 = out.filter(pl.col("origin_week") == weeks[1]).row(0, named=True)
    for metric in METRIC_KEYS:
        assert row0[metric] == pytest.approx((0.2 + 0.4) / 2)
        assert row1[metric] == pytest.approx((0.6 + 1.0) / 2)
    assert row0["has_52w_lag"] is False
    assert row1["is_covid"] is True
