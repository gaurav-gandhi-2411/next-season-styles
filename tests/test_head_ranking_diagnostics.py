from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from nss.features.targets import HORIZON_WEEKS, compute_forward_target
from nss.models.backtest import STEP_WEEKS, Origin
from nss.models.head_ranking_diagnostics import (
    LAG_ORIGIN_STEPS,
    LAG_WEEKS,
    compression_stats,
    head_churn_fraction,
    minimal_causal_lag_origin_steps,
    near_tie_gaps,
    persistence_oracle_predictions,
    score_persistence_oracle,
)

_WEEK0 = date(2018, 1, 1)


def _weeks(n: int) -> list[date]:
    """`n` consecutive Monday week_start dates starting at `_WEEK0`."""
    return [_WEEK0 + timedelta(weeks=i) for i in range(n)]


# ---------------------------------------------------------------------------
# minimal_causal_lag_origin_steps -- lag-validity arithmetic
# ---------------------------------------------------------------------------


def test_minimal_causal_lag_origin_steps_matches_real_project_params() -> None:
    """The real project uses STEP_WEEKS=4, HORIZON_WEEKS=13 -- ceil(13/4) = 4."""
    assert minimal_causal_lag_origin_steps(step_weeks=4, horizon_weeks=13) == 4
    # Module-level constants derived from the real project params, not hardcoded.
    assert LAG_ORIGIN_STEPS == 4
    assert LAG_WEEKS == 16


def test_minimal_causal_lag_origin_steps_k3_not_sufficient_k4_sufficient_hand_verified() -> None:
    """Hand-verifiable synthetic proof, via real date arithmetic (not the ceil-division formula
    itself), that k=3 origin-steps (12 calendar weeks) leaves the earlier origin's forward window
    STILL OPEN at time t (a future leak), while k=4 (16 calendar weeks) has it fully closed.

    Origins step STEP_WEEKS=4 weeks apart; HORIZON_WEEKS=13. For an origin `k` steps earlier, its
    own forward window closes at `(t - k*step_weeks) + horizon_weeks`. Closure requires that date
    to be `<= t`.
    """
    t = date(2020, 1, 6)  # arbitrary Monday, matches this project's week_start convention

    # k=3: earlier = t - 12 weeks; its window closes at (t-12w)+13w = t+1w -- STILL AFTER t.
    earlier_k3 = t - timedelta(weeks=3 * STEP_WEEKS)
    window_close_k3 = earlier_k3 + timedelta(weeks=HORIZON_WEEKS)
    assert window_close_k3 == t + timedelta(weeks=1)
    assert window_close_k3 > t, "k=3 must NOT be causally valid -- window closes after t"

    # k=4: earlier = t - 16 weeks; its window closes at (t-16w)+13w = t-3w -- BEFORE t, closed.
    earlier_k4 = t - timedelta(weeks=4 * STEP_WEEKS)
    window_close_k4 = earlier_k4 + timedelta(weeks=HORIZON_WEEKS)
    assert window_close_k4 == t - timedelta(weeks=3)
    assert window_close_k4 <= t, "k=4 must be causally valid -- window already closed by t"

    # Confirms the module's own derivation picks exactly k=4, the minimal such k.
    assert minimal_causal_lag_origin_steps(STEP_WEEKS, HORIZON_WEEKS) == 4


@pytest.mark.parametrize(
    ("step_weeks", "horizon_weeks", "expected_k"),
    [
        (4, 13, 4),  # real project params: 13/4 = 3.25 -> ceil = 4
        (4, 12, 3),  # exact multiple: 12/4 = 3.0 -> ceil = 3 (k=3 lands exactly on closure)
        (4, 16, 4),  # exact multiple: 16/4 = 4.0 -> ceil = 4
        (1, 1, 1),
        (7, 1, 1),  # horizon smaller than one step -- still need at least 1 whole step
    ],
)
def test_minimal_causal_lag_origin_steps_ceiling_division(
    step_weeks: int, horizon_weeks: int, expected_k: int
) -> None:
    assert minimal_causal_lag_origin_steps(step_weeks, horizon_weeks) == expected_k


def test_minimal_causal_lag_origin_steps_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        minimal_causal_lag_origin_steps(0, 13)
    with pytest.raises(ValueError, match="positive"):
        minimal_causal_lag_origin_steps(4, 0)


# ---------------------------------------------------------------------------
# persistence_oracle_predictions / score_persistence_oracle
# ---------------------------------------------------------------------------


def _two_style_panel(n_weeks: int) -> pl.DataFrame:
    """Two styles, `n_weeks` consecutive weeks, distinct deterministic `units_per_active_article`
    series so each style's forward-window mean is hand-computable."""
    weeks = _weeks(n_weeks)
    return pl.DataFrame(
        {
            "style_key": (["A"] * n_weeks) + (["B"] * n_weeks),
            "week_start": weeks + weeks,
            "units_per_active_article": [float(i) for i in range(n_weeks)]
            + [float(i) * 10 for i in range(n_weeks)],
        }
    )


def test_persistence_oracle_predictions_matches_direct_compute_forward_target_call() -> None:
    """The oracle "prediction" for origin `t` is exactly `compute_forward_target`'s own realised
    target at the earlier origin `t - lag_weeks` -- same value, just relabeled onto `t`."""
    panel = _two_style_panel(n_weeks=40)
    later_origin = _WEEK0 + timedelta(weeks=30)
    lag_weeks = 16
    earlier_origin = later_origin - timedelta(weeks=lag_weeks)

    direct = compute_forward_target(panel, earlier_origin, horizon_weeks=HORIZON_WEEKS).select(
        "style_key", pl.col("target").alias("target_direct")
    )
    oracle = persistence_oracle_predictions(panel, later_origin, lag_weeks)

    assert oracle["origin_week"].unique().to_list() == [later_origin]
    joined = oracle.join(direct, on="style_key").sort("style_key")
    for row in joined.iter_rows(named=True):
        if row["target_direct"] is None:
            assert row["y_pred_oracle"] is None
        else:
            assert row["y_pred_oracle"] == pytest.approx(row["target_direct"])


def test_score_persistence_oracle_perfect_prediction_scores_perfectly() -> None:
    """If the oracle's earlier-origin target happens to exactly equal the later origin's true
    target (a style with perfectly flat, repeating demand), Precision@3/Hit@3 should score 1.0."""
    n_weeks = 60
    weeks = _weeks(n_weeks)
    # 5 styles, each with a CONSTANT units_per_active_article over time -- so any origin's forward
    # target equals any other origin's forward target for the same style (perfect persistence).
    levels = {"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0, "E": 1.0}
    rows = []
    for style, level in levels.items():
        for w in weeks:
            rows.append({"style_key": style, "week_start": w, "units_per_active_article": level})
    panel = pl.DataFrame(rows)

    origin = Origin(origin_week=_WEEK0 + timedelta(weeks=40), has_52w_lag=False, is_covid=False)
    eval_frame = pl.DataFrame(
        {
            "style_key": list(levels.keys()),
            "origin_week": [origin.origin_week] * len(levels),
            "y_true": [
                compute_forward_target(panel, origin.origin_week).filter(pl.col("style_key") == s)[
                    "target"
                ][0]
                for s in levels
            ],
            "weight": [1.0] * len(levels),
        }
    )

    result = score_persistence_oracle(panel, [origin], eval_frame, lag_weeks=16)

    assert result.height == 1
    row = result.row(0, named=True)
    assert row["precision_at_3"] == pytest.approx(1.0)
    assert row["hit_at_3_in_top10"] == pytest.approx(1.0)
    assert row["hit_at_3_in_top20"] == pytest.approx(1.0)
    assert row["n_eval"] == 5


# ---------------------------------------------------------------------------
# head_churn_fraction
# ---------------------------------------------------------------------------


def _eval_frame_at(origin_week: date, style_values: dict[str, float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "style_key": list(style_values.keys()),
            "origin_week": [origin_week] * len(style_values),
            "y_true": list(style_values.values()),
        }
    )


def test_head_churn_fraction_hand_computed_overlap() -> None:
    """Origin t's true top-3 = {A, B, C} (values 10, 9, 8). Earlier origin's true top-20 (only 4
    styles exist, so top-20 clips to all 4) = {B, C, D, X} (values 5, 4, 3, 1). Overlap = {B, C} ->
    fraction = 2/3."""
    t = _WEEK0 + timedelta(weeks=40)
    earlier = t - timedelta(weeks=16)
    origin = Origin(origin_week=t, has_52w_lag=False, is_covid=False)

    current = _eval_frame_at(t, {"A": 10.0, "B": 9.0, "C": 8.0, "D": 1.0})
    earlier_frame = _eval_frame_at(earlier, {"B": 5.0, "C": 4.0, "D": 3.0, "X": 1.0})
    eval_frame = pl.concat([current, earlier_frame])

    result = head_churn_fraction(eval_frame, [origin], [earlier], top_k=3, top_n=20)

    assert result.height == 1
    row = result.row(0, named=True)
    assert row["n_true_top_k"] == 3
    assert row["n_overlap"] == 2
    assert row["churn_fraction"] == pytest.approx(2 / 3)


def test_head_churn_fraction_clips_k_and_n_to_eval_set_size() -> None:
    """Only 2 styles in the current eval set (< top_k=3) -- k_eff clips to 2, not a crash."""
    t = _WEEK0 + timedelta(weeks=40)
    earlier = t - timedelta(weeks=16)
    origin = Origin(origin_week=t, has_52w_lag=False, is_covid=False)

    current = _eval_frame_at(t, {"A": 10.0, "B": 9.0})
    earlier_frame = _eval_frame_at(earlier, {"A": 5.0})
    eval_frame = pl.concat([current, earlier_frame])

    result = head_churn_fraction(eval_frame, [origin], [earlier], top_k=3, top_n=20)

    row = result.row(0, named=True)
    assert row["n_true_top_k"] == 2  # clipped from 3
    assert row["n_overlap"] == 1  # only A present in both
    assert row["churn_fraction"] == pytest.approx(0.5)


def test_head_churn_fraction_rejects_mismatched_lengths() -> None:
    t = _WEEK0 + timedelta(weeks=40)
    origin = Origin(origin_week=t, has_52w_lag=False, is_covid=False)
    empty = pl.DataFrame({"style_key": [], "origin_week": [], "y_true": []})
    with pytest.raises(ValueError, match="same length"):
        head_churn_fraction(empty, [origin], [])


# ---------------------------------------------------------------------------
# near_tie_gaps
# ---------------------------------------------------------------------------


def test_near_tie_gaps_hand_computed() -> None:
    """Values sorted descending: 100 (1st), 50 (2nd), 40 (#3), 39 (#4), 20 (#5), 10 (#6).
    gap_3_vs_4 = (40-39)/40*100 = 2.5%. gap_3_vs_5 = (40-20)/40*100 = 50%. #10 doesn't exist (only
    6 styles) -> None."""
    t = _WEEK0
    origin = Origin(origin_week=t, has_52w_lag=False, is_covid=False)
    eval_frame = _eval_frame_at(
        t, {"a": 100.0, "b": 50.0, "c": 40.0, "d": 39.0, "e": 20.0, "f": 10.0}
    )

    result = near_tie_gaps(eval_frame, [origin])

    row = result.row(0, named=True)
    assert row["value_3"] == pytest.approx(40.0)
    assert row["gap_3_vs_4_pct"] == pytest.approx(2.5)
    assert row["gap_3_vs_5_pct"] == pytest.approx(50.0)
    assert row["gap_3_vs_10_pct"] is None


def test_near_tie_gaps_zero_value_3_is_none_not_zero_division() -> None:
    t = _WEEK0
    origin = Origin(origin_week=t, has_52w_lag=False, is_covid=False)
    eval_frame = _eval_frame_at(t, {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0})

    result = near_tie_gaps(eval_frame, [origin])

    row = result.row(0, named=True)
    assert row["value_3"] == 0.0
    assert row["gap_3_vs_4_pct"] is None


def test_near_tie_gaps_fewer_than_3_styles_is_none() -> None:
    t = _WEEK0
    origin = Origin(origin_week=t, has_52w_lag=False, is_covid=False)
    eval_frame = _eval_frame_at(t, {"a": 5.0, "b": 4.0})

    result = near_tie_gaps(eval_frame, [origin])

    row = result.row(0, named=True)
    assert row["value_3"] is None
    assert row["gap_3_vs_4_pct"] is None


# ---------------------------------------------------------------------------
# compression_stats
# ---------------------------------------------------------------------------


def test_compression_stats_hand_computed_ratios() -> None:
    y_true = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([10.0, 12.0, 15.0, 18.0, 20.0])  # deliberately narrower spread

    result = compression_stats(y_true, y_pred)

    assert result["std_true"] == pytest.approx(float(np.std(y_true)))
    assert result["std_pred"] == pytest.approx(float(np.std(y_pred)))
    assert result["std_ratio_pred_over_true"] == pytest.approx(
        float(np.std(y_pred)) / float(np.std(y_true))
    )
    assert result["std_ratio_pred_over_true"] < 1.0  # narrower predicted spread, confirms shrink


def test_compression_stats_identical_distributions_ratio_is_one() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    result = compression_stats(y, y.copy())
    assert result["std_ratio_pred_over_true"] == pytest.approx(1.0)
    assert result["spread_ratio_pred_over_true"] == pytest.approx(1.0)


def test_compression_stats_zero_true_spread_is_nan_not_crash() -> None:
    y_true = np.array([5.0, 5.0, 5.0])  # zero std, zero p99-median spread
    y_pred = np.array([1.0, 2.0, 3.0])
    result = compression_stats(y_true, y_pred)
    assert math.isnan(result["std_ratio_pred_over_true"])
    assert math.isnan(result["spread_ratio_pred_over_true"])
