"""Task G1: four read-only diagnostics probing whether the Precision@3 head-ranking result is a
genuine noise floor or a fixable objective-mismatch defect. NO TRAINING happens in this module
except the one sanctioned exception in (b) -- see COMPRESSION (b) below -- which reproduces the
FROZEN, already-established LightGBM model's own predictions (never a new model/config).

MOTIVATION (see PLAN.md / task A1): the TRUE top-3 overlaps 0.421 between consecutive backtest
origins (4 weeks apart), suggesting the head IS reachable from recent history. Yet the current
LightGBM L2-regression model's pooled Precision@3 is ~0.056
(`reports/tables/backtest_summary_v2.csv`). This module runs 4 diagnostics to tell defect from
noise floor:

(a) CAUSAL PERSISTENCE ORACLE (`score_persistence_oracle`): "predict" each backtest origin `t`
using the fully-realised (already-closed) forward target of a causally-valid EARLIER origin
`t - LAG_WEEKS`. `LAG_WEEKS` is derived, not assumed -- see `minimal_causal_lag_origin_steps` for
the ceiling-division arithmetic proving `k=4` origin-steps (16 calendar weeks) is the minimal lag
whose own 13-week forward window is guaranteed closed by time `t`, given `STEP_WEEKS=4` origins.
Scored with the exact same `nss.models.metrics.score_predictions` used everywhere else in this
project, over the SAME 12 origins LightGBM was walk-forward-evaluated on
(`nss.models.backtest_v2.identify_lightgbm_origins`). If this oracle's Precision@3 is far above
LightGBM's, LightGBM has a fixable defect (a real signal it isn't reaching); if it's close, ~0.056
may be closer to this eval population's genuine noise floor even with perfect causal information.

(b) PREDICTION COMPRESSION (`compression_stats`): compares the pooled distribution spread of the
FROZEN LightGBM model's predictions vs. the realised actuals, testing the hypothesis that L2 loss
on a log1p target shrinks extreme predictions toward the mean (so the model never confidently
surfaces true extremes even if it "knows" the right direction). Raw per-style predictions were
never checked in (only aggregate per-origin metrics in `backtest_per_origin_lightgbm.csv`), so
`lightgbm_walk_forward_raw_predictions` re-runs the EXACT existing walk-forward inference loop
(`nss.models.lightgbm_model.build_model_frame` / `train_lightgbm` / `predict_lightgbm`, unchanged)
with the FROZEN winning hyperparameters (`nss.models.final_forecast.FINAL_MODEL_CONFIG`) to
recover them -- per this project's own verified cross-process determinism guarantee (D3a, see
`nss.models.lightgbm_model` module docstring), this reproduces BIT-IDENTICAL predictions to the
ones already checked in, not a new model. `main()` cross-checks this directly against
`backtest_per_origin_lightgbm.csv`'s own `precision_at_3` column before trusting the reproduced
predictions for anything.

(c) HEAD CHURN (`head_churn_fraction`): using the SAME causally-valid `LAG_WEEKS` lag as (a), what
fraction of an origin's TRUE top-3 styles were ALSO in the causally-earlier origin's TRUE top-20?
High overlap means the head is reachable from causally-available history via lag-type features and
a ranking objective should be able to find it; low overlap means additional feature information is
needed.

(d) NEAR-TIE ANALYSIS (`near_tie_gaps`): for each origin, using REAL realised target values, the
gap between the true #3 style's target and the true #4/#5/#10 styles', as a percentage of the #3
value. A small gap means exact-argmax Precision@3 has a coin-flip-like component (several
near-identical styles competing for position 3 vs. 4).

Every number in the final report is written to `reports/tables/g1_diagnostics_summary.csv`
(pooled headline numbers) and `reports/tables/g1_diagnostics_per_origin.csv` (full per-origin
traceability for a/c/d) -- nothing is narrated without a backing row in one of these two files.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from nss.features.targets import HORIZON_WEEKS, compute_forward_target
from nss.models.backtest import (
    STEP_WEEKS,
    Origin,
    block_bootstrap_ci,
    build_predictions_frame,
    generate_origin_schedule,
)
from nss.models.backtest_v2 import identify_lightgbm_origins
from nss.models.final_forecast import FINAL_MODEL_CONFIG
from nss.models.lightgbm_model import (
    INITIAL_POOL_SIZE,
    LGBMConfig,
    build_model_frame,
    feature_columns,
    predict_lightgbm,
    train_lightgbm,
)
from nss.models.metrics import (
    HIT_TOP_K,
    HIT_TOP_N_PRIMARY,
    score_predictions,
)

DEFAULT_PANEL_PATH = Path("data/processed/style_week_panel.parquet")
DEFAULT_LIGHTGBM_PER_ORIGIN_PATH = Path("reports/tables/backtest_per_origin_lightgbm.csv")
DEFAULT_SUMMARY_V2_PATH = Path("reports/tables/backtest_summary_v2.csv")
DEFAULT_SUMMARY_OUT_PATH = Path("reports/tables/g1_diagnostics_summary.csv")
DEFAULT_PER_ORIGIN_OUT_PATH = Path("reports/tables/g1_diagnostics_per_origin.csv")

# Ranks (1-indexed, by realised target value) whose gap to the true #3 is reported by
# `near_tie_gaps`. See module docstring (d).
NEAR_TIE_GAP_RANKS: tuple[int, ...] = (4, 5, 10)

# A per-origin cross-check tolerance for `main()`'s reproduced-vs-checked-in-CSV determinism
# verification (see module docstring (b)) -- this project's own D3a ablation measured bit-identical
# (`np.array_equal`, max abs diff 0.0) reproduction across separate processes, so any tolerance
# above float64 noise would already hide a real regression; this is deliberately tight, not loose.
DETERMINISM_CHECK_RTOL = 1e-9
DETERMINISM_CHECK_ATOL = 1e-9


def minimal_causal_lag_origin_steps(step_weeks: int, horizon_weeks: int) -> int:
    """Minimal integer number of origin-steps `k` making an earlier origin causally usable.

    Origins are spaced `step_weeks` calendar weeks apart. An origin `k` steps before a later origin
    `t` sits `k * step_weeks` calendar weeks before `t`; its OWN forward-target window closes
    `horizon_weeks` weeks after it, i.e. `k * step_weeks - horizon_weeks` weeks before `t`. For that
    window to have already closed by time `t` (so its realised target is genuinely known, not a
    future leak), we need `k * step_weeks - horizon_weeks >= 0`, i.e. `k >= horizon_weeks /
    step_weeks`. The minimal integer satisfying this is the ceiling of that ratio.

    Args:
        step_weeks: Calendar weeks between consecutive origins (e.g.
            `nss.models.backtest.STEP_WEEKS`).
        horizon_weeks: The forward-target horizon in weeks (e.g.
            `nss.features.targets.HORIZON_WEEKS`).

    Returns:
        The minimal integer `k` (in units of origin-steps) such that an origin `k` steps earlier has
        its own forward window fully closed by the time of the later origin.
    """
    if step_weeks <= 0 or horizon_weeks <= 0:
        raise ValueError("step_weeks and horizon_weeks must be positive")
    return -(-horizon_weeks // step_weeks)  # ceil division, same idiom as backtest.py elsewhere


# Derived, not assumed: with STEP_WEEKS=4 and HORIZON_WEEKS=13, this evaluates to 4 (16 calendar
# weeks) -- see module docstring (a) and tests/test_head_ranking_diagnostics.py for a hand-verified
# synthetic proof that k=3 (12 weeks) is NOT sufficient while k=4 (16 weeks) is.
LAG_ORIGIN_STEPS = minimal_causal_lag_origin_steps(STEP_WEEKS, HORIZON_WEEKS)
LAG_WEEKS = STEP_WEEKS * LAG_ORIGIN_STEPS


# ---------------------------------------------------------------------------
# (a) Causal persistence oracle
# ---------------------------------------------------------------------------


def persistence_oracle_predictions(
    panel: pl.DataFrame, origin_week: date, lag_weeks: int
) -> pl.DataFrame:
    """The causal persistence-oracle "prediction" for `origin_week`: the fully-resolved realised
    forward target of the causally-earlier origin `origin_week - lag_weeks`.

    Args:
        panel: The dense style-week panel.
        origin_week: The (later) origin being "predicted".
        lag_weeks: Calendar weeks to look back. Must be `>= HORIZON_WEEKS` for the earlier origin's
            own forward window to have already closed by `origin_week` (see
            `minimal_causal_lag_origin_steps`); not re-validated here, callers are trusted to pass a
            causally-valid lag.

    Returns:
        `style_key`, `origin_week` (set to the LATER `origin_week`, not the earlier one), and
        `y_pred_oracle` (the earlier origin's realised `target`, nullable per
        `compute_forward_target`'s own null-handling rule).
    """
    earlier_origin = origin_week - timedelta(weeks=lag_weeks)
    earlier_target = compute_forward_target(panel, earlier_origin, horizon_weeks=HORIZON_WEEKS)
    return earlier_target.select(
        "style_key",
        pl.lit(origin_week).alias("origin_week"),
        pl.col("target").alias("y_pred_oracle"),
    )


def score_persistence_oracle(
    panel: pl.DataFrame,
    origins: Sequence[Origin],
    eval_frame: pl.DataFrame,
    lag_weeks: int = LAG_WEEKS,
) -> pl.DataFrame:
    """Score the causal persistence oracle at every origin, via the project's own metric functions.

    Args:
        panel: The dense style-week panel.
        origins: The origins to score (the 12 LightGBM walk-forward test origins).
        eval_frame: `style_key`, `origin_week`, `y_true`, `weight` for (at least) every origin in
            `origins` -- the SAME eval population LightGBM/baselines are scored against (see
            `nss.models.backtest.build_predictions_frame`).
        lag_weeks: Causal lookback lag. Defaults to `LAG_WEEKS`.

    Returns:
        One row per origin: `origin_week`, `n_eval_set`, `n_eval` (after dropping styles where the
        oracle prediction is null, matching `nss.models.backtest.run_backtest`'s own per-method
        null-drop convention), plus `precision_at_3`, `hit_at_3_in_top10`, `hit_at_3_in_top20`
        (and every other `nss.models.metrics.METRIC_KEYS` entry, for completeness).
    """
    rows: list[dict[str, object]] = []
    for origin in origins:
        origin_eval = eval_frame.filter(pl.col("origin_week") == origin.origin_week)
        oracle = persistence_oracle_predictions(panel, origin.origin_week, lag_weeks)
        joined = origin_eval.join(oracle, on=["style_key", "origin_week"], how="left")
        sub = joined.filter(pl.col("y_pred_oracle").is_not_null())

        metrics = score_predictions(
            sub["y_true"].to_numpy(), sub["y_pred_oracle"].to_numpy(), sub["weight"].to_numpy()
        )
        n_eval = metrics.pop("n_eval")
        rows.append(
            {
                "origin_week": origin.origin_week,
                "n_eval_set": origin_eval.height,
                "n_eval": int(n_eval),
                **metrics,
            }
        )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# (b) Prediction compression
# ---------------------------------------------------------------------------


def lightgbm_walk_forward_raw_predictions(
    panel: pl.DataFrame,
    origins: list[Origin],
    config: LGBMConfig,
    pool_size: int = INITIAL_POOL_SIZE,
) -> pl.DataFrame:
    """Reproduce the FROZEN LightGBM walk-forward inference loop's raw per-style predictions.

    Mirrors `nss.models.lightgbm_model.run_lightgbm_walk_forward`'s expanding-window loop exactly
    (same `build_model_frame`, `train_lightgbm`, `predict_lightgbm` calls, same train/test split per
    origin), but returns the raw `(style_key, origin_week, y_true, y_pred)` rows instead of
    aggregated metrics -- those raw predictions were never checked in. Uses the FROZEN, already-
    established `config` (never re-tuned here) -- per this project's own verified cross-process
    determinism guarantee (D3a), this reproduces bit-identical predictions to whatever already
    produced `backtest_per_origin_lightgbm.csv`, not a new model.

    Args:
        panel: The dense style-week panel.
        origins: The FULL origin schedule (`nss.models.backtest.generate_origin_schedule(panel)`),
            NOT just the walk-forward test subset -- the pool origins are needed to train each
            expanding window.
        config: The frozen hyperparameter config. Pass
            `nss.models.final_forecast.FINAL_MODEL_CONFIG` (the winning config from the original
            bounded search).
        pool_size: Origins reserved as the initial training-only pool. Defaults to
            `nss.models.lightgbm_model.INITIAL_POOL_SIZE`.

    Returns:
        `style_key`, `origin_week`, `y_true`, `y_pred_lightgbm` for every walk-forward test origin's
        eval set, concatenated in origin order.
    """
    origin_weeks = [o.origin_week for o in origins]
    model_frame = build_model_frame(panel, origin_weeks)
    columns = feature_columns(model_frame)

    frames: list[pl.DataFrame] = []
    for test_index in range(pool_size, len(origins)):
        test_origin = origins[test_index]
        train_weeks = {o.origin_week for o in origins[:test_index]}
        train_frame = model_frame.filter(pl.col("origin_week").is_in(train_weeks))
        test_frame = model_frame.filter(pl.col("origin_week") == test_origin.origin_week)

        model = train_lightgbm(train_frame, config, columns)
        preds = predict_lightgbm(model, test_frame, columns)
        frames.append(
            test_frame.select("style_key", "origin_week", "y_true").with_columns(
                pl.Series("y_pred_lightgbm", preds)
            )
        )
    return pl.concat(frames)


def compression_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compare the pooled distribution spread of predicted vs. actual target values.

    Tests the "L2-on-log1p-target shrinks extreme predictions toward the mean" hypothesis (see
    module docstring (b)) with two ratios: the pooled std ratio, and the ratio of (p99 - median)
    spreads. A ratio well below 1.0 on both is confirming evidence; a ratio near 1.0 refutes it.

    Args:
        y_true: Pooled realised target values across every scored origin.
        y_pred: Pooled predicted values, same order/length as `y_true`.

    Returns:
        `std_true`, `std_pred`, `std_ratio_pred_over_true`, `true_p99_minus_median`,
        `pred_p99_minus_median`, `spread_ratio_pred_over_true`.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    std_true = float(np.std(y_true_arr))
    std_pred = float(np.std(y_pred_arr))
    std_ratio = std_pred / std_true if std_true != 0.0 else float("nan")

    true_p99, true_median = np.percentile(y_true_arr, 99), float(np.median(y_true_arr))
    pred_p99, pred_median = np.percentile(y_pred_arr, 99), float(np.median(y_pred_arr))
    true_spread = float(true_p99 - true_median)
    pred_spread = float(pred_p99 - pred_median)
    spread_ratio = pred_spread / true_spread if true_spread != 0.0 else float("nan")

    return {
        "std_true": std_true,
        "std_pred": std_pred,
        "std_ratio_pred_over_true": std_ratio,
        "true_p99_minus_median": true_spread,
        "pred_p99_minus_median": pred_spread,
        "spread_ratio_pred_over_true": spread_ratio,
    }


# ---------------------------------------------------------------------------
# (c) Head churn
# ---------------------------------------------------------------------------


def head_churn_fraction(
    eval_frame: pl.DataFrame,
    origins: Sequence[Origin],
    earlier_weeks: Sequence[date],
    top_k: int = HIT_TOP_K,
    top_n: int = HIT_TOP_N_PRIMARY,
) -> pl.DataFrame:
    """Fraction of each origin's TRUE top-`top_k` styles also in the causally-earlier origin's TRUE
    top-`top_n`.

    Args:
        eval_frame: `style_key`, `origin_week`, `y_true` covering every origin in `origins` AND
            every week in `earlier_weeks` (see `nss.models.backtest.build_predictions_frame`).
        origins: The origins to score (in the same order as `earlier_weeks`).
        earlier_weeks: The causally-earlier origin week for each entry of `origins`, same order/
            length (typically `origin.origin_week - timedelta(weeks=LAG_WEEKS)` for each).
        top_k: The later origin's TRUE top-k cutoff. Defaults to `HIT_TOP_K` (3).
        top_n: The earlier origin's TRUE top-n cutoff. Defaults to `HIT_TOP_N_PRIMARY` (20).

    Returns:
        One row per origin: `origin_week`, `earlier_origin_week`, `n_true_top_k` (the ACTUAL number
        of styles in the later origin's true top-k, clipped by eval-set size), `n_overlap`,
        `churn_fraction` (`n_overlap / n_true_top_k`, `nan` if `n_true_top_k` is 0).
    """
    if len(origins) != len(earlier_weeks):
        raise ValueError("origins and earlier_weeks must have the same length, in matching order")

    rows: list[dict[str, object]] = []
    for origin, earlier_week in zip(origins, earlier_weeks, strict=True):
        current = eval_frame.filter(pl.col("origin_week") == origin.origin_week).sort(
            "y_true", descending=True
        )
        earlier = eval_frame.filter(pl.col("origin_week") == earlier_week).sort(
            "y_true", descending=True
        )

        k_eff = min(top_k, current.height)
        n_eff = min(top_n, earlier.height)
        current_top_k = set(current.head(k_eff)["style_key"].to_list())
        earlier_top_n = set(earlier.head(n_eff)["style_key"].to_list())

        overlap = len(current_top_k & earlier_top_n)
        fraction = overlap / k_eff if k_eff > 0 else float("nan")
        rows.append(
            {
                "origin_week": origin.origin_week,
                "earlier_origin_week": earlier_week,
                "n_true_top_k": k_eff,
                "n_overlap": overlap,
                "churn_fraction": fraction,
            }
        )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# (d) Near-tie analysis
# ---------------------------------------------------------------------------


def near_tie_gaps(
    eval_frame: pl.DataFrame,
    origins: Sequence[Origin],
    gap_ranks: tuple[int, ...] = NEAR_TIE_GAP_RANKS,
) -> pl.DataFrame:
    """Per-origin gap between the true #3 style's target value and the true #4/#5/#10 (etc.), as a
    percentage of the #3 value.

    Args:
        eval_frame: `style_key`, `origin_week`, `y_true` covering every origin in `origins`.
        origins: The origins to score.
        gap_ranks: 1-indexed ranks (by realised target value, descending) to compare against #3.
            Defaults to `NEAR_TIE_GAP_RANKS` (4, 5, 10).

    Returns:
        One row per origin: `origin_week`, `value_3` (the true #3 style's target value), and
        `gap_3_vs_{k}_pct` for each `k` in `gap_ranks` -- `(value_3 - value_k) / value_3 * 100`.
        A gap is `None` if the eval set has fewer than `k` styles, or if `value_3` is exactly 0.0
        (percentage-of-zero is undefined, not "0% gap").
    """
    rows: list[dict[str, object]] = []
    for origin in origins:
        values = (
            eval_frame.filter(pl.col("origin_week") == origin.origin_week)
            .sort("y_true", descending=True)["y_true"]
            .to_list()
        )
        n = len(values)
        row: dict[str, object] = {"origin_week": origin.origin_week}
        value_3 = values[2] if n >= 3 else None
        row["value_3"] = value_3
        for k in gap_ranks:
            key = f"gap_3_vs_{k}_pct"
            if value_3 is None or n < k or value_3 == 0.0:
                row[key] = None
                continue
            value_k = values[k - 1]
            row[key] = (value_3 - value_k) / value_3 * 100.0
        rows.append(row)
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point: run all 4 diagnostics, print the headline numbers, write both CSVs."""
    panel = pl.read_parquet(DEFAULT_PANEL_PATH)
    origins = identify_lightgbm_origins(panel, DEFAULT_LIGHTGBM_PER_ORIGIN_PATH)
    full_origins = generate_origin_schedule(panel)

    print(f"LAG_ORIGIN_STEPS={LAG_ORIGIN_STEPS}, LAG_WEEKS={LAG_WEEKS}")

    origin_weeks = [o.origin_week for o in origins]
    earlier_weeks = [ow - timedelta(weeks=LAG_WEEKS) for ow in origin_weeks]
    all_weeks = sorted(set(origin_weeks) | set(earlier_weeks))
    eval_all = build_predictions_frame(panel, all_weeks).select(
        "style_key", "origin_week", "y_true", "weight"
    )

    # (a) persistence oracle
    oracle_per_origin = score_persistence_oracle(panel, origins, eval_all, lag_weeks=LAG_WEEKS)
    oracle_hit20_mean, oracle_hit20_lo, oracle_hit20_hi = block_bootstrap_ci(
        oracle_per_origin["hit_at_3_in_top20"].to_list()
    )
    oracle_hit10_mean, oracle_hit10_lo, oracle_hit10_hi = block_bootstrap_ci(
        oracle_per_origin["hit_at_3_in_top10"].to_list()
    )
    oracle_p3_mean, oracle_p3_lo, oracle_p3_hi = block_bootstrap_ci(
        oracle_per_origin["precision_at_3"].to_list()
    )
    print(
        f"(a) persistence oracle -- precision_at_3={oracle_p3_mean:.4f} "
        f"hit_at_3_in_top10={oracle_hit10_mean:.4f} hit_at_3_in_top20={oracle_hit20_mean:.4f}"
    )

    # (b) compression -- reproduce the frozen model's raw predictions, cross-check determinism.
    raw_preds = lightgbm_walk_forward_raw_predictions(panel, full_origins, FINAL_MODEL_CONFIG)
    checked_in = pl.read_csv(DEFAULT_LIGHTGBM_PER_ORIGIN_PATH, try_parse_dates=True)
    for origin in origins:
        reproduced = raw_preds.filter(pl.col("origin_week") == origin.origin_week)
        reproduced_metrics = score_predictions(
            reproduced["y_true"].to_numpy(), reproduced["y_pred_lightgbm"].to_numpy()
        )
        checked_in_p3 = checked_in.filter(pl.col("origin_week") == origin.origin_week)[
            "precision_at_3"
        ][0]
        np.testing.assert_allclose(
            reproduced_metrics["precision_at_3"],
            checked_in_p3,
            rtol=DETERMINISM_CHECK_RTOL,
            atol=DETERMINISM_CHECK_ATOL,
            err_msg=(
                f"Reproduced LightGBM predictions at origin {origin.origin_week} do not match "
                f"the checked-in {DEFAULT_LIGHTGBM_PER_ORIGIN_PATH} -- determinism guarantee "
                "broken, refusing to trust reproduced predictions for compression stats."
            ),
        )
    print(
        f"(b) determinism cross-check PASSED for all {len(origins)} origins "
        f"(reproduced precision_at_3 == checked-in CSV, rtol={DETERMINISM_CHECK_RTOL})"
    )

    compression = compression_stats(
        raw_preds["y_true"].to_numpy(), raw_preds["y_pred_lightgbm"].to_numpy()
    )
    std_ratio = compression["std_ratio_pred_over_true"]
    spread_ratio = compression["spread_ratio_pred_over_true"]
    print(f"(b) compression -- std_ratio={std_ratio:.4f} spread_ratio={spread_ratio:.4f}")

    # (c) head churn
    churn_per_origin = head_churn_fraction(eval_all, origins, earlier_weeks)
    pooled_churn = churn_per_origin["n_overlap"].sum() / churn_per_origin["n_true_top_k"].sum()
    print(f"(c) head churn -- pooled fraction={pooled_churn:.4f}")

    # (d) near-tie gaps
    near_tie_per_origin = near_tie_gaps(eval_all, origins)
    gap_summary: dict[int, dict[str, float]] = {}
    for k in NEAR_TIE_GAP_RANKS:
        col = f"gap_3_vs_{k}_pct"
        vals = near_tie_per_origin[col].drop_nulls().to_list()
        gap_summary[k] = {"mean": float(np.mean(vals)), "median": float(np.median(vals))}
        print(
            f"(d) near-tie gap #3 vs #{k}: mean={gap_summary[k]['mean']:.2f}% "
            f"median={gap_summary[k]['median']:.2f}%"
        )

    # Read the current LightGBM pooled numbers to report side-by-side (data-driven, not
    # hand-copied -- see module docstring).
    summary_v2 = pl.read_csv(DEFAULT_SUMMARY_V2_PATH)
    lgbm_pooled = summary_v2.filter(
        (pl.col("method") == "lightgbm") & (pl.col("split") == "pooled")
    )
    lgbm_hit20 = lgbm_pooled["hit_at_3_in_top20_mean"][0]
    lgbm_hit10 = lgbm_pooled["hit_at_3_in_top10_mean"][0]
    lgbm_p3 = lgbm_pooled["precision_at_3_mean"][0]

    summary_rows: list[dict[str, object]] = [
        {
            "diagnostic": "a_persistence_oracle",
            "metric": "precision_at_3",
            "value": oracle_p3_mean,
            "ci_low": oracle_p3_lo,
            "ci_high": oracle_p3_hi,
            "lightgbm_pooled_value": lgbm_p3,
        },
        {
            "diagnostic": "a_persistence_oracle",
            "metric": "hit_at_3_in_top10",
            "value": oracle_hit10_mean,
            "ci_low": oracle_hit10_lo,
            "ci_high": oracle_hit10_hi,
            "lightgbm_pooled_value": lgbm_hit10,
        },
        {
            "diagnostic": "a_persistence_oracle",
            "metric": "hit_at_3_in_top20",
            "value": oracle_hit20_mean,
            "ci_low": oracle_hit20_lo,
            "ci_high": oracle_hit20_hi,
            "lightgbm_pooled_value": lgbm_hit20,
        },
        {
            "diagnostic": "b_compression",
            "metric": "std_ratio_pred_over_true",
            "value": compression["std_ratio_pred_over_true"],
            "ci_low": None,
            "ci_high": None,
            "lightgbm_pooled_value": None,
        },
        {
            "diagnostic": "b_compression",
            "metric": "spread_ratio_pred_over_true_p99_minus_median",
            "value": compression["spread_ratio_pred_over_true"],
            "ci_low": None,
            "ci_high": None,
            "lightgbm_pooled_value": None,
        },
        {
            "diagnostic": "c_head_churn",
            "metric": f"pooled_fraction_top{HIT_TOP_K}_in_earlier_top{HIT_TOP_N_PRIMARY}",
            "value": float(pooled_churn),
            "ci_low": None,
            "ci_high": None,
            "lightgbm_pooled_value": None,
        },
    ]
    for k in NEAR_TIE_GAP_RANKS:
        summary_rows.append(
            {
                "diagnostic": "d_near_tie",
                "metric": f"gap_3_vs_{k}_pct_mean",
                "value": gap_summary[k]["mean"],
                "ci_low": None,
                "ci_high": None,
                "lightgbm_pooled_value": None,
            }
        )
        summary_rows.append(
            {
                "diagnostic": "d_near_tie",
                "metric": f"gap_3_vs_{k}_pct_median",
                "value": gap_summary[k]["median"],
                "ci_low": None,
                "ci_high": None,
                "lightgbm_pooled_value": None,
            }
        )

    per_origin = (
        oracle_per_origin.select(
            "origin_week",
            "n_eval_set",
            "n_eval",
            pl.col("precision_at_3").alias("oracle_precision_at_3"),
            pl.col("hit_at_3_in_top10").alias("oracle_hit_at_3_in_top10"),
            pl.col("hit_at_3_in_top20").alias("oracle_hit_at_3_in_top20"),
        )
        .join(
            churn_per_origin.select(
                "origin_week", "earlier_origin_week", "n_true_top_k", "n_overlap", "churn_fraction"
            ),
            on="origin_week",
        )
        .join(near_tie_per_origin, on="origin_week")
    )

    DEFAULT_SUMMARY_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(summary_rows).write_csv(DEFAULT_SUMMARY_OUT_PATH)
    per_origin.write_csv(DEFAULT_PER_ORIGIN_OUT_PATH)
    print(f"Wrote {DEFAULT_SUMMARY_OUT_PATH} ({len(summary_rows)} rows)")
    print(f"Wrote {DEFAULT_PER_ORIGIN_OUT_PATH} ({per_origin.height} rows)")


if __name__ == "__main__":
    main()
