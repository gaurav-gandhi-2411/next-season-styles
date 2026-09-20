"""Q1: do cross-style neighbourhood features improve the embargoed forecast? One shot.

Arms (both retrained here, same code path, same 12 test origins, same 13-week embargo protocol as
`backtest_embargo_check`, both with the frozen `final_forecast.FINAL_MODEL_CONFIG`, seed 42):

- control:   `build_model_frame` features (the shipped feature set);
- treatment: control + the 15 `neighbourhood_features.NEIGHBOURHOOD_FEATURE_COLS`.

Baselines (seasonal_naive, ewma_persistence, global_mean, parent_category_mean) come from
`backtest.run_backtest` on the same origins. Paired per-origin differences (block bootstrap: 4,
2,000 resamples, seed 42, `backtest_v2.paired_diff_table`) are written for treatment - control and
treatment - each baseline.

DECISION RULE (fixed before running): ADOPT the neighbourhood features only if the paired
treatment - control difference on Hit@3-in-top20 (pooled over the 12 origins) has a block-bootstrap
95% CI that excludes zero on the improving side (`ci_lo > 0`). Otherwise keep the current model and
report the negative result. No tuning, no fifth objective, no re-run to chase a better number
(re-running only to confirm determinism is fine).

CONTROL REPRODUCTION GATE: the control arm must reproduce the embargoed headline recorded by
`backtest_embargo_check` (Hit@3-in-top20 0.528 over the 12 origins, to +-0.0006) and equal the
metrics of `run_embargoed_walk_forward` called directly; otherwise the run aborts BEFORE the
treatment arm is trained.

SHAP: for the treatment model trained on the LAST embargoed fold's training set (test origin
2020-06-01, train origins <= 2020-02-10, i.e. 16 origins), mean |SHAP| over that training frame,
with each feature's rank among all features. The neighbourhood features' ranks are a finding about
trend propagation whether or not the headline metric moves. (In-sample SHAP: it says what the model
uses, not that the use generalises -- the paired out-of-sample table is what decides.)

Outputs (new files only): reports/tables/neighbourhood_per_origin.csv,
neighbourhood_paired_diff.csv, neighbourhood_decision.csv, neighbourhood_shap.csv, plus the last-fold
treatment booster in the gitignored scratch folder `models/q1_scratch/` (never committed).

Usage:
    python -m nss.models.neighbourhood_experiment
"""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from nss.features.neighbourhood_features import (
    NEIGHBOURHOOD_FEATURE_COLS,
    add_neighbourhood_features,
    build_neighbourhood_features,
)
from nss.models.backtest import (
    WMAPE_WEIGHT_COL,
    Origin,
    generate_origin_schedule,
    run_backtest,
)
from nss.models.backtest_embargo_check import (
    DEFAULT_PANEL_PATH,
    embargoed_train_origin_weeks,
    run_embargoed_walk_forward,
)
from nss.models.backtest_v2 import identify_lightgbm_origins, paired_diff_table
from nss.models.final_forecast import FINAL_MODEL_CONFIG
from nss.models.lightgbm_model import (
    INITIAL_POOL_SIZE,
    LGBMConfig,
    build_model_frame,
    compute_global_shap_importance,
    feature_columns,
    predict_lightgbm,
    train_lightgbm,
)
from nss.models.metrics import METRIC_KEYS, score_predictions

CONTROL = "lightgbm_control"
TREATMENT = "lightgbm_neighbourhood"
BASELINES = ("seasonal_naive", "ewma_persistence", "global_mean", "parent_category_mean")
PRIMARY_METRIC = "hit_at_3_in_top20"
# recorded embargoed control Hit@3-in-top20 over the 12 origins (backtest_embargo_check / brief)
CONTROL_EXPECTED_TOP20 = 0.528
CONTROL_TOL = 0.0006  # 0.5278 rounds to 0.528; 19/36 = 0.52777...

SCRATCH_DIR = Path("models/q1_scratch")
OUT_DIR = Path("reports/tables")
PER_ORIGIN_OUT = OUT_DIR / "neighbourhood_per_origin.csv"
PAIRED_OUT = OUT_DIR / "neighbourhood_paired_diff.csv"
DECISION_OUT = OUT_DIR / "neighbourhood_decision.csv"
SHAP_OUT = OUT_DIR / "neighbourhood_shap.csv"


def _frame(panel: pl.DataFrame, origins: list[Origin], with_neighbourhood: bool) -> pl.DataFrame:
    frame = build_model_frame(panel, [o.origin_week for o in origins])
    return add_neighbourhood_features(frame, panel) if with_neighbourhood else frame


def run_arm(
    panel: pl.DataFrame,
    origins: list[Origin],
    config: LGBMConfig,
    method: str,
    with_neighbourhood: bool,
) -> pl.DataFrame:
    """Embargoed walk-forward for one arm; same loop as `run_embargoed_walk_forward`.

    Only the feature frame differs between arms. Raises if any test origin has no embargoed
    training data (the 12 shared origins all do).
    """
    model_frame = _frame(panel, origins, with_neighbourhood)
    columns = feature_columns(model_frame)
    rows: list[dict[str, object]] = []
    for test_index in range(INITIAL_POOL_SIZE, len(origins)):
        test_origin = origins[test_index]
        train_weeks = embargoed_train_origin_weeks(origins, test_index)
        train_frame = model_frame.filter(pl.col("origin_week").is_in(train_weeks))
        test_frame = model_frame.filter(pl.col("origin_week") == test_origin.origin_week)
        if not train_weeks or train_frame.height == 0 or test_frame.height == 0:
            raise RuntimeError(f"no embargoed training data for {test_origin.origin_week}")
        model = train_lightgbm(train_frame, config, columns)
        preds = predict_lightgbm(model, test_frame, columns)
        metrics = score_predictions(
            test_frame["y_true"].to_numpy(), preds, test_frame[WMAPE_WEIGHT_COL].to_numpy()
        )
        n_eval = metrics.pop("n_eval")
        rows.append(
            {
                "origin_week": test_origin.origin_week,
                "method": method,
                "has_52w_lag": test_origin.has_52w_lag,
                "is_covid": test_origin.is_covid,
                "n_eval_set": test_frame.height,
                "n_eval": int(n_eval),
                **metrics,
            }
        )
    return pl.DataFrame(rows)


def shap_table(panel: pl.DataFrame, origins: list[Origin], config: LGBMConfig) -> pl.DataFrame:
    """Mean |SHAP| of every feature for the treatment model on the LAST embargoed fold."""
    model_frame = _frame(panel, origins, with_neighbourhood=True)
    columns = feature_columns(model_frame)
    train_weeks = embargoed_train_origin_weeks(origins, len(origins) - 1)
    train_frame = model_frame.filter(pl.col("origin_week").is_in(train_weeks))
    model = train_lightgbm(train_frame, config, columns)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)  # gitignored scratch path, never committed
    model.booster_.save_model(str(SCRATCH_DIR / "treatment_last_fold.txt"))
    imp = compute_global_shap_importance(model, train_frame, columns)
    return imp.with_row_index("rank", offset=1).with_columns(
        pl.col("feature").is_in(NEIGHBOURHOOD_FEATURE_COLS).alias("is_neighbourhood"),
        pl.lit("last_embargoed_fold_train_frame").alias("shap_model"),
        pl.lit(len(train_weeks)).alias("n_train_origins"),
        pl.lit(train_frame.height).alias("n_train_rows"),
    )


def decision_table(paired_vs_control: pl.DataFrame, per_origin: pl.DataFrame) -> pl.DataFrame:
    """The recorded decision: pooled treatment vs control on the five metrics, rule on top-20."""
    pooled = paired_vs_control.filter(pl.col("split") == "pooled")
    top20 = pooled.filter(pl.col("metric") == PRIMARY_METRIC).row(0, named=True)
    adopt = bool(top20["diff_ci_low"] > 0)
    decision = "ADOPT" if adopt else "REJECT (keep current model)"
    rows = []
    for metric in METRIC_KEYS:
        r = pooled.filter(pl.col("metric") == metric).row(0, named=True)
        means = {
            m: float(per_origin.filter(pl.col("method") == m)[metric].mean())
            for m in (CONTROL, TREATMENT)
        }
        rows.append(
            {
                "metric": metric,
                "control": means[CONTROL],
                "treatment": means[TREATMENT],
                "diff": r["mean_diff"],
                "ci_lo": r["diff_ci_low"],
                "ci_hi": r["diff_ci_high"],
                "decision_rule": (
                    "adopt iff paired diff CI lower bound > 0 on hit_at_3_in_top20 (decides)"
                    if metric == PRIMARY_METRIC
                    else "reported only (not decision-bearing)"
                ),
                "decision": decision,
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    """Run both arms + baselines, write the four Q1 tables, print the decision."""
    panel = pl.read_parquet(DEFAULT_PANEL_PATH)
    all_origins = generate_origin_schedule(panel)
    shared = identify_lightgbm_origins(panel)

    # causality sanity on the REAL panel (the unit tests use synthetic data): the neighbourhood
    # frame for the last origin must be identical when every row after it is dropped.
    last = all_origins[-1].origin_week
    full = build_neighbourhood_features(panel, [last]).sort("style_key")
    trunc = build_neighbourhood_features(panel.filter(pl.col("week_start") <= last), [last]).sort(
        "style_key"
    )
    assert full.equals(trunc), "neighbourhood features changed when post-origin rows were dropped"

    control = run_arm(panel, all_origins, FINAL_MODEL_CONFIG, CONTROL, with_neighbourhood=False)
    # control reproduction gate (vs P's/the embargo module's own loop and the recorded headline)
    direct, skipped = run_embargoed_walk_forward(panel, all_origins, FINAL_MODEL_CONFIG)
    assert not skipped
    for metric in METRIC_KEYS:
        a = control.sort("origin_week")[metric].to_numpy()
        b = direct.sort("origin_week")[metric].to_numpy()
        assert all(
            math.isclose(x, y, rel_tol=0, abs_tol=1e-12) for x, y in zip(a, b, strict=True)
        ), metric
    control_top20 = float(control[PRIMARY_METRIC].mean())
    print(f"control {PRIMARY_METRIC} = {control_top20:.6f} (recorded {CONTROL_EXPECTED_TOP20})")
    if abs(control_top20 - CONTROL_EXPECTED_TOP20) > CONTROL_TOL:
        raise SystemExit("control did not reproduce the recorded embargoed headline; stopping")

    treatment = run_arm(panel, all_origins, FINAL_MODEL_CONFIG, TREATMENT, with_neighbourhood=True)
    baseline = run_backtest(panel, shared)
    combined = pl.concat(
        [baseline, control.select(baseline.columns), treatment.select(baseline.columns)],
        how="vertical",
    )
    combined.write_csv(PER_ORIGIN_OUT)

    paired = paired_diff_table(
        combined, treatment_method=TREATMENT, baseline_methods=(CONTROL, *BASELINES)
    )
    paired.write_csv(PAIRED_OUT)
    decision = decision_table(paired.filter(pl.col("method_b") == CONTROL), combined)
    decision.write_csv(DECISION_OUT)
    shap = shap_table(panel, all_origins, FINAL_MODEL_CONFIG)
    shap.write_csv(SHAP_OUT)

    with pl.Config(
        tbl_rows=80, tbl_width_chars=200, tbl_formatting="ASCII_FULL", float_precision=4
    ):
        print(decision)
        print(
            paired.filter(
                (pl.col("split") == "pooled") & pl.col("metric").is_in(METRIC_KEYS)
            ).select("method_b", "metric", "mean_diff", "diff_ci_low", "diff_ci_high")
        )
        print(shap.filter(pl.col("is_neighbourhood")).select("rank", "feature", "mean_abs_shap"))
        print(shap.head(10).select("rank", "feature", "mean_abs_shap"))
        print(f"n features total: {shap.height}")


if __name__ == "__main__":
    main()
