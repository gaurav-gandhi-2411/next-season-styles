"""Train the external-signal treatment model and decide. ONE SHOT.

HYPOTHESIS (stated before any treatment result): every feature in the model, and the seasonal-naive
baseline, derives from the retailer's own sales history. That symmetry is why the model only
matches seasonal-naive on top-k. External search interest is information seasonal-naive
structurally cannot have; a style rising in public search before it rises in sales would be a
leading indicator at the 13-week horizon. Source: weekly Google Trends per "<colour> <product>"
term (`nss.external`); features: `nss.features.signal_features`.

ARMS (same 12 embargoed test origins = `origins[INITIAL_POOL_SIZE:]`, training origins
`o <= t - 16 weeks`, seed 42, one shared `train_lightgbm`, ONE LOCKED CONFIG for both arms,
`final_forecast.FINAL_MODEL_CONFIG`, the shipped model; nothing is tuned):

- control:   `build_model_frame` features (the shipped model under the embargo protocol);
- treatment: control features + `SIGNAL_FEATURE_COLS`.

Baselines (seasonal_naive, ewma_persistence, global_mean, parent_category_mean) come from
`backtest.run_backtest` on the same origins. Paired per-origin differences (block bootstrap: block
4, 2,000 resamples, seed 42; `backtest_v2.paired_diff_table`) are written for treatment - control
and treatment - each baseline, all seven metrics, pooled / covid / non_covid.

DECISION RULE (`RULE_TEXT`, fixed before the treatment arm was ever run; not to be edited after):
ADOPT iff the paired treatment - control difference (pooled over the 12 origins)
(A) has a block-bootstrap 95% CI excluding zero on the improving side on Hit@3-in-top20
    (`ci_lo > 0`), OR
(B) has `ci_lo > 0` on BOTH NDCG@10 and Spearman AND a Hit@3-in-top20 point estimate (mean paired
    difference) that is not worse (`>= 0`).
Otherwise REJECT and record the negative result. No tuning, no second variant, no re-run to chase
a better number (re-running only to confirm determinism is fine).

CONTROL REPRODUCTION GATE: before the treatment arm is trained, the control arm must equal
`backtest_embargo_check.run_embargoed_walk_forward` metric-for-metric and reproduce the recorded
embargoed headline Hit@3-in-top20 (`reports/tables/backtest_embargo_summary.csv`, lightgbm/pooled,
0.528, tolerance 0.0006); otherwise the run aborts.

SHAP (in-sample, treatment model trained on the LAST embargoed fold's training set, test origin
2020-06-01): mean |SHAP| rank of every feature, and the share of total attributed to the signal
features and to the LEAD features specifically. In-sample SHAP says what the model uses, not that
the use generalises; the paired out-of-sample table decides adoption.

NOT RUN, on purpose: a placebo arm (signal shuffled across styles). It would say whether any gain
is signal content or extra columns, which only matters if the treatment wins; the brief allows one
shot. If the treatment is adopted, that control is the first thing to add.

Outputs (new files): `external_signals_per_origin.csv`, `external_signals_paired_diff.csv`,
`external_signals_decision.csv`, `external_signals_shap.csv`, `external_signals_origin_coverage.csv`
under `reports/tables/`.

Usage:
    python -m nss.models.external_signals_experiment
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl
import shap

from nss.external.term_mapping import map_styles
from nss.external.trends_fetch import ordered_terms
from nss.external.trends_load import load_weekly
from nss.features.signal_features import LEAD_FEATURE_COLS, SIGNAL_FEATURE_COLS, add_signal_features
from nss.models.backtest import Origin, generate_origin_schedule, run_backtest
from nss.models.backtest_embargo_check import (
    embargoed_train_origin_weeks,
    run_embargoed_walk_forward,
)
from nss.models.backtest_v2 import identify_lightgbm_origins, paired_diff_table
from nss.models.experiment_common import load_panel, run_arm
from nss.models.final_forecast import FINAL_MODEL_CONFIG
from nss.models.lightgbm_model import (
    _to_lgb_matrix,
    build_model_frame,
    feature_columns,
    train_lightgbm,
)
from nss.models.metrics import METRIC_KEYS

CONTROL = "lightgbm_control"
TREATMENT = "lightgbm_signal"
BASELINES = ("seasonal_naive", "ewma_persistence", "global_mean", "parent_category_mean")
PRIMARY_METRIC = "hit_at_3_in_top20"
CONTROL_TOL = 0.0006
EMBARGO_SUMMARY = Path("reports/tables/backtest_embargo_summary.csv")
OUT_DIR = Path("reports/tables")
PER_ORIGIN_OUT = OUT_DIR / "external_signals_per_origin.csv"
PAIRED_OUT = OUT_DIR / "external_signals_paired_diff.csv"
DECISION_OUT = OUT_DIR / "external_signals_decision.csv"
SHAP_OUT = OUT_DIR / "external_signals_shap.csv"
ORIGIN_COVERAGE_OUT = OUT_DIR / "external_signals_origin_coverage.csv"

RULE_TEXT = (
    "ADOPT iff the pooled paired treatment-minus-control difference (12 origins, block bootstrap "
    "4 / 2000 resamples / seed 42) has (A) ci_lo > 0 on hit_at_3_in_top20, OR (B) ci_lo > 0 on "
    "BOTH ndcg_at_10 and spearman_rho AND a hit_at_3_in_top20 mean difference >= 0. "
    "Otherwise REJECT."
)


def decide(paired_vs_control: pl.DataFrame) -> dict[str, object]:
    """Apply the fixed decision rule to the pooled treatment - control paired rows."""
    pooled = paired_vs_control.filter(pl.col("split") == "pooled")

    def row(metric: str) -> dict[str, float]:
        return pooled.filter(pl.col("metric") == metric).row(0, named=True)

    top20, ndcg, rho = row(PRIMARY_METRIC), row("ndcg_at_10"), row("spearman_rho")
    rule_a = bool(top20["diff_ci_low"] > 0)
    rule_b = bool(ndcg["diff_ci_low"] > 0 and rho["diff_ci_low"] > 0 and top20["mean_diff"] >= 0)
    return {
        "rule_a": rule_a,
        "rule_b": rule_b,
        "decision": "ADOPT" if (rule_a or rule_b) else "REJECT",
    }


def decision_table(paired_vs_control: pl.DataFrame, per_origin: pl.DataFrame) -> pl.DataFrame:
    """The recorded decision: pooled control vs treatment on all seven metrics plus the rule."""
    verdict = decide(paired_vs_control)
    pooled = paired_vs_control.filter(pl.col("split") == "pooled")
    bearing = {PRIMARY_METRIC: "rule A", "ndcg_at_10": "rule B", "spearman_rho": "rule B"}
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
                "decision_bearing": bearing.get(metric, "reported only"),
                "rule_a_met": verdict["rule_a"],
                "rule_b_met": verdict["rule_b"],
                "decision": verdict["decision"],
                "rule_text": RULE_TEXT,
            }
        )
    return pl.DataFrame(rows)


def _group(name: str) -> str:
    if name in LEAD_FEATURE_COLS:
        return "signal_lead"
    if name in SIGNAL_FEATURE_COLS:
        return "signal"
    return "existing"


def shap_analysis(frame: pl.DataFrame, origins: list[Origin]) -> pl.DataFrame:
    """Global SHAP for the treatment model on the LAST embargoed fold, one row per feature."""
    columns = feature_columns(frame)
    train_weeks = embargoed_train_origin_weeks(origins, len(origins) - 1)
    train_frame = frame.filter(pl.col("origin_week").is_in(train_weeks))
    model = train_lightgbm(train_frame, FINAL_MODEL_CONFIG, columns)
    values = np.asarray(shap.TreeExplainer(model).shap_values(_to_lgb_matrix(train_frame, columns)))
    mean_abs = np.abs(values).mean(axis=0)
    table = (
        pl.DataFrame(
            {
                "feature": columns,
                "group": [_group(c) for c in columns],
                "mean_abs_shap": mean_abs.tolist(),
            }
        )
        .sort("mean_abs_shap", descending=True)
        .with_row_index("rank", offset=1)
        .with_columns((pl.col("mean_abs_shap") / pl.col("mean_abs_shap").sum()).alias("share"))
    )
    return table.with_columns(
        pl.lit(len(train_weeks)).alias("n_train_origins"),
        pl.lit(train_frame.height).alias("n_train_rows"),
        pl.lit(len(columns)).alias("n_features"),
    )


def origin_coverage(treatment_frame: pl.DataFrame, test_weeks: list) -> pl.DataFrame:
    """Per test origin: share of eval-set styles that have a non-null signal slope."""
    return (
        treatment_frame.filter(pl.col("origin_week").is_in(test_weeks))
        .group_by("origin_week")
        .agg(
            pl.len().alias("n_eval_styles"),
            pl.col("sg_slope_4w").is_not_null().sum().alias("n_with_signal"),
        )
        .with_columns(
            (pl.col("n_with_signal") / pl.col("n_eval_styles")).alias("share_with_signal")
        )
        .sort("origin_week")
    )


def main() -> None:
    """Control gate -> treatment arm -> paired tables -> decision -> SHAP."""
    panel = load_panel()
    all_origins = generate_origin_schedule(panel)
    shared = identify_lightgbm_origins(panel)
    test_weeks = [o.origin_week for o in shared]
    control_frame = build_model_frame(panel, [o.origin_week for o in all_origins])

    control = run_arm(control_frame, all_origins, FINAL_MODEL_CONFIG, CONTROL)
    direct, skipped = run_embargoed_walk_forward(panel, all_origins, FINAL_MODEL_CONFIG)
    assert not skipped
    for metric in METRIC_KEYS:
        a = control.sort("origin_week")[metric].to_numpy()
        b = direct.sort("origin_week")[metric].to_numpy()
        assert all(math.isclose(x, y, rel_tol=0, abs_tol=1e-12) for x, y in zip(a, b, strict=True))
    recorded = float(
        pl.read_csv(EMBARGO_SUMMARY)
        .filter((pl.col("method") == "lightgbm") & (pl.col("split") == "pooled"))[
            f"{PRIMARY_METRIC}_mean"
        ]
        .item()
    )
    control_top20 = float(control[PRIMARY_METRIC].mean())
    print(f"control {PRIMARY_METRIC} = {control_top20:.6f}; recorded {recorded:.6f} (0.528)")
    if abs(control_top20 - recorded) > CONTROL_TOL or abs(control_top20 - 0.528) > CONTROL_TOL:
        raise SystemExit("control did not reproduce the recorded embargoed headline; stopping")

    styles = panel.select("style_key", "product_type_name", "perceived_colour_master_name")
    style_terms = map_styles(styles)
    weekly = load_weekly([t for t, _ in ordered_terms()])
    treatment_frame = add_signal_features(control_frame, panel, style_terms, weekly)
    assert treatment_frame.height == control_frame.height
    assert treatment_frame.select(control_frame.columns).equals(control_frame)
    n_new = len(feature_columns(treatment_frame)) - len(feature_columns(control_frame))
    print(f"treatment adds {n_new} features")
    origin_coverage(treatment_frame, test_weeks).write_csv(ORIGIN_COVERAGE_OUT)
    treatment = run_arm(treatment_frame, all_origins, FINAL_MODEL_CONFIG, TREATMENT)

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
    shap_table = shap_analysis(treatment_frame, all_origins)
    shap_table.write_csv(SHAP_OUT)

    with pl.Config(
        tbl_rows=80,
        tbl_width_chars=220,
        tbl_formatting="ASCII_FULL",
        float_precision=4,
        tbl_cols=-1,
    ):
        print(decision.drop("rule_text"))
        keep = ["method_b", "metric", "mean_diff", "diff_ci_low", "diff_ci_high"]
        print(
            paired.filter(
                (pl.col("split") == "pooled") & pl.col("metric").is_in(METRIC_KEYS)
            ).select(keep)
        )
        print(shap_table.filter(pl.col("group") != "existing"))
        print(shap_table.group_by("group").agg(pl.col("share").sum()).sort("group"))
    print(f"DECISION: {decision['decision'][0]}")


if __name__ == "__main__":
    main()
