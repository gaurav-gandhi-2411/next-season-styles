"""U4: train the treatment model (current features + U1 customer + U2 price) and decide. ONE SHOT.

Arms (both retrained here on the same 12 embargoed test origins, seed 42, same `train_lightgbm`):

- control:   `build_model_frame` features with `final_forecast.FINAL_MODEL_CONFIG` (the shipped
             model under the embargo protocol);
- treatment: control features + `CUSTOMER_FEATURE_COLS` + `PRICE_FEATURE_COLS`, with the config
             chosen by U3 (`reports/tables/u3_selected_config.csv`, the row with `selected`).

Baselines (seasonal_naive, ewma_persistence, global_mean, parent_category_mean) come from
`backtest.run_backtest` on the same origins. Paired per-origin differences (block bootstrap: block
4, 2,000 resamples, seed 42; `backtest_v2.paired_diff_table`) are written for treatment - control
and treatment - each baseline, all seven metrics, pooled / covid / non_covid splits.

DECISION RULE (`RULE_TEXT`, fixed before any treatment result was computed; not to be edited after):
ADOPT iff the paired treatment - control difference (pooled over the 12 origins)
(A) has a block-bootstrap 95% CI excluding zero on the improving side on Hit@3-in-top20
    (`ci_lo > 0`), OR
(B) has `ci_lo > 0` on BOTH NDCG@10 and Spearman AND a Hit@3-in-top20 point estimate (mean paired
    difference) that is not worse (`>= 0`).
Otherwise REJECT and record the negative result with its SHAP finding. No tuning, no further
variant, no re-run to chase a better number (re-running only to confirm determinism is fine).

CONTROL REPRODUCTION GATE: before the treatment arm is trained, the control arm must (i) equal
`backtest_embargo_check.run_embargoed_walk_forward` metric-for-metric and (ii) reproduce the
recorded embargoed headline Hit@3-in-top20 (`reports/tables/backtest_embargo_summary.csv`,
lightgbm/pooled, 0.528 over the 12 origins, tolerance 0.0006); otherwise the run aborts.

SHAP (in-sample, treatment model trained on the LAST embargoed fold's training set, i.e. test origin
2020-06-01, 16 training origins): mean |SHAP| rank of every feature, and for the growth/narrowing
hypothesis the sign of the Spearman correlation between each hypothesis feature's value and its SHAP
value. HYPOTHESIS (stated before running): a style whose buyer base is BROADENING -- falling age
concentration (`cust_age_std`, `cust_age_gini` slopes < 0), rising postal spread
(`cust_geo_distinct_postal` slope > 0, `cust_geo_hhi` slope < 0) and rising new-buyer share
(`cust_repeat_share` slope < 0) -- sustains growth; a NARROWING one is peaking. Expected sign of
corr(feature, SHAP) is therefore `HYPOTHESIS_FEATURES[feature]`. VERDICT (fixed now): SUPPORTED if
at least 8 of the 10 features have the expected sign with |rho| >= 0.05; CONTRADICTED if at most
2 do; otherwise MIXED. A model-free supplementary check reports the per-origin Spearman between
each hypothesis feature and realised log-growth (`y_true - log1p(ewma_halflife_13w)`) over the 12
test origins with a block-bootstrap CI.
In-sample SHAP says what the model uses, not that the use generalises; the paired out-of-sample
table decides adoption.

ERRATUM (added AFTER the run; the code, the recorded rule and `u4_hypothesis.csv` are unchanged):
"falling age concentration" means RISING age dispersion, so the expected sign for the four
`cust_age_std_*` / `cust_age_gini_*` slopes should have been +1, not -1 as coded in
`HYPOTHESIS_FEATURES`. As declared (age signs -1) 4 of 10 features are SHAP-consistent and 6 of 10
growth-consistent (verdict MIXED); with the four age signs flipped, 7 of 10 are SHAP-consistent
(verdict still MIXED, needs 8) and 10 of 10 growth-consistent. The verdict is MIXED under both
readings, so the specification error does not change the conclusion; the adoption rule is
unaffected.

NEW TOP-3: the treatment model is retrained on the wide final-model origin set with the U3 config,
scored at the forecast origin, and passed through the SAME selection guards as the shipped pipeline
(`final_forecast` guards -> `diversity_forecast.select_t2_emerging` ->
`reselect_final_three.reselect`). The same procedure run with the control (current features, shipped
config) must reproduce the committed `top_styles_final_three.csv`, otherwise the comparison is
flagged.

Outputs (new files only): u4_per_origin.csv, u4_paired_diff.csv, u4_decision.csv, u4_shap.csv,
u4_hypothesis.csv, u4_new_top3.csv under `reports/tables/`; the last-fold booster goes to the
gitignored `models_scratch/`.

Usage:
    python -m nss.models.u4_train_decide
"""

from __future__ import annotations

import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import shap
from scipy.stats import spearmanr

from nss.features.customer_features import CUSTOMER_FEATURE_COLS
from nss.features.model_features import build_features
from nss.features.price_features import PRICE_FEATURE_COLS
from nss.features.style_panel import STYLE_KEY_COLS
from nss.models import final_forecast
from nss.models.backtest import Origin, block_bootstrap_ci, generate_origin_schedule, run_backtest
from nss.models.backtest_embargo_check import (
    embargoed_train_origin_weeks,
    run_embargoed_walk_forward,
)
from nss.models.backtest_v2 import identify_lightgbm_origins, paired_diff_table
from nss.models.diversity_forecast import select_t2_emerging
from nss.models.final_forecast import FINAL_MODEL_CONFIG
from nss.models.lightgbm_model import (
    HYPERPARAM_GRID,
    LGBMConfig,
    _to_lgb_matrix,
    build_model_frame,
    feature_columns,
    train_lightgbm,
)
from nss.models.metrics import METRIC_KEYS
from nss.models.reselect_final_three import intimate_product_types, reselect
from nss.models.u_common import (
    ARTICLES_PATH,
    build_treatment_frame,
    load_panel,
    run_arm,
)

CONTROL = "lightgbm_control"
TREATMENT = "lightgbm_treatment_u"
BASELINES = ("seasonal_naive", "ewma_persistence", "global_mean", "parent_category_mean")
PRIMARY_METRIC = "hit_at_3_in_top20"
CONTROL_TOL = 0.0006
EMBARGO_SUMMARY = Path("reports/tables/backtest_embargo_summary.csv")
U3_SELECTED = Path("reports/tables/u3_selected_config.csv")
COMMITTED_FINAL_THREE = Path("reports/tables/top_styles_final_three.csv")
COMMITTED_T2 = Path("reports/tables/top_styles_t2_emerging.csv")
SCRATCH_DIR = Path("models_scratch")
OUT_DIR = Path("reports/tables")
PER_ORIGIN_OUT = OUT_DIR / "u4_per_origin.csv"
PAIRED_OUT = OUT_DIR / "u4_paired_diff.csv"
DECISION_OUT = OUT_DIR / "u4_decision.csv"
SHAP_OUT = OUT_DIR / "u4_shap.csv"
HYPOTHESIS_OUT = OUT_DIR / "u4_hypothesis.csv"
TOP3_OUT = OUT_DIR / "u4_new_top3.csv"

RULE_TEXT = (
    "ADOPT iff the pooled paired treatment-minus-control difference (12 origins, block bootstrap "
    "4 / 2000 resamples / seed 42) has (A) ci_lo > 0 on hit_at_3_in_top20, OR (B) ci_lo > 0 on "
    "BOTH ndcg_at_10 and spearman_rho AND a hit_at_3_in_top20 mean difference >= 0. "
    "Otherwise REJECT."
)

# feature -> expected sign of corr(feature value, SHAP value) if the "broadening sustains growth"
# hypothesis holds. See module docstring HYPOTHESIS.
HYPOTHESIS_FEATURES: dict[str, int] = {
    "cust_age_std_slope_4w": -1,
    "cust_age_std_slope_13w": -1,
    "cust_age_gini_slope_4w": -1,
    "cust_age_gini_slope_13w": -1,
    "cust_geo_distinct_postal_slope_4w": +1,
    "cust_geo_distinct_postal_slope_13w": +1,
    "cust_geo_hhi_slope_4w": -1,
    "cust_geo_hhi_slope_13w": -1,
    "cust_repeat_share_slope_4w": -1,
    "cust_repeat_share_slope_13w": -1,
}
HYPOTHESIS_MIN_ABS_RHO = 0.05
HYPOTHESIS_SUPPORT_MIN = 8
HYPOTHESIS_CONTRADICT_MAX = 2


def decide(paired_vs_control: pl.DataFrame) -> dict[str, object]:
    """Apply the fixed decision rule to the pooled treatment - control paired rows.

    Args:
        paired_vs_control: `paired_diff_table` rows with `method_b == control` (any split; only
            `split == "pooled"` is read), columns `split`, `metric`, `mean_diff`, `diff_ci_low`.

    Returns:
        Dict with `rule_a`, `rule_b` (bool), `decision` (`ADOPT` / `REJECT`) and the numbers used.
    """
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
        "top20_diff": top20["mean_diff"],
        "top20_ci_lo": top20["diff_ci_low"],
        "ndcg_ci_lo": ndcg["diff_ci_low"],
        "spearman_ci_lo": rho["diff_ci_low"],
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


def selected_u3_config() -> LGBMConfig:
    """The grid config U3 selected (row with `selected`), looked up in `HYPERPARAM_GRID`."""
    row = pl.read_csv(U3_SELECTED).filter(pl.col("selected")).row(0, named=True)
    return HYPERPARAM_GRID[int(row["grid_index"])]


def _feature_group(name: str) -> str:
    if name in CUSTOMER_FEATURE_COLS:
        return "customer_u1"
    if name in PRICE_FEATURE_COLS:
        return "price_u2"
    return "existing"


def shap_analysis(
    frame: pl.DataFrame, origins: list[Origin], config: LGBMConfig
) -> tuple[pl.DataFrame, np.ndarray, list[str], pl.DataFrame]:
    """Global SHAP + value-vs-SHAP Spearman for the treatment model on the LAST embargoed fold.

    Returns:
        `(table, shap_values, columns, train_frame)`; `table` has one row per feature sorted by
        mean |SHAP| with `rank`, `group`, `share_of_total`, `shap_spearman` and `n_valid`.
    """
    columns = feature_columns(frame)
    train_weeks = embargoed_train_origin_weeks(origins, len(origins) - 1)
    train_frame = frame.filter(pl.col("origin_week").is_in(train_weeks))
    model = train_lightgbm(train_frame, config, columns)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(SCRATCH_DIR / "u4_treatment_last_fold.txt"))
    x = _to_lgb_matrix(train_frame, columns)
    values = np.asarray(shap.TreeExplainer(model).shap_values(x))
    mean_abs = np.abs(values).mean(axis=0)
    rows = []
    for j, name in enumerate(columns):
        ok = ~np.isnan(x[:, j])
        rho = float("nan")
        if ok.sum() > 2 and np.ptp(x[ok, j]) > 0 and np.ptp(values[ok, j]) > 0:
            rho = float(spearmanr(x[ok, j], values[ok, j]).statistic)
        rows.append(
            {
                "feature": name,
                "group": _feature_group(name),
                "mean_abs_shap": float(mean_abs[j]),
                "shap_spearman": rho,
                "n_valid": int(ok.sum()),
            }
        )
    table = (
        pl.DataFrame(rows)
        .sort("mean_abs_shap", descending=True)
        .with_row_index("rank", offset=1)
        .with_columns((pl.col("mean_abs_shap") / pl.col("mean_abs_shap").sum()).alias("share"))
        .with_columns(
            pl.lit(len(train_weeks)).alias("n_train_origins"),
            pl.lit(train_frame.height).alias("n_train_rows"),
        )
    )
    return table, values, columns, train_frame


def hypothesis_table(
    shap_table: pl.DataFrame, treatment_frame: pl.DataFrame, test_origin_weeks: list
) -> tuple[pl.DataFrame, str]:
    """SHAP direction and model-free realised-growth check for the broadening hypothesis.

    Returns:
        `(table, verdict)`, verdict in `SUPPORTED` / `MIXED` / `CONTRADICTED` (see module doc).
    """
    test = treatment_frame.filter(pl.col("origin_week").is_in(test_origin_weeks)).with_columns(
        (pl.col("y_true") - pl.col("ewma_halflife_13w").log1p()).alias("_growth")
    )
    rows = []
    for name, sign in HYPOTHESIS_FEATURES.items():
        s = shap_table.filter(pl.col("feature") == name).row(0, named=True)
        per_origin = []
        for week in sorted(test_origin_weeks):
            sub = test.filter(pl.col("origin_week") == week).select(name, "_growth").drop_nulls()
            if sub.height > 30 and sub[name].n_unique() > 1:
                per_origin.append(
                    float(spearmanr(sub[name].to_numpy(), sub["_growth"].to_numpy()).statistic)
                )
        mean, lo, hi = block_bootstrap_ci(per_origin) if per_origin else (math.nan,) * 3
        rows.append(
            {
                "feature": name,
                "expected_sign": sign,
                "shap_rank": s["rank"],
                "mean_abs_shap": s["mean_abs_shap"],
                "shap_spearman": s["shap_spearman"],
                "shap_consistent": bool(
                    s["shap_spearman"] == s["shap_spearman"]
                    and abs(s["shap_spearman"]) >= HYPOTHESIS_MIN_ABS_RHO
                    and np.sign(s["shap_spearman"]) == sign
                ),
                "n_test_origins_with_rho": len(per_origin),
                "growth_spearman_mean": mean,
                "growth_spearman_ci_lo": lo,
                "growth_spearman_ci_hi": hi,
                "growth_consistent": bool(mean == mean and np.sign(mean) == sign),
            }
        )
    table = pl.DataFrame(rows)
    n_ok = int(table["shap_consistent"].sum())
    if n_ok >= HYPOTHESIS_SUPPORT_MIN:
        verdict = "SUPPORTED"
    elif n_ok <= HYPOTHESIS_CONTRADICT_MAX:
        verdict = "CONTRADICTED"
    else:
        verdict = "MIXED"
    return table.with_columns(pl.lit(verdict).alias("shap_verdict")), verdict


def _ranking_frame(
    panel: pl.DataFrame,
    model: lgb.LGBMRegressor,
    columns: list[str],
    forecast_frame: pl.DataFrame,
) -> pl.DataFrame:
    """`final_forecast.build_ranking_frame` on an explicit forecast frame (same guards/columns)."""
    origin = final_forecast.FORECAST_ORIGIN
    predicted = final_forecast.predict_intensity(model, forecast_frame, columns)
    guard = final_forecast.build_guard_frame(panel, origin)
    ranking = (
        forecast_frame.select("style_key", *STYLE_KEY_COLS, "price_index_level")
        .with_columns(pl.Series("predicted_intensity", predicted))
        .join(guard, on="style_key", how="left")
    )
    ranking = ranking.with_columns(
        (
            pl.col("guard1_n_active_articles_trailing_mean")
            >= final_forecast.GUARD1_MIN_MEAN_N_ACTIVE_ARTICLES
        ).alias("guard1_pass"),
        pl.when(pl.col("price_index_level").is_not_null())
        .then(pl.col("price_index_level") >= final_forecast.GUARD2_MIN_PRICE_INDEX)
        .otherwise(False)
        .alias("guard2_pass"),
        (pl.col("guard3_n_weeks_active_trailing") >= final_forecast.GUARD3_MIN_WEEKS_ACTIVE).alias(
            "guard3_pass"
        ),
    )
    return ranking.sort("predicted_intensity", descending=True).with_row_index(
        "rank_unguarded", offset=1
    )


def forecast_final_three(
    panel: pl.DataFrame, config: LGBMConfig, with_new_features: bool
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Wide-origin final model -> ranking -> T2 emerging -> final three (shipped selection rules).

    Returns:
        `(final_three, t2_emerging, ranking)`.
    """
    weeks = final_forecast.final_training_origin_weeks(panel)
    frame = build_model_frame(panel, weeks)
    forecast = build_features(panel, [final_forecast.FORECAST_ORIGIN])
    if with_new_features:
        frame = build_treatment_frame(frame, panel)
        forecast = build_treatment_frame(forecast, panel)
    columns = feature_columns(frame)
    model = train_lightgbm(frame, config, columns)
    ranking = _ranking_frame(panel, model, columns, forecast)
    t2 = select_t2_emerging(panel, ranking)
    intimate = intimate_product_types(pl.read_csv(ARTICLES_PATH))
    final, _log = reselect(t2, intimate)
    return final, t2, ranking


def main() -> None:
    """Control gate -> treatment arm -> paired tables -> decision -> SHAP -> new top-3."""
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

    config = selected_u3_config()
    print(f"treatment config (U3 selected): {config}")
    treatment_frame = build_treatment_frame(control_frame, panel)
    assert treatment_frame.height == control_frame.height
    assert treatment_frame.select(control_frame.columns).equals(control_frame)
    n_new = len(feature_columns(treatment_frame)) - len(feature_columns(control_frame))
    print(f"treatment adds {n_new} features (customer + price)")
    treatment = run_arm(treatment_frame, all_origins, config, TREATMENT)

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

    shap_table, _values, _cols, _train = shap_analysis(treatment_frame, all_origins, config)
    shap_table.write_csv(SHAP_OUT)
    hyp, verdict = hypothesis_table(shap_table, treatment_frame, test_weeks)
    hyp.write_csv(HYPOTHESIS_OUT)

    current = pl.read_csv(COMMITTED_FINAL_THREE)["style_key"].to_list()
    parity, _t2c, _rk = forecast_final_three(panel, FINAL_MODEL_CONFIG, with_new_features=False)
    parity_keys = parity["style_key"].to_list()
    print(f"parity (control pipeline reproduces committed final three): {parity_keys == current}")
    new_final, new_t2, _ranking = forecast_final_three(panel, config, with_new_features=True)
    new_keys = new_final["style_key"].to_list()
    committed_t2 = pl.read_csv(COMMITTED_T2)["style_key"].to_list()
    top3 = new_final.select(
        pl.lit(TREATMENT).alias("model"),
        pl.col("source_table"),
        "rank_in_source_table",
        "style_key",
        "predicted_intensity",
        "growth_ratio",
        pl.col("style_key").is_in(current).alias("in_current_top3"),
    ).with_columns(
        pl.lit(parity_keys == current).alias("control_pipeline_reproduces_committed"),
        pl.lit(", ".join(k.split(" || ")[1] + "/" + k.split(" || ")[3] for k in current)).alias(
            "current_top3"
        ),
    )
    top3.write_csv(TOP3_OUT)

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
        print(
            shap_table.head(15).select("rank", "feature", "group", "mean_abs_shap", "shap_spearman")
        )
        print(
            shap_table.filter(pl.col("group") != "existing").select(
                "rank", "feature", "group", "mean_abs_shap", "shap_spearman"
            )
        )
        print(shap_table.group_by("group").agg(pl.col("share").sum()))
        print(hyp.select(pl.exclude("shap_verdict")))
        print(f"SHAP hypothesis verdict: {verdict}")
        print(top3)
    print(f"new top-3 == current: {new_keys == current} (as sets: {set(new_keys) == set(current)})")
    print(
        "new T2 (rank 1..10) present in committed T2: "
        f"{[k in committed_t2 for k in new_t2['style_key'].to_list()]}"
    )
    print(f"DECISION: {decision['decision'][0]}")


if __name__ == "__main__":
    main()
