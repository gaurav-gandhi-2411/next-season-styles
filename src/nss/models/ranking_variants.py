"""Task G3: ranking-improvement variants tried against the G2 finding that `lambdarank`
UNDERPERFORMS the existing L2 model on both headline Hit@3-in-topN metrics (see
`nss.models.lambdarank_vs_l2`'s paired, block-bootstrapped comparison). Three variants, in the
order the task specifies, each scored through the SAME `nss.models.metrics.score_predictions` /
`nss.models.backtest.block_bootstrap_ci` machinery over the SAME 12 walk-forward test origins
(`nss.models.backtest_v2.identify_lightgbm_origins`) as L2/lambdarank, so every number here is
directly, honestly comparable to `reports/tables/lambdarank_vs_l2_comparison.csv`.

FIXED TREE HYPERPARAMETERS (JUDGMENT CALL, same precedent as `nss.models.lambdarank_model`): every
variant below reuses `nss.models.final_forecast.FINAL_MODEL_CONFIG` (the L2 model's own winning
`num_leaves`/`learning_rate`/`n_estimators`/`min_child_samples`) rather than re-running
`select_hyperparameters` -- this isolates each comparison to "same tree-building budget, different
weighting/staging/ensembling", which is the actual question this task asks, and keeps the search
bounded (no re-tuning against the walk-forward test set, which would be leakage).

VARIANT (a) TOP-HEAVY SAMPLE WEIGHTING: plain L2 objective (`train_lightgbm_weighted` below is
`nss.models.lightgbm_model.train_lightgbm` plus a `sample_weight` argument -- nothing else differs),
but each TRAINING row is weighted by `1 + TOP_HEAVY_WEIGHT_SCALE * rank_percentile`, where
`rank_percentile` is that row's `y_true` rank (descending, 1=highest) PERCENTILE within its own
`origin_week` (1.0 for that origin's single highest-`y_true` style, 0.0 for its single lowest) --
computed per-origin (not globally) so every origin's own population sets its own scale regardless of
how many eval-eligible styles it has. `TOP_HEAVY_WEIGHT_SCALE=9.0` means the single top style in an
origin is trained on with 10x the weight of the single bottom style -- a deliberately aggressive
tilt toward the head, since the headline metrics (Hit@3-in-top20/top10) only ever look at the head.

VARIANT (b) TWO-STAGE RANKING: stage 1 is the plain L2 model, trained/predicted exactly as
`nss.models.lightgbm_model` does. Stage 2 is a SEPARATE L2-objective model trained ONLY on the
TRAINING set's true-top-`HEAD_SIZE` rows per origin (by realised `y_true`, `HEAD_SIZE=50` -- the
task's own suggested cutoff), i.e. a specialist that never sees tail rows at all. At TEST time, true
ranks are unknown, so the "head subset re-ranked by stage 2" is necessarily STAGE 1's OWN predicted
top-`HEAD_SIZE` styles (the closest available proxy to "true top-50" at prediction time) -- stage 2
re-scores exactly those `HEAD_SIZE` styles, and the final predicted ranking places all `HEAD_SIZE`
stage-2-re-ranked styles above every non-head style (stage-2 score + a constant offset that exceeds
every possible stage-1 score, then ties within/outside the head broken by each stage's own score).
`HEAD_SIZE=50` safely covers `HIT_TOP_N_PRIMARY=20` with margin, so every headline-metric-relevant
position is stage-2-re-ranked, not stage-1-inherited.

VARIANT (c) ENSEMBLE (RANK-AVERAGE, documented choice -- NOT normalized-score averaging): for each
test origin, both the plain L2 model and a freshly-retrained `lambdarank` model (same
`select_truncation_level`-chosen truncation, retrained here rather than reusing the checked-in
per-origin CSV because this module needs RAW per-style predictions, which the aggregated backtest
CSVs don't carry) predict the full eval set. Each style gets a `rank` (1=best) under EACH model's
OWN descending-sorted predictions; the ensemble score is `-(rank_l2 + rank_lambdarank) / 2` (negated
so "higher ensemble score = better", matching every other method's convention that
`score_predictions` sorts descending by). Rank-averaging (not raw-score averaging) is chosen because
L2's raw score and lambdarank's raw score live on genuinely different, uncalibrated scales (see
`nss.models.lambdarank_vs_l2`'s WMAPE CAVEAT) -- averaging RANKS sidesteps that scale mismatch
entirely, at the cost of discarding each model's own confidence/margin information.

WMAPE CAVEAT (inherited, same as `nss.models.lambdarank_vs_l2`): the two-stage pipeline's final
predicted "scores" (stage-2-score-plus-offset for the head, stage-1-score for the tail) and the
ensemble's rank-average are BOTH deliberately non-calibrated composite scales -- WMAPE and
`spearman_rho` are still computed (dropping a mandatory `METRIC_KEYS` entry would be dishonest) but
must not be read as "X is worse at forecasting a calibrated value" the way the plain L2 model's
WMAPE can be.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from nss.models.backtest import (
    WMAPE_WEIGHT_COL,
    Origin,
    block_bootstrap_ci,
    generate_origin_schedule,
)
from nss.models.backtest_v2 import identify_lightgbm_origins
from nss.models.final_forecast import FINAL_MODEL_CONFIG
from nss.models.lambdarank_model import (
    LAMBDARANK_CONFIG,
    add_relevance_grades,
    predict_lambdarank,
    select_truncation_level,
    train_lambdarank,
)
from nss.models.lightgbm_model import (
    INITIAL_POOL_SIZE,
    LGBM_DETERMINISM_PARAMS,
    LGBM_OBJECTIVE,
    RANDOM_SEED,
    LGBMConfig,
    _categorical_indices,
    _expanding_train_origin_weeks,
    _to_lgb_matrix,
    build_model_frame,
    feature_columns,
    predict_lightgbm,
    run_lightgbm_walk_forward,
    train_lightgbm,
)
from nss.models.metrics import METRIC_KEYS, score_predictions

VARIANT_A_METHOD = "l2_top_heavy_weighted"
VARIANT_B_METHOD = "l2_two_stage"
VARIANT_C_METHOD = "ensemble_rank_avg"

# See module docstring VARIANT (a).
TOP_HEAVY_WEIGHT_SCALE = 9.0

# See module docstring VARIANT (b).
HEAD_SIZE = 50
# Exceeds any plausible stage-1/stage-2 log1p-intensity score by a wide margin -- guarantees every
# head-subset style outranks every non-head style in the final constructed ranking, regardless of
# the two stages' own score scales.
HEAD_SCORE_OFFSET = 1000.0

DEFAULT_PANEL_PATH = Path("data/processed/style_week_panel.parquet")
DEFAULT_OUT_PATH = Path("reports/tables/ranking_variants.csv")


def train_lightgbm_weighted(
    train_frame: pl.DataFrame,
    config: LGBMConfig,
    columns: list[str],
    sample_weight: np.ndarray,
) -> lgb.LGBMRegressor:
    """`nss.models.lightgbm_model.train_lightgbm`, plus a `sample_weight` argument. Identical in
    every other respect (same objective, same determinism params, same categorical handling)."""
    X = _to_lgb_matrix(train_frame, columns)
    y = train_frame["y_true"].to_numpy().astype(np.float64)
    cat_indices = _categorical_indices(train_frame, columns)

    model = lgb.LGBMRegressor(
        objective=LGBM_OBJECTIVE,
        random_state=RANDOM_SEED,
        verbosity=-1,
        num_leaves=int(config["num_leaves"]),
        learning_rate=float(config["learning_rate"]),
        n_estimators=int(config["n_estimators"]),
        min_child_samples=int(config["min_child_samples"]),
        **LGBM_DETERMINISM_PARAMS,
    )
    model.fit(
        X, y, sample_weight=sample_weight, categorical_feature=cat_indices, feature_name=columns
    )
    return model


def rank_percentile_weights(train_frame: pl.DataFrame) -> np.ndarray:
    """Per-row training weight for VARIANT (a): `1 + TOP_HEAVY_WEIGHT_SCALE * rank_percentile`.
    See module docstring VARIANT (a) for the exact definition of `rank_percentile`.

    Args:
        train_frame: Any `build_model_frame`-shaped frame with `y_true` and `origin_week`.

    Returns:
        One weight per row, same order as `train_frame`.
    """
    ranked = train_frame.with_columns(
        pl.col("y_true").rank(method="ordinal", descending=True).over("origin_week").alias("_rank"),
        pl.len().over("origin_week").alias("_n"),
    )
    n = ranked["_n"].to_numpy().astype(np.float64)
    rank = ranked["_rank"].to_numpy().astype(np.float64)
    denom = np.maximum(n - 1.0, 1.0)  # avoid /0 for a degenerate 1-row origin
    percentile = (n - rank) / denom
    return 1.0 + TOP_HEAVY_WEIGHT_SCALE * percentile


def _score_origin(
    test_frame: pl.DataFrame, pred_col: str, method: str, origin: Origin
) -> dict[str, object]:
    """Score one origin's `pred_col` predictions via `score_predictions`, in the same per-origin row
    schema every other method in this project uses (`origin_week`, `method`, `has_52w_lag`,
    `is_covid`, `n_eval_set`, `n_eval`, plus every `METRIC_KEYS` metric)."""
    metrics = score_predictions(
        test_frame["y_true"].to_numpy(),
        test_frame[pred_col].to_numpy(),
        test_frame[WMAPE_WEIGHT_COL].to_numpy(),
    )
    n_eval = metrics.pop("n_eval")
    return {
        "origin_week": origin.origin_week,
        "method": method,
        "has_52w_lag": origin.has_52w_lag,
        "is_covid": origin.is_covid,
        "n_eval_set": test_frame.height,
        "n_eval": int(n_eval),
        **metrics,
    }


def run_variant_a_top_heavy(
    panel: pl.DataFrame,
    origins: list[Origin],
    config: LGBMConfig = FINAL_MODEL_CONFIG,
    pool_size: int = INITIAL_POOL_SIZE,
) -> pl.DataFrame:
    """VARIANT (a): top-heavy sample-weighted L2, walk-forward evaluated. See module docstring."""
    origin_weeks = [o.origin_week for o in origins]
    model_frame = build_model_frame(panel, origin_weeks)
    columns = feature_columns(model_frame)

    rows: list[dict[str, object]] = []
    for test_index in range(pool_size, len(origins)):
        test_origin = origins[test_index]
        train_weeks = _expanding_train_origin_weeks(origins, test_index)
        train_frame = model_frame.filter(pl.col("origin_week").is_in(train_weeks))
        test_frame = model_frame.filter(pl.col("origin_week") == test_origin.origin_week)

        weights = rank_percentile_weights(train_frame)
        model = train_lightgbm_weighted(train_frame, config, columns, weights)
        preds = predict_lightgbm(model, test_frame, columns)
        scored = test_frame.with_columns(pl.Series("_pred", preds))
        rows.append(_score_origin(scored, "_pred", VARIANT_A_METHOD, test_origin))
    return pl.DataFrame(rows)


def run_variant_b_two_stage(
    panel: pl.DataFrame,
    origins: list[Origin],
    config: LGBMConfig = FINAL_MODEL_CONFIG,
    pool_size: int = INITIAL_POOL_SIZE,
    head_size: int = HEAD_SIZE,
) -> pl.DataFrame:
    """VARIANT (b): two-stage ranking (plain-L2 stage 1, head-only-specialist stage 2). See module
    docstring VARIANT (b) for the exact test-time re-ranking construction."""
    origin_weeks = [o.origin_week for o in origins]
    model_frame = build_model_frame(panel, origin_weeks)
    columns = feature_columns(model_frame)

    rows: list[dict[str, object]] = []
    for test_index in range(pool_size, len(origins)):
        test_origin = origins[test_index]
        train_weeks = _expanding_train_origin_weeks(origins, test_index)
        train_frame = model_frame.filter(pl.col("origin_week").is_in(train_weeks))
        test_frame = model_frame.filter(pl.col("origin_week") == test_origin.origin_week)

        # Stage 1: plain L2, identical to nss.models.lightgbm_model.
        stage1_model = train_lightgbm(train_frame, config, columns)
        stage1_train_preds = predict_lightgbm(stage1_model, train_frame, columns)
        stage1_test_preds = predict_lightgbm(stage1_model, test_frame, columns)

        # Stage 2 training set: TRUE top-head_size rows per TRAINING origin (by realised y_true).
        head_train = train_frame.with_columns(
            pl.col("y_true")
            .rank(method="ordinal", descending=True)
            .over("origin_week")
            .alias("_true_rank")
        ).filter(pl.col("_true_rank") <= head_size)
        stage2_model = train_lightgbm(head_train, config, columns)

        # Test-time head subset: STAGE 1's OWN predicted top-head_size (true ranks unknown at test
        # time -- see module docstring VARIANT (b)).
        test_with_stage1 = test_frame.with_columns(pl.Series("_stage1_pred", stage1_test_preds))
        head_test = test_with_stage1.sort("_stage1_pred", descending=True).head(head_size)
        tail_test = test_with_stage1.sort("_stage1_pred", descending=True).slice(head_size)

        stage2_head_preds = predict_lightgbm(stage2_model, head_test, columns)
        head_final = head_test.with_columns(
            (pl.Series("_stage2_pred", stage2_head_preds) + HEAD_SCORE_OFFSET).alias("_final_pred")
        )
        tail_final = tail_test.with_columns(pl.col("_stage1_pred").alias("_final_pred"))

        combined = pl.concat(
            [
                head_final.select(*test_frame.columns, "_final_pred"),
                tail_final.select(*test_frame.columns, "_final_pred"),
            ],
            how="vertical",
        )
        del stage1_train_preds  # kept only to document stage-1's train-set predictions aren't used
        rows.append(_score_origin(combined, "_final_pred", VARIANT_B_METHOD, test_origin))
    return pl.DataFrame(rows)


def run_variant_c_ensemble(
    panel: pl.DataFrame,
    origins: list[Origin],
    lgbm_config: LGBMConfig = FINAL_MODEL_CONFIG,
    lambdarank_config: LGBMConfig = LAMBDARANK_CONFIG,
    pool_size: int = INITIAL_POOL_SIZE,
) -> pl.DataFrame:
    """VARIANT (c): rank-average ensemble of a freshly-retrained `lambdarank` and the plain L2
    model. See module docstring VARIANT (c) for why RAW predictions must be retrained here (the
    checked-in per-origin CSVs only carry aggregated metrics, not per-style scores)."""
    origin_weeks = [o.origin_week for o in origins]
    l2_frame = build_model_frame(panel, origin_weeks)
    columns = feature_columns(l2_frame)
    lr_frame = add_relevance_grades(l2_frame)  # same columns; relevance_grade appended

    pool_origins = origins[:pool_size]
    truncation_search = select_truncation_level(lr_frame, pool_origins, columns, lambdarank_config)
    truncation_level = truncation_search.best_truncation_level

    rows: list[dict[str, object]] = []
    for test_index in range(pool_size, len(origins)):
        test_origin = origins[test_index]
        train_weeks = _expanding_train_origin_weeks(origins, test_index)

        l2_train = l2_frame.filter(pl.col("origin_week").is_in(train_weeks))
        l2_test = l2_frame.filter(pl.col("origin_week") == test_origin.origin_week)
        l2_model = train_lightgbm(l2_train, lgbm_config, columns)
        l2_preds = predict_lightgbm(l2_model, l2_test, columns)

        lr_train = lr_frame.filter(pl.col("origin_week").is_in(train_weeks))
        lr_test = lr_frame.filter(pl.col("origin_week") == test_origin.origin_week)
        lr_model = train_lambdarank(lr_train, lambdarank_config, truncation_level, columns)
        lr_preds = predict_lambdarank(lr_model, lr_test, columns)

        # l2_test and lr_test are the SAME (style_key, origin_week) rows in the SAME row order
        # (both filtered from build_model_frame outputs sharing the same style_key/origin_week
        # identity and join order -- add_relevance_grades adds a column, never reorders rows).
        combined = (
            l2_test.with_columns(pl.Series("_l2_pred", l2_preds), pl.Series("_lr_pred", lr_preds))
            .with_columns(
                pl.col("_l2_pred")
                .rank(method="ordinal", descending=True)
                .cast(pl.Float64)
                .alias("_l2_rank"),
                pl.col("_lr_pred")
                .rank(method="ordinal", descending=True)
                .cast(pl.Float64)
                .alias("_lr_rank"),
            )
            .with_columns(
                (-(pl.col("_l2_rank") + pl.col("_lr_rank")) / 2.0).alias("_ensemble_pred")
            )
        )
        rows.append(_score_origin(combined, "_ensemble_pred", VARIANT_C_METHOD, test_origin))
    return pl.DataFrame(rows)


def build_g3_comparison(panel: pl.DataFrame) -> pl.DataFrame:
    """Assemble the full G3 comparison table: all 3 variants, each scored AND paired-diffed
    against a FRESHLY-recomputed L2 model over the SAME 12 walk-forward test origins.

    L2 is recomputed here (via `run_lightgbm_walk_forward`) rather than read back from the
    checked-in `reports/tables/backtest_per_origin_lightgbm.csv` -- that checked-in file predates
    this project's Hit@3-in-topN headline metrics (`METRIC_KEYS`'s v2 addition) and is missing
    those columns entirely, discovered directly while building this comparison; recomputing is the
    only way to get an honest paired L2 series with every `METRIC_KEYS` metric present.

    OUTPUT SCHEMA: identical to `nss.models.lambdarank_vs_l2.build_g2_comparison`'s (one row per
    `(method, split, metric)`, `method` in the 3 `VARIANT_*_METHOD` constants, `diff_vs_lightgbm_*`
    columns are the paired, block-bootstrapped `variant - lightgbm` per-origin difference).

    Returns:
        One row per `(method, split, metric)`.
    """
    walk_forward_origins = identify_lightgbm_origins(panel)
    all_origins = generate_origin_schedule(panel)
    pool_size = len(all_origins) - len(walk_forward_origins)

    l2_per_origin, _config, _grid = run_lightgbm_walk_forward(
        panel, all_origins, pool_size=pool_size
    )
    l2_by_origin = l2_per_origin.sort("origin_week")

    variant_per_origin = {
        VARIANT_A_METHOD: run_variant_a_top_heavy(panel, all_origins, pool_size=pool_size),
        VARIANT_B_METHOD: run_variant_b_two_stage(panel, all_origins, pool_size=pool_size),
        VARIANT_C_METHOD: run_variant_c_ensemble(panel, all_origins, pool_size=pool_size),
    }

    rows: list[dict[str, object]] = []
    for method, per_origin in variant_per_origin.items():
        method_df = per_origin.sort("origin_week")
        joined_for_diff = method_df.join(l2_by_origin, on="origin_week", how="inner", suffix="_l2")
        splits = (
            ("pooled", method_df, joined_for_diff),
            (
                "covid",
                method_df.filter(pl.col("is_covid")),
                joined_for_diff.filter(pl.col("is_covid")),
            ),
            (
                "non_covid",
                method_df.filter(~pl.col("is_covid")),
                joined_for_diff.filter(~pl.col("is_covid")),
            ),
        )
        for split_name, split_df, diff_split_df in splits:
            for metric in METRIC_KEYS:
                mean, ci_low, ci_high = block_bootstrap_ci(split_df[metric].to_list())
                diff_series = (diff_split_df[metric] - diff_split_df[f"{metric}_l2"]).to_list()
                d_mean, d_lo, d_hi = block_bootstrap_ci(diff_series)
                rows.append(
                    {
                        "method": method,
                        "split": split_name,
                        "metric": metric,
                        "n_origins": split_df.height,
                        "mean": mean,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "diff_vs_lightgbm_mean": d_mean,
                        "diff_vs_lightgbm_ci_low": d_lo,
                        "diff_vs_lightgbm_ci_high": d_hi,
                    }
                )
    return pl.DataFrame(rows)


def main() -> None:
    """CLI entry point: build the full G3 comparison table, write it to `DEFAULT_OUT_PATH`."""
    panel = pl.read_parquet(DEFAULT_PANEL_PATH)
    comparison = build_g3_comparison(panel)

    DEFAULT_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.write_csv(DEFAULT_OUT_PATH)
    print(f"Wrote {DEFAULT_OUT_PATH} ({comparison.height} rows)")

    for method in (VARIANT_A_METHOD, VARIANT_B_METHOD, VARIANT_C_METHOD):
        pooled = comparison.filter((pl.col("method") == method) & (pl.col("split") == "pooled"))
        for row in pooled.filter(pl.col("metric").is_in(METRIC_KEYS[:3])).iter_rows(named=True):
            print(
                f"  {method} {row['metric']}: mean={row['mean']:.4f} "
                f"diff_vs_lightgbm={row['diff_vs_lightgbm_mean']:.4f} "
                f"[{row['diff_vs_lightgbm_ci_low']:.4f}, {row['diff_vs_lightgbm_ci_high']:.4f}]"
            )


if __name__ == "__main__":
    main()
