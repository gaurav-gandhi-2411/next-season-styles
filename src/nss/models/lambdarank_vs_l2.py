"""Task G2 output artifact: the full before/after comparison table, `lambdarank` vs the existing
L2 `lightgbm` model, in the same broader context `backtest_v2.py` already established for L2 (the
random floor, the 4 causal baselines) plus the G1 causal persistence oracle -- all scored on the
SAME 12 walk-forward test origins, via the EXISTING harness (`nss.models.backtest`/`nss.models.
metrics`), nothing new built.

WHY A SEPARATE MODULE (not folded into `lambdarank_model.py`): mirrors this project's own existing
split between `lightgbm_model.py` (trains/walk-forward-evaluates ONE method) and `backtest_v2.py`
(assembles that method alongside every other method for an honest, paired, like-for-like
comparison) -- this module is `backtest_v2.py`'s counterpart for the `lambdarank` vs. `lightgbm`
(L2) question specifically, reusing every piece of that existing machinery rather than
re-implementing any of it.

WMAPE CAVEAT (report honestly, do not hide): `lgb.LGBMRanker.predict`'s raw output is an
UNCALIBRATED ranking score (there is no constraint tying its scale to `log1p(mean(units_per_
active_article))`, unlike the L2 model's regression output, which IS that quantity by
construction) -- WMAPE (`sum(weight * |actual - pred|) / sum(weight * |actual|)`) compares
`y_pred` to `y_true` on an absolute scale, so `lambdarank`'s WMAPE number is expected to look far
worse than the L2 model's REGARDLESS of ranking quality, and is not a fair like-for-like
comparison. It is still reported here (dropping a mandatory `METRIC_KEYS` entry would itself be
dishonest), but must never be read as "lambdarank is 30x worse at forecasting" -- it isn't
forecasting a calibrated value at all. Same caveat, to a lesser degree, applies to `spearman_rho`
(a FULL-eval-set rank correlation): `lambdarank_truncation_level` deliberately concentrates the
loss's gradient on the head of the ranking (see `nss.models.lambdarank_model` module docstring
TRUNCATION LEVEL SELECTION), so a WORSE full-distribution Spearman rho alongside a BETTER head
metric (Precision@3) is the expected signature of that trade-off working as intended, not a
contradiction.

OUTPUT SCHEMA (`reports/tables/lambdarank_vs_l2_comparison.csv`): one row per `(method, split, metric)` --
`method` in `{lambdarank, lightgbm, seasonal_naive, ewma_persistence, parent_category_mean,
global_mean, random_floor, persistence_oracle}`, `split` in `{pooled, covid, non_covid}`, `metric`
in `nss.models.metrics.METRIC_KEYS`. Columns: `n_origins`, `mean`, `ci_low`, `ci_high` (that
method/split/metric's own block-bootstrapped value), plus `diff_vs_lightgbm_mean`/`diff_vs_lightgbm
_ci_low`/`diff_vs_lightgbm_ci_high` -- POPULATED ONLY for `method == "lambdarank"` rows (the
paired, block-bootstrapped `lambdarank - lightgbm` per-origin difference, reusing `nss.models.
backtest.block_bootstrap_ci` on the diff series directly, the identical statistical machinery
`backtest_v2.paired_diff_table` uses) -- `null` for every other method's rows (a paired diff vs.
itself, or vs. a method this task was not asked to diff, is not computed rather than filled with a
value that would misleadingly suggest it was).
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import polars as pl

from nss.models.backtest import (
    BASELINE_METHODS,
    Origin,
    block_bootstrap_ci,
    build_predictions_frame,
    generate_origin_schedule,
    run_backtest,
)
from nss.models.backtest_v2 import identify_lightgbm_origins
from nss.models.head_ranking_diagnostics import LAG_WEEKS, score_persistence_oracle
from nss.models.lambdarank_model import DEFAULT_PER_ORIGIN_OUT_PATH as LAMBDARANK_PER_ORIGIN_PATH
from nss.models.lambdarank_model import METHOD_NAME as LAMBDARANK_METHOD
from nss.models.lightgbm_model import METHOD_NAME as LIGHTGBM_METHOD
from nss.models.lightgbm_model import run_lightgbm_walk_forward
from nss.models.metrics import METRIC_KEYS
from nss.models.random_floor import (
    RANDOM_FLOOR_METHOD,
    aggregate_random_floor_over_seeds,
    score_random_floor_per_origin,
)

ORACLE_METHOD = "persistence_oracle"

# `LAG_WEEKS` (from `nss.models.head_ranking_diagnostics`) is an int number of weeks; the oracle's
# own earlier-origin-week computation needs a `timedelta`, matching that module's own convention.
LAG_WEEKS_TIMEDELTA = timedelta(weeks=LAG_WEEKS)

DEFAULT_PANEL_PATH = Path("data/processed/style_week_panel.parquet")
DEFAULT_OUT_PATH = Path("reports/tables/lambdarank_vs_l2_comparison.csv")


def _oracle_per_origin(
    panel: pl.DataFrame, origins: list[Origin], eval_frame: pl.DataFrame
) -> pl.DataFrame:
    """`score_persistence_oracle`'s output, reshaped to the shared per-origin schema (`method`,
    `has_52w_lag`, `is_covid` columns added) so it can be concatenated with every other method's
    per-origin table below."""
    oracle = score_persistence_oracle(panel, origins, eval_frame, lag_weeks=LAG_WEEKS)
    tagged = oracle.with_columns(
        pl.lit(ORACLE_METHOD).alias("method"),
        pl.Series("has_52w_lag", [o.has_52w_lag for o in origins]),
        pl.Series("is_covid", [o.is_covid for o in origins]),
    )
    return tagged.select(
        "origin_week", "method", "has_52w_lag", "is_covid", "n_eval_set", "n_eval", *METRIC_KEYS
    )


def _load_lambdarank_per_origin(path: Path = LAMBDARANK_PER_ORIGIN_PATH) -> pl.DataFrame:
    """The checked-in `lambdarank` per-origin CSV (`nss.models.lambdarank_model.main`'s own
    output) -- read back rather than re-run, since `run_lambdarank_walk_forward` is verified
    bit-identical across separate process runs (`tests/test_lambdarank_determinism_cross_process.
    py`), so re-running it here would reproduce the exact same numbers at needless extra cost."""
    return pl.read_csv(path, try_parse_dates=True).select(
        "origin_week", "method", "has_52w_lag", "is_covid", "n_eval_set", "n_eval", *METRIC_KEYS
    )


def build_g2_comparison(panel: pl.DataFrame) -> pl.DataFrame:
    """Assemble the full before/after comparison table. See module docstring OUTPUT SCHEMA.

    Args:
        panel: The dense style-week panel.

    Returns:
        The full comparison table, one row per `(method, split, metric)`.
    """
    origins = identify_lightgbm_origins(panel)
    origin_weeks = [o.origin_week for o in origins]

    baseline_per_origin = run_backtest(panel, origins)
    lightgbm_per_origin, _config, _grid = run_lightgbm_walk_forward(
        panel, generate_origin_schedule(panel)
    )
    lightgbm_per_origin = lightgbm_per_origin.filter(pl.col("origin_week").is_in(origin_weeks))
    lambdarank_per_origin = _load_lambdarank_per_origin()

    predictions = build_predictions_frame(panel, origin_weeks)
    rf_per_seed = score_random_floor_per_origin(predictions, origins)
    rf_per_origin = aggregate_random_floor_over_seeds(rf_per_seed)

    earlier_weeks = [ow - LAG_WEEKS_TIMEDELTA for ow in origin_weeks]
    all_weeks = sorted(set(origin_weeks) | set(earlier_weeks))
    eval_all = build_predictions_frame(panel, all_weeks).select(
        "style_key", "origin_week", "y_true", "weight"
    )
    oracle_per_origin = _oracle_per_origin(panel, origins, eval_all)

    combined = pl.concat(
        [
            baseline_per_origin,
            lightgbm_per_origin,
            lambdarank_per_origin,
            rf_per_origin,
            oracle_per_origin,
        ],
        how="vertical",
    )

    all_methods = (
        LAMBDARANK_METHOD,
        LIGHTGBM_METHOD,
        *BASELINE_METHODS,
        RANDOM_FLOOR_METHOD,
        ORACLE_METHOD,
    )

    # Per-origin diff series (lambdarank - lightgbm), keyed by origin_week, for every metric -- used
    # below to populate diff_vs_lightgbm_* only on lambdarank rows. See module docstring OUTPUT
    # SCHEMA.
    lgbm_by_origin = lightgbm_per_origin.sort("origin_week")
    lambdarank_by_origin = lambdarank_per_origin.sort("origin_week")
    joined_for_diff = lambdarank_by_origin.join(
        lgbm_by_origin, on="origin_week", how="inner", suffix="_lightgbm"
    )

    rows: list[dict[str, object]] = []
    for method in all_methods:
        method_df = combined.filter(pl.col("method") == method).sort("origin_week")
        splits = (
            ("pooled", method_df),
            ("covid", method_df.filter(pl.col("is_covid"))),
            ("non_covid", method_df.filter(~pl.col("is_covid"))),
        )
        for split_name, split_df in splits:
            if split_name == "pooled":
                diff_split_df = joined_for_diff
            elif split_name == "covid":
                diff_split_df = joined_for_diff.filter(pl.col("is_covid"))
            else:
                diff_split_df = joined_for_diff.filter(~pl.col("is_covid"))

            for metric in METRIC_KEYS:
                mean, ci_low, ci_high = block_bootstrap_ci(split_df[metric].to_list())
                row: dict[str, object] = {
                    "method": method,
                    "split": split_name,
                    "metric": metric,
                    "n_origins": split_df.height,
                    "mean": mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "diff_vs_lightgbm_mean": None,
                    "diff_vs_lightgbm_ci_low": None,
                    "diff_vs_lightgbm_ci_high": None,
                }
                if method == LAMBDARANK_METHOD:
                    diff_series = (
                        diff_split_df[metric] - diff_split_df[f"{metric}_lightgbm"]
                    ).to_list()
                    d_mean, d_lo, d_hi = block_bootstrap_ci(diff_series)
                    row["diff_vs_lightgbm_mean"] = d_mean
                    row["diff_vs_lightgbm_ci_low"] = d_lo
                    row["diff_vs_lightgbm_ci_high"] = d_hi
                rows.append(row)
    return pl.DataFrame(rows)


def main() -> None:
    """CLI entry point: build the G2 comparison table, write it to `DEFAULT_OUT_PATH`."""
    panel = pl.read_parquet(DEFAULT_PANEL_PATH)
    comparison = build_g2_comparison(panel)

    DEFAULT_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.write_csv(DEFAULT_OUT_PATH)
    print(f"Wrote {DEFAULT_OUT_PATH} ({comparison.height} rows)")

    pooled = comparison.filter((pl.col("split") == "pooled") & (pl.col("method") == "lambdarank"))
    for row in pooled.iter_rows(named=True):
        print(
            f"  {row['metric']}: lambdarank={row['mean']:.4f} "
            f"diff_vs_lightgbm={row['diff_vs_lightgbm_mean']:.4f} "
            f"[{row['diff_vs_lightgbm_ci_low']:.4f}, {row['diff_vs_lightgbm_ci_high']:.4f}]"
        )


if __name__ == "__main__":
    main()
