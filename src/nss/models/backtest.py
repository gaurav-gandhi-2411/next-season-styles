"""Rolling-origin backtest harness: origin schedule, 4 causal baselines, and scoring/aggregation.

ORIGIN SCHEDULE (`generate_origin_schedule`): rolling origins every `STEP_WEEKS` (4) weeks,
starting `BURN_IN_WEEKS` (13, matching `nss.features.targets.HORIZON_WEEKS`) weeks after the
panel's earliest `week_start`, stopping at the last origin whose full forward target window still
fits inside the panel's latest `week_start`. Each origin is tagged `has_52w_lag` (is the origin
itself >= 52 weeks after the panel's start -- a structural/data-availability fact about the origin,
NOT a claim that every style has 52 weeks of its OWN history; a young style can still have a null
`lag_52` at an origin where `has_52w_lag=True`) and `is_covid` (does the origin's 13-week forecast
horizon, `origin+1..origin+13` weeks, overlap `COVID_WINDOW_START..COVID_WINDOW_END` at all).

EVAL SET (`build_predictions_frame`): per origin, the eval population is every `style_key` that (a)
has a feature row from `build_features` at that origin (the style already existed as of that
origin -- see `nss.features.model_features`), AND (b) has a non-null forward target from
`compute_forward_target` at that origin (the style survives long enough to have a full 13-week
forward window to score against -- see `nss.features.targets`). This shrinks for later origins as
fewer styles have a full future window still inside the panel, and varies with which styles are
active -- both are causal, expected, and NOT a bug.

SCALE: every prediction and the target itself are on the SAME `log1p(mean(units_per_active_article
over the relevant 13-week window))` scale as `compute_forward_target`'s own target definition. This
is deliberate and load-bearing (see module docstring of `nss.models.metrics` for why WMAPE/NDCG are
meaningless across mismatched scales) -- see each `_predict_*` function's docstring for exactly how
it gets there for that baseline.

THE 4 BASELINES, each producing one `y_pred_<method>` column in `build_predictions_frame`'s output:
- `seasonal_naive`: the style's own realized `units_per_active_article` over the SAME 13 calendar
  weeks one year (52 weeks) earlier, aggregated identically to the target
  (`log1p(mean(...))`) -- implemented by calling `compute_forward_target(panel, origin_week - 52
  weeks)` directly (see `_predict_seasonal_naive` for why that call's window is exactly right).
  Null (not imputed) if that historical window isn't fully present for a style.
- `ewma_persistence`: `log1p(ewma_halflife_13w)`, the style's own causal EWMA-of-`intensity_shrunk`
  feature as of the origin (already built by `build_features`). See `_predict_ewma_persistence`'s
  docstring for the one deliberate scale caveat (this baseline's underlying signal is EB-shrunk
  intensity, not raw `units_per_active_article`, per the task's own definition of this baseline).
- `parent_category_mean`: TRAILING (causal, current-week-EXCLUDED, expanding all-history) mean of
  `units_per_active_article` across all OTHER styles in the same `PARENT_CATEGORY_GROUP_COL` group
  (`index_group_name` -- see that constant's docstring for why, over `garment_group_name`), log1p'd.
- `global_mean`: TRAILING (causal, current-week-EXCLUDED, expanding all-history) mean of
  `units_per_active_article` across ALL styles (no exclusion -- the task specifies "all styles" for
  this one, unlike the parent-category baseline's explicit "all OTHER styles"), log1p'd.

Both trailing-mean baselines are built by `_trailing_sum_count_strictly_before`, the SAME causal
`join_asof`-on-`week_start - 1 day` pattern already used twice in this codebase
(`nss.features.style_panel.add_intensity_shrunk`'s `group_mean_trailing`,
`nss.features.model_features._trailing_group_sum_strictly_before`) -- a same-week cross-sectional
mean across other styles is NOT knowable at prediction time (see those two functions' docstrings
for the full rationale); this module deliberately reuses the identical fix rather than
re-introducing that leakage hazard a third time.

METRICS + AGGREGATION: `run_backtest` scores every method at every origin via
`nss.models.metrics.score_predictions` (dropping, per method, only the styles where THAT method's
own prediction is null -- so different baselines can be scored over slightly different subsets of
the same origin's eval set, which is correct: there is nothing to score a null prediction against).
`block_bootstrap_ci` resamples CONTIGUOUS BLOCKS of origins (not i.i.d. individual origins) because
adjacent origins share overlapping 13-week forecast horizons and trailing feature windows and are
therefore temporally correlated -- resampling origins independently would understate the true
variance of the mean. `BOOTSTRAP_BLOCK_SIZE = 4` is chosen because origins are 4 weeks apart but
each forecasts 13 weeks ahead, so a block of 4 origins spans a full 16-week window -- comfortably
wider than the 13-week horizon-overlap window that creates the correlation in the first place, while
still leaving multiple blocks available across the ~20-origin schedule. `summarize_backtest`
reports pooled, COVID-only, and non-COVID-only aggregates (each its own bootstrap over its own
chronologically ordered origin subset) per method.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from nss.features.model_features import build_features
from nss.features.targets import HORIZON_WEEKS, compute_forward_target
from nss.models.metrics import METRIC_KEYS, score_predictions

STEP_WEEKS = 4
# Matches HORIZON_WEEKS: an origin needs at least a forward-target-horizon's worth of history to be
# a remotely sensible starting point for a rolling-origin schedule.
BURN_IN_WEEKS = HORIZON_WEEKS

COVID_WINDOW_START = date(2020, 3, 1)
COVID_WINDOW_END = date(2020, 6, 30)

LAG_52_CUTOFF_WEEKS = 52

BASELINE_METHODS: tuple[str, ...] = (
    "seasonal_naive",
    "ewma_persistence",
    "parent_category_mean",
    "global_mean",
)

# JUDGMENT CALL: index_group_name (e.g. "Ladieswear"/"Menswear", a handful of broad categories) over
# garment_group_name (e.g. "Trousers"/"Shirts", ~20 finer categories) for the parent-category-mean
# baseline. index_group_name is coarser and far more stable in trailing-history terms -- every
# index_group has many styles and dense weekly history from the start of the panel, so its trailing
# mean is a low-variance "what's the parent-category doing" signal; garment_group_name has more
# groups with fewer styles each, giving a noisier trailing mean that behaves more like a smaller,
# less stable version of the global-mean baseline it's meant to sit between. Reasonable people could
# choose garment_group_name instead for a more granular category signal -- this default is the
# project's call, not a forced conclusion from the data.
PARENT_CATEGORY_GROUP_COL = "index_group_name"

# JUDGMENT CALL: WMAPE weight = n_active_articles_level (the origin's own causal, as-of-origin
# article-count feature from build_features), NOT the realized target value. Weighting by the actual
# target would degenerate here: log1p(mean(units_per_active_article)) is legitimately 0 for any
# style-window with zero sales throughout, and a meaningful fraction of eval-set style-origins have
# exactly that -- weighting by actual would zero out those rows' contribution to BOTH the numerator
# and denominator, silently dropping them from the metric rather than scoring them.
# n_active_articles (how many distinct articles the style currently comprises) is a stable, always-
# positive, strictly-causal proxy for "commercial scale" that never zeroes out a low-sales style.
WMAPE_WEIGHT_COL = "n_active_articles_level"

BOOTSTRAP_BLOCK_SIZE = 4
BOOTSTRAP_N_RESAMPLES = 2000
BOOTSTRAP_SEED = 42

DEFAULT_PANEL_PATH = Path("data/processed/style_week_panel.parquet")
DEFAULT_PER_ORIGIN_OUT_PATH = Path("reports/tables/backtest_per_origin.csv")
DEFAULT_SUMMARY_OUT_PATH = Path("reports/tables/backtest_summary.csv")


@dataclass(frozen=True)
class Origin:
    """One rolling-origin week plus its two data-availability/period tags. See module docstring."""

    origin_week: date
    has_52w_lag: bool
    is_covid: bool


def generate_origin_schedule(
    panel: pl.DataFrame,
    step_weeks: int = STEP_WEEKS,
    burn_in_weeks: int = BURN_IN_WEEKS,
    horizon_weeks: int = HORIZON_WEEKS,
) -> list[Origin]:
    """Generate the rolling-origin schedule. See module docstring ORIGIN SCHEDULE section.

    Args:
        panel: The dense style-week panel (or any subset with a `week_start` column) -- only
            `week_start`'s min/max are read.
        step_weeks: Weeks between consecutive origins. Defaults to `STEP_WEEKS` (4).
        burn_in_weeks: Weeks of burn-in after the panel's earliest week before the first origin.
            Defaults to `BURN_IN_WEEKS` (13).
        horizon_weeks: Forward target horizon (must match `compute_forward_target`'s own
            `horizon_weeks`, or `is_covid`/eligibility would be computed against the wrong window).
            Defaults to `HORIZON_WEEKS` (13).

    Returns:
        Origins in chronological order, each tagged `has_52w_lag` and `is_covid`.
    """
    min_week: date = panel["week_start"].min()
    max_week: date = panel["week_start"].max()
    first_origin = min_week + timedelta(weeks=burn_in_weeks)
    last_origin = max_week - timedelta(weeks=horizon_weeks)
    lag52_cutoff = min_week + timedelta(weeks=LAG_52_CUTOFF_WEEKS)

    origins: list[Origin] = []
    origin_week = first_origin
    while origin_week <= last_origin:
        horizon_start = origin_week + timedelta(weeks=1)
        horizon_end = origin_week + timedelta(weeks=horizon_weeks)
        is_covid = horizon_start <= COVID_WINDOW_END and horizon_end >= COVID_WINDOW_START
        has_52w_lag = origin_week >= lag52_cutoff
        origins.append(Origin(origin_week=origin_week, has_52w_lag=has_52w_lag, is_covid=is_covid))
        origin_week = origin_week + timedelta(weeks=step_weeks)
    return origins


def _trailing_sum_count_strictly_before(
    panel: pl.DataFrame,
    group_cols: list[str],
    value_col: str,
    out_sum_col: str,
    out_count_col: str,
) -> pl.DataFrame:
    """Attach a TRAILING (expanding, current-week-EXCLUDED) sum + row-count of `value_col`.

    `group_cols=[]` means "global" (no grouping -- trailing sum/count across the whole panel).
    Reuses the exact `join_asof`-on-`week_start - 1 day` causal pattern as
    `nss.features.model_features._trailing_group_sum_strictly_before` and
    `nss.features.style_panel.add_intensity_shrunk`'s `group_mean_trailing` (see those two
    docstrings for the full rationale: a plain equi-join or row-order `cum_sum` would incorrectly
    null every week where the group itself had no rows, not just the group's genuinely-first week).
    Returning both a sum AND a count (rather than a ready-made mean) lets callers derive both an
    inclusive trailing mean (global baseline) and an exclude-self trailing mean (parent-category
    baseline: subtract the style's own trailing sum/count before dividing) from one implementation.

    Args:
        panel: A frame with at least `group_cols`, `week_start`, `value_col`.
        group_cols: Columns defining the group. Empty list = global (no grouping).
        value_col: Column to sum within the group.
        out_sum_col: Name of the output trailing-sum column.
        out_count_col: Name of the output trailing-row-count column.

    Returns:
        `panel` with `out_sum_col`/`out_count_col` added, both `0` (not null) for rows with no
        trailing history at all yet -- "zero prior rows" is a known, computable quantity, matching
        this project's established null-vs-zero convention (see
        `nss.features.model_features._add_share_of_parent_group`'s identical fill-0 choice).
    """
    group_week = (
        panel.group_by([*group_cols, "week_start"])
        .agg(_sum=pl.col(value_col).sum(), _count=pl.len())
        .sort([*group_cols, "week_start"])
    )
    if group_cols:
        group_week = group_week.with_columns(
            _cum_sum=pl.col("_sum").cum_sum().over(group_cols, order_by="week_start"),
            _cum_count=pl.col("_count").cum_sum().over(group_cols, order_by="week_start"),
        )
    else:
        group_week = group_week.with_columns(
            _cum_sum=pl.col("_sum").cum_sum(),
            _cum_count=pl.col("_count").cum_sum(),
        )
    group_week = group_week.select([*group_cols, "week_start", "_cum_sum", "_cum_count"])

    probe = (
        panel.select([*group_cols, "week_start"])
        .with_row_index("_row_id")
        .with_columns((pl.col("week_start") - pl.duration(days=1)).alias("_probe_week"))
        .sort([*group_cols, "_probe_week"])
    )
    by = group_cols if group_cols else None
    trailing = (
        probe.join_asof(
            group_week.sort([*group_cols, "week_start"]),
            left_on="_probe_week",
            right_on="week_start",
            by=by,
            strategy="backward",
        )
        .select(["_row_id", "_cum_sum", "_cum_count"])
        .rename({"_cum_sum": out_sum_col, "_cum_count": out_count_col})
    )
    return (
        panel.with_row_index("_row_id")
        .join(trailing, on="_row_id", how="left", maintain_order="left")
        .drop("_row_id")
        .with_columns(pl.col(out_sum_col).fill_null(0.0), pl.col(out_count_col).fill_null(0))
    )


def _predict_seasonal_naive(panel: pl.DataFrame, origin_week: date) -> pl.DataFrame:
    """Seasonal-naive baseline: `log1p(mean(units_per_active_article))` over the SAME 13 calendar
    weeks one year (52 weeks) earlier as the origin.

    `compute_forward_target(panel, X)`'s window is `X+1 .. X+13` weeks by construction. Setting
    `X = origin_week - 52 weeks` makes that window exactly `origin_week-51 .. origin_week-39`, i.e.
    `(origin_week-52)+1 .. (origin_week-52)+13` -- precisely "the same 13 calendar weeks one year
    earlier" this baseline is specified to use. Reusing `compute_forward_target` directly (rather
    than re-implementing an equivalent aggregation) inherits its exact null-handling (a style
    without the FULL 13-week window one year back gets a null prediction, not a partial-window
    mean or an imputed value) and its own tested causal-safety guarantee for free.

    Args:
        panel: The dense style-week panel.
        origin_week: The origin week.

    Returns:
        One row per style_key in `panel`, columns `style_key`, `origin_week`,
        `y_pred_seasonal_naive` (nullable).
    """
    lookback_origin = origin_week - timedelta(weeks=52)
    naive = compute_forward_target(panel, lookback_origin, horizon_weeks=HORIZON_WEEKS)
    return naive.select(
        "style_key",
        pl.lit(origin_week).alias("origin_week"),
        pl.col("target").alias("y_pred_seasonal_naive"),
    )


def _predict_ewma_persistence(features: pl.DataFrame) -> pl.DataFrame:
    """13-week EWMA persistence baseline: `log1p(ewma_halflife_13w)` as of the origin.

    `ewma_halflife_13w` is already a causal, as-of-origin feature computed by `build_features`
    (a strictly recursive EWMA of `intensity_shrunk` -- see that module's EWMA section). SCALE
    CAVEAT (deliberate, per this baseline's task-specified definition): `intensity_shrunk` is
    empirical-Bayes-SHRUNK toward the style's `(index_group_name, garment_group_name)` trailing
    mean, not the raw `units_per_active_article` the target itself aggregates -- this is the one
    baseline whose underlying signal genuinely differs from the target's own raw metric. `log1p` is
    applied purely to land the prediction on the target's log1p scale for comparison; it does not
    undo the EB shrinkage.

    Args:
        features: `build_features`'s output (must include `ewma_halflife_13w`).

    Returns:
        `style_key`, `origin_week`, `y_pred_ewma_persistence`.
    """
    return features.select(
        "style_key",
        "origin_week",
        pl.col("ewma_halflife_13w").log1p().alias("y_pred_ewma_persistence"),
    )


def _add_trailing_baseline_means(
    panel: pl.DataFrame, group_col: str = PARENT_CATEGORY_GROUP_COL
) -> pl.DataFrame:
    """Compute the parent-category-mean and global-mean baseline predictions, densely, once.

    Follows the same "compute densely across the whole panel with causal window/as-of functions,
    filter to requested origins at the very end" strategy as `nss.features.model_features`'s module
    docstring (see IMPLEMENTATION STRATEGY there) -- cheaper than recomputing per origin, and the
    causal-safety argument is identical: every trailing sum/count here is a function of rows
    strictly before a given `week_start`, so filtering afterward to any subset of origin weeks
    cannot introduce leakage no matter which weeks end up selected as "the origin".

    Args:
        panel: The dense style-week panel.
        group_col: Column defining the parent-category group. Defaults to
            `PARENT_CATEGORY_GROUP_COL` (`index_group_name`).

    Returns:
        `style_key`, `week_start`, `y_pred_parent_category_mean`, `y_pred_global_mean` -- one row
        per (style_key, week_start) in `panel`. Both nullable (null only when NO trailing history
        exists at all yet for the relevant population -- the group's/panel's genuinely first week).
    """
    working = panel.select("style_key", group_col, "week_start", "units_per_active_article")

    working = _trailing_sum_count_strictly_before(
        working, [], "units_per_active_article", "_global_sum", "_global_count"
    )
    working = _trailing_sum_count_strictly_before(
        working, [group_col], "units_per_active_article", "_group_sum", "_group_count"
    )
    working = _trailing_sum_count_strictly_before(
        working, ["style_key"], "units_per_active_article", "_own_sum", "_own_count"
    )

    other_count = pl.col("_group_count") - pl.col("_own_count")
    other_sum = pl.col("_group_sum") - pl.col("_own_sum")
    working = working.with_columns(
        pl.when(pl.col("_global_count") > 0)
        .then(pl.col("_global_sum") / pl.col("_global_count"))
        .otherwise(None)
        .alias("_global_mean_raw"),
        pl.when(other_count > 0)
        .then(other_sum / other_count)
        .otherwise(None)
        .alias("_parent_category_mean_raw"),
    )
    return working.select(
        "style_key",
        "week_start",
        pl.col("_parent_category_mean_raw").log1p().alias("y_pred_parent_category_mean"),
        pl.col("_global_mean_raw").log1p().alias("y_pred_global_mean"),
    )


def build_predictions_frame(panel: pl.DataFrame, origin_weeks: list[date]) -> pl.DataFrame:
    """Build the eval-set + all-4-baselines-predictions frame for a set of origins.

    See module docstring EVAL SET section for the exact (a) + (b) eligibility rule.

    Args:
        panel: The dense style-week panel.
        origin_weeks: Origin weeks to build predictions for.

    Returns:
        One row per `(style_key, origin_week)` in the eval set, columns: `style_key`,
        `origin_week`, `y_true` (never null -- eval-set membership already filters this),
        `weight` (`WMAPE_WEIGHT_COL`, i.e. `n_active_articles_level`), and one
        `y_pred_<method>` column per `BASELINE_METHODS` entry (each independently nullable).
    """
    features = build_features(panel, origin_weeks)
    targets = pl.concat(
        [compute_forward_target(panel, ow, horizon_weeks=HORIZON_WEEKS) for ow in origin_weeks]
    )
    seasonal_naive = pl.concat([_predict_seasonal_naive(panel, ow) for ow in origin_weeks])
    trailing_means = _add_trailing_baseline_means(panel).rename({"week_start": "origin_week"})

    base = features.select("style_key", "origin_week", WMAPE_WEIGHT_COL, "ewma_halflife_13w")
    base = base.join(
        targets.select("style_key", "origin_week", pl.col("target").alias("y_true")),
        on=["style_key", "origin_week"],
        how="inner",
    )
    base = base.filter(pl.col("y_true").is_not_null())

    base = base.with_columns(pl.col("ewma_halflife_13w").log1p().alias("y_pred_ewma_persistence"))
    base = base.join(seasonal_naive, on=["style_key", "origin_week"], how="left")
    base = base.join(trailing_means, on=["style_key", "origin_week"], how="left")
    base = base.rename({WMAPE_WEIGHT_COL: "weight"})

    pred_cols = [f"y_pred_{method}" for method in BASELINE_METHODS]
    return base.select("style_key", "origin_week", "y_true", "weight", *pred_cols)


def run_backtest(panel: pl.DataFrame, origins: Sequence[Origin] | None = None) -> pl.DataFrame:
    """Run the full rolling-origin backtest: every baseline, scored, at every origin.

    Args:
        panel: The dense style-week panel.
        origins: Origins to backtest. Defaults to `generate_origin_schedule(panel)`.

    Returns:
        One row per `(origin_week, method)`, columns: `origin_week`, `method`, `has_52w_lag`,
        `is_covid`, `n_eval_set` (the origin's full eval-set size, shared across all 4 methods),
        `n_eval` (that method's own scored count, after dropping styles where its prediction is
        null), plus one column per metric in `nss.models.metrics.METRIC_KEYS`.
    """
    if origins is None:
        origins = generate_origin_schedule(panel)
    origin_weeks = [origin.origin_week for origin in origins]
    predictions = build_predictions_frame(panel, origin_weeks)

    rows: list[dict[str, object]] = []
    for origin in origins:
        origin_df = predictions.filter(pl.col("origin_week") == origin.origin_week)
        n_eval_set = origin_df.height
        for method in BASELINE_METHODS:
            pred_col = f"y_pred_{method}"
            sub = origin_df.filter(pl.col(pred_col).is_not_null())
            metrics = score_predictions(
                sub["y_true"].to_numpy(),
                sub[pred_col].to_numpy(),
                sub["weight"].to_numpy(),
            )
            n_eval = metrics.pop("n_eval")
            rows.append(
                {
                    "origin_week": origin.origin_week,
                    "method": method,
                    "has_52w_lag": origin.has_52w_lag,
                    "is_covid": origin.is_covid,
                    "n_eval_set": n_eval_set,
                    "n_eval": int(n_eval),
                    **metrics,
                }
            )
    return pl.DataFrame(rows)


def block_bootstrap_ci(
    values: Sequence[float],
    block_size: int = BOOTSTRAP_BLOCK_SIZE,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Moving-block bootstrap 95% CI for the mean of a chronologically ordered metric sequence.

    See module docstring METRICS + AGGREGATION section for why contiguous blocks (not i.i.d.
    per-origin resampling) are required, and the `BOOTSTRAP_BLOCK_SIZE` reasoning.

    Args:
        values: Per-origin metric values, in chronological (origin) order. `nan` entries (e.g. an
            origin with zero styles scored for a given method) are dropped before resampling.
        block_size: Contiguous block length. Defaults to `BOOTSTRAP_BLOCK_SIZE` (4).
        n_resamples: Number of bootstrap resamples. Defaults to `BOOTSTRAP_N_RESAMPLES` (2000).
        seed: `numpy` RNG seed, for reproducibility. Defaults to `BOOTSTRAP_SEED` (42).

    Returns:
        `(point_estimate, ci_low_2.5pct, ci_high_97.5pct)`, all `nan` if `values` has no non-nan
        entries. If there are fewer values than `block_size`, the single available block is used
        for every resample (a degenerate, zero-variance bootstrap -- CI collapses to the point
        estimate), which is the correct, if uninformative, behavior for a too-small sample.
    """
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    n = arr.shape[0]
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    point_estimate = float(np.mean(arr))
    if n == 1:
        return point_estimate, point_estimate, point_estimate

    rng = np.random.default_rng(seed)
    n_blocks_needed = -(-n // block_size)  # ceil division
    max_start = max(n - block_size, 0)
    boot_means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        starts = rng.integers(0, max_start + 1, size=n_blocks_needed)
        resampled = np.concatenate([arr[s : s + block_size] for s in starts])[:n]
        boot_means[i] = np.mean(resampled)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return point_estimate, float(ci_low), float(ci_high)


def summarize_backtest(per_origin: pl.DataFrame) -> pl.DataFrame:
    """Pooled / COVID-only / non-COVID-only block-bootstrap aggregates, per method, per metric.

    Args:
        per_origin: `run_backtest`'s output.

    Returns:
        One row per `(method, split)` where `split` in `{"pooled", "covid", "non_covid"}`, columns
        `n_origins` plus `<metric>_mean` / `<metric>_ci_low` / `<metric>_ci_high` for every metric
        in `nss.models.metrics.METRIC_KEYS`.
    """
    rows: list[dict[str, object]] = []
    for method in sorted(per_origin["method"].unique().to_list()):
        method_df = per_origin.filter(pl.col("method") == method).sort("origin_week")
        splits = (
            ("pooled", method_df),
            ("covid", method_df.filter(pl.col("is_covid"))),
            ("non_covid", method_df.filter(~pl.col("is_covid"))),
        )
        for split_name, split_df in splits:
            row: dict[str, object] = {
                "method": method,
                "split": split_name,
                "n_origins": split_df.height,
            }
            for metric in METRIC_KEYS:
                mean, ci_low, ci_high = block_bootstrap_ci(split_df[metric].to_list())
                row[f"{metric}_mean"] = mean
                row[f"{metric}_ci_low"] = ci_low
                row[f"{metric}_ci_high"] = ci_high
            rows.append(row)
    return pl.DataFrame(rows)


def main() -> None:
    """CLI entry point: run the full backtest, write the per-origin and summary CSVs."""
    panel = pl.read_parquet(DEFAULT_PANEL_PATH)
    origins = generate_origin_schedule(panel)
    print(
        f"Origin schedule: {len(origins)} origins, {origins[0].origin_week} .. "
        f"{origins[-1].origin_week}"
    )
    print(f"  is_covid: {sum(o.is_covid for o in origins)}")
    print(f"  has_52w_lag: {sum(o.has_52w_lag for o in origins)}")

    per_origin = run_backtest(panel, origins)
    summary = summarize_backtest(per_origin)

    DEFAULT_PER_ORIGIN_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    per_origin.write_csv(DEFAULT_PER_ORIGIN_OUT_PATH)
    summary.write_csv(DEFAULT_SUMMARY_OUT_PATH)
    print(f"Wrote {DEFAULT_PER_ORIGIN_OUT_PATH} ({per_origin.height} rows)")
    print(f"Wrote {DEFAULT_SUMMARY_OUT_PATH} ({summary.height} rows)")


if __name__ == "__main__":
    main()
