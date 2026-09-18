"""Forward-looking target construction for the style-week forecasting task.

This module intentionally does NOT write anything into `data/processed/style_week_panel.parquet`.
The panel itself is a feature store: every column in it must be safe to read at *any* origin week
without leaking the future (see `nss.features.style_panel`'s causal-safety warnings). A forecasting
target is, by definition, the opposite -- it is only meaningful relative to a specific origin week
and is built entirely from weeks AFTER that origin. Materializing it as a static panel column risks
a later step naively treating it as an input feature and leaking the future into the past. Instead,
`compute_forward_target` is a pure function of (panel, origin_week) that a rolling-origin backtest
harness calls once per origin, producing a fresh target frame for that origin only.

TARGET DEFINITION: for a given `style_key` and `origin_week`, the target is
`log1p(mean(units_per_active_article) over the HORIZON_WEEKS calendar weeks strictly after
origin_week, i.e. origin_week + 1 week through origin_week + HORIZON_WEEKS weeks inclusive)`.

WINDOWING / NULL-HANDLING RULE (deliberate, not incidental): a style_key must have the FULL
`horizon_weeks`-week forward window present in `panel` (all `horizon_weeks` calendar weeks
represented as rows -- true whenever the window sits fully inside the style's densified
[`first_week_seen`, `last_week_seen`] lifetime) to receive a non-null target. A style with fewer
than `horizon_weeks` future weeks available from a given origin (e.g. near its own end-of-life, or
an origin close to the end of the observed dataset) gets `target = null` for that origin, rather
than a partial-window mean. A partial-window mean would be systematically biased relative to a
full-window mean for any style with trend or seasonality over the horizon (the two are not
computing the same underlying quantity), and would silently make near-end-of-life origins look like
normal training examples with an incomparable target scale -- rejecting them is the safer, more
defensible default; a rolling-origin backtest simply won't have a scored prediction for those
(style_key, origin_week) pairs, which is correct (there is nothing to fairly score against).

CAUSAL SAFETY: `compute_forward_target` filters the input panel to
`origin_week < week_start <= origin_week + horizon_weeks weeks` before touching it -- every value
that reaches the aggregation is dated strictly after `origin_week`. No column or row at or before
`origin_week` is read. See `tests/test_targets.py::test_compute_forward_target_is_causally_safe`
for a test that actively corrupts pre-origin and out-of-window data and confirms the target is
unaffected, and corrupts a genuinely in-window row and confirms the target DOES change.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

HORIZON_WEEKS = 13


def compute_forward_target(
    panel: pl.DataFrame, origin_week: date, horizon_weeks: int = HORIZON_WEEKS
) -> pl.DataFrame:
    """Compute the forward `horizon_weeks`-week target for every style_key, as of `origin_week`.

    See module docstring for the exact target definition and null-handling rule.

    Args:
        panel: The dense style-week panel (or any subset/superset with at least `style_key`,
            `week_start`, `units_per_active_article` columns).
        origin_week: The origin week (a Monday `week_start` value). Only weeks strictly after this
            date are read -- see the module docstring's CAUSAL SAFETY note.
        horizon_weeks: Number of forward calendar weeks in the target window. Defaults to
            `HORIZON_WEEKS` (13).

    Returns:
        One row per distinct style_key present anywhere in `panel`, with columns `style_key`,
        `origin_week`, `n_weeks_in_window`, `target`. `target` is `log1p(mean(
        units_per_active_article))` over the forward window if all `horizon_weeks` weeks are
        present for that style_key in `panel`, else null (see WINDOWING / NULL-HANDLING RULE).
    """
    window_start = origin_week + timedelta(weeks=1)
    window_end = origin_week + timedelta(weeks=horizon_weeks)

    window = panel.filter(
        (pl.col("week_start") >= window_start) & (pl.col("week_start") <= window_end)
    )
    agg = window.group_by("style_key").agg(
        mean_units_per_active_article=pl.col("units_per_active_article").mean(),
        n_weeks_in_window=pl.len(),
    )

    all_styles = panel.select("style_key").unique()
    result = all_styles.join(agg, on="style_key", how="left").with_columns(
        pl.col("n_weeks_in_window").fill_null(0)
    )

    full_window = pl.col("n_weeks_in_window") == horizon_weeks
    result = result.with_columns(
        target=pl.when(full_window)
        .then(pl.col("mean_units_per_active_article").log1p())
        .otherwise(None),
        origin_week=pl.lit(origin_week),
    )
    return result.select(["style_key", "origin_week", "n_weeks_in_window", "target"])
