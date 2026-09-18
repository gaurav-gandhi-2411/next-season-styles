"""Backtest v2: like-for-like LightGBM-vs-baselines comparison, random-guessing floor, paired diffs.

FIXES THE PHASE 2 GAP (see PLAN / session report): Phase 2's `backtest_summary.csv` scored the 4
baselines over all 20 rolling origins, while `backtest_summary_lightgbm.csv` scored LightGBM over
only the 12 walk-forward test origins (`nss.models.lightgbm_model`'s `INITIAL_POOL_SIZE` reserves
the first 8 as a training-only pool) -- an invalid head-to-head comparison. This module re-runs
the 4 baselines restricted to the SAME 12 origins, adds a `random_floor` 5th "method" (see
`nss.models.random_floor`), and computes PAIRED per-origin differences (LightGBM minus each
baseline) so every comparison in `backtest_summary_v2.csv` / `backtest_paired_diff.csv` is like-for-
like: same origins, same eval-set population, same block-bootstrap machinery.

WHICH 12 ORIGINS (JUDGMENT CALL, verified not assumed): `identify_lightgbm_origins` reads the
origin weeks directly out of the CHECKED-IN
`nss.models.lightgbm_model.DEFAULT_PER_ORIGIN_OUT_PATH` CSV -- the actual artifact LightGBM was
scored against -- rather than assuming `origins[INITIAL_POOL_SIZE:]` matches
`lightgbm_model.INITIAL_POOL_SIZE`'s CURRENT value (that constant could in principle drift from what
produced the checked-in CSV). It then cross-checks that set against
`generate_origin_schedule(panel)[INITIAL_POOL_SIZE:]` and raises if they disagree, so a future
silent mismatch between the code and the checked-in artifact fails loudly instead of comparing the
wrong origins.

"NO DEMONSTRATED SIGNAL" (see module docstring of `nss.models.random_floor` for the floor's own
construction): `add_signal_flags` adds one `<metric>_no_demonstrated_signal` boolean column per
metric to the summary table -- `True` iff that (method, split, metric) row's 95% CI overlaps
`random_floor`'s 95% CI for the SAME split and metric. `None` (not `True`/`False`) if either CI is
undefined (`nan` -- e.g. a metric with zero valid values across every origin in that split), since
"can't tell" must never silently read as "no signal" (or "has signal") -- see
`nss.models.random_floor` module docstring and the operating-manual's fail-closed-not-open rule for
guards inspecting external/computed state. The check is purely CI-overlap; it is deliberately
NOT metric-direction-aware (WMAPE is "lower is better", every other metric here is "higher is
better") -- overlap in either direction is still "can't statistically distinguish from chance."

PAIRED DIFFERENCES: `paired_diff_table` computes, for every (LightGBM, baseline) pair and every
metric, the PER-ORIGIN difference (LightGBM's value minus that baseline's value, same origin) and
block-bootstraps THAT difference series directly (reusing
`nss.models.backtest.block_bootstrap_ci`, same block size/resamples/seed) -- this is the
statistically correct way to test "does LightGBM beat baseline X", since it accounts for the fact
that both methods' per-origin errors are correlated (same origin, same eval set, same underlying
demand shock), which two independently-computed CIs compared informally would miss.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from nss.models.backtest import (
    BASELINE_METHODS,
    BOOTSTRAP_BLOCK_SIZE,
    BOOTSTRAP_N_RESAMPLES,
    BOOTSTRAP_SEED,
    Origin,
    block_bootstrap_ci,
    build_predictions_frame,
    generate_origin_schedule,
    run_backtest,
    summarize_backtest,
)
from nss.models.lightgbm_model import DEFAULT_PER_ORIGIN_OUT_PATH as LIGHTGBM_PER_ORIGIN_PATH
from nss.models.lightgbm_model import INITIAL_POOL_SIZE, run_lightgbm_walk_forward
from nss.models.lightgbm_model import METHOD_NAME as LIGHTGBM_METHOD
from nss.models.metrics import METRIC_KEYS
from nss.models.random_floor import (
    RANDOM_FLOOR_METHOD,
    aggregate_random_floor_over_seeds,
    score_random_floor_per_origin,
)

# A split needs at least this many origins for its CI to be reported as more than "directional
# only" -- see module docstring / task spec. Deliberately less than BOOTSTRAP_BLOCK_SIZE*2 (8) would
# leave under 2 full blocks for the bootstrap to draw from; 8 is the task's own specified cutoff.
MIN_ORIGINS_FOR_NON_DIRECTIONAL = 8

DEFAULT_PANEL_PATH = Path("data/processed/style_week_panel.parquet")
DEFAULT_SUMMARY_V2_OUT_PATH = Path("reports/tables/backtest_summary_v2.csv")
DEFAULT_PAIRED_DIFF_OUT_PATH = Path("reports/tables/backtest_paired_diff.csv")


def identify_lightgbm_origins(
    panel: pl.DataFrame, per_origin_csv_path: Path = LIGHTGBM_PER_ORIGIN_PATH
) -> list[Origin]:
    """The exact `Origin`s LightGBM was walk-forward-evaluated on. See module docstring.

    Args:
        panel: The dense style-week panel (used to regenerate the full `Origin` schedule, so the
            returned objects carry their `has_52w_lag`/`is_covid` tags).
        per_origin_csv_path: Path to LightGBM's checked-in per-origin CSV (the source of truth for
            which origin weeks it was actually scored on). Defaults to
            `nss.models.lightgbm_model.DEFAULT_PER_ORIGIN_OUT_PATH`.

    Returns:
        The matching `Origin`s, in chronological order.

    Raises:
        ValueError: if the CSV's origin weeks don't exactly match
            `generate_origin_schedule(panel)[INITIAL_POOL_SIZE:]` -- a silent mismatch here would
            mean every downstream comparison in this module is scored against the wrong origins.
    """
    csv_weeks = set(
        pl.read_csv(per_origin_csv_path, try_parse_dates=True)["origin_week"].unique().to_list()
    )
    all_origins = generate_origin_schedule(panel)
    expected_origins = all_origins[INITIAL_POOL_SIZE:]
    expected_weeks = {o.origin_week for o in expected_origins}
    if csv_weeks != expected_weeks:
        raise ValueError(
            f"LightGBM per-origin CSV origin weeks {sorted(csv_weeks)} do not match "
            f"generate_origin_schedule(panel)[INITIAL_POOL_SIZE:] "
            f"{sorted(expected_weeks)} -- refusing to compare against the wrong origins."
        )
    return expected_origins


def add_signal_flags(
    summary: pl.DataFrame,
    metric_keys: Sequence[str] = METRIC_KEYS,
    min_origins_for_non_directional: int = MIN_ORIGINS_FOR_NON_DIRECTIONAL,
) -> pl.DataFrame:
    """Add `<metric>_no_demonstrated_signal` (per metric) and `directional_only` columns.

    See module docstring "NO DEMONSTRATED SIGNAL".

    Args:
        summary: `nss.models.backtest.summarize_backtest`'s output, with a `random_floor` method
            present for every split in `summary["split"]`.
        metric_keys: Metrics to flag. Defaults to `nss.models.metrics.METRIC_KEYS`.
        min_origins_for_non_directional: `n_origins` below this gets `directional_only=True`.
            Defaults to `MIN_ORIGINS_FOR_NON_DIRECTIONAL` (8).

    Returns:
        `summary` with the new columns appended (row order and existing columns unchanged).
    """
    floor_ci: dict[str, dict[str, tuple[float, float]]] = {}
    for row in summary.filter(pl.col("method") == RANDOM_FLOOR_METHOD).iter_rows(named=True):
        floor_ci[row["split"]] = {
            metric: (row[f"{metric}_ci_low"], row[f"{metric}_ci_high"]) for metric in metric_keys
        }

    flagged_rows: list[dict[str, object]] = []
    for row in summary.iter_rows(named=True):
        out = dict(row)
        is_floor_row = row["method"] == RANDOM_FLOOR_METHOD
        for metric in metric_keys:
            if is_floor_row:
                out[f"{metric}_no_demonstrated_signal"] = None
                continue
            # A missing random_floor row for this split is a "can't verify" case, not a "no
            # overlap" case -- fail closed to None rather than raising or silently treating it as
            # a pass (see operating-manual guard fail-closed-not-open rule).
            floor_bounds = floor_ci.get(row["split"], {}).get(metric, (float("nan"), float("nan")))
            floor_lo, floor_hi = floor_bounds
            method_lo, method_hi = row[f"{metric}_ci_low"], row[f"{metric}_ci_high"]
            any_nan = any(v != v for v in (floor_lo, floor_hi, method_lo, method_hi))  # nan != nan
            out[f"{metric}_no_demonstrated_signal"] = (
                None if any_nan else bool(method_lo <= floor_hi and floor_lo <= method_hi)
            )
        out["directional_only"] = bool(row["n_origins"] < min_origins_for_non_directional)
        flagged_rows.append(out)
    return pl.DataFrame(flagged_rows)


def paired_diff_table(
    per_origin: pl.DataFrame,
    treatment_method: str = LIGHTGBM_METHOD,
    baseline_methods: Sequence[str] = BASELINE_METHODS,
    metric_keys: Sequence[str] = METRIC_KEYS,
    block_size: int = BOOTSTRAP_BLOCK_SIZE,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    min_origins_for_non_directional: int = MIN_ORIGINS_FOR_NON_DIRECTIONAL,
) -> pl.DataFrame:
    """Paired per-origin (treatment - baseline) differences, block-bootstrapped. See module
    docstring PAIRED DIFFERENCES.

    Args:
        per_origin: A combined per-origin table (same schema as
            `nss.models.backtest.run_backtest`'s output) containing rows for `treatment_method` and
            every entry in `baseline_methods`, all over the SAME set of origins.
        treatment_method: The method being evaluated against each baseline. Defaults to
            `nss.models.lightgbm_model.METHOD_NAME` (`"lightgbm"`).
        baseline_methods: Methods to diff against. Defaults to
            `nss.models.backtest.BASELINE_METHODS`.
        metric_keys: Metrics to diff. Defaults to `nss.models.metrics.METRIC_KEYS`.
        block_size, n_resamples, seed: Passed through to
            `nss.models.backtest.block_bootstrap_ci`. Defaults match the project's standard
            (`BOOTSTRAP_BLOCK_SIZE`=4, `BOOTSTRAP_N_RESAMPLES`=2000, `BOOTSTRAP_SEED`=42).
        min_origins_for_non_directional: `n_origins` below this gets `directional_only=True`.

    Returns:
        One row per `(method_a=treatment_method, method_b=baseline, split, metric)`, columns
        `n_origins`, `directional_only`, `mean_diff`, `diff_ci_low`, `diff_ci_high`.
    """
    treat_df = per_origin.filter(pl.col("method") == treatment_method).sort("origin_week")

    rows: list[dict[str, object]] = []
    for baseline in baseline_methods:
        base_df = per_origin.filter(pl.col("method") == baseline).sort("origin_week")
        joined = treat_df.join(base_df, on="origin_week", how="inner", suffix="_baseline")
        splits = (
            ("pooled", joined),
            ("covid", joined.filter(pl.col("is_covid"))),
            ("non_covid", joined.filter(~pl.col("is_covid"))),
        )
        for split_name, split_df in splits:
            n_origins = split_df.height
            for metric in metric_keys:
                diff_series = (split_df[metric] - split_df[f"{metric}_baseline"]).to_list()
                mean_diff, ci_low, ci_high = block_bootstrap_ci(
                    diff_series, block_size=block_size, n_resamples=n_resamples, seed=seed
                )
                rows.append(
                    {
                        "method_a": treatment_method,
                        "method_b": baseline,
                        "split": split_name,
                        "metric": metric,
                        "n_origins": n_origins,
                        "directional_only": n_origins < min_origins_for_non_directional,
                        "mean_diff": mean_diff,
                        "diff_ci_low": ci_low,
                        "diff_ci_high": ci_high,
                    }
                )
    return pl.DataFrame(rows)


def build_backtest_v2(panel: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run the full v2 rebuild: restricted baselines + LightGBM + random floor, summary + paired
    diffs.

    Args:
        panel: The dense style-week panel.

    Returns:
        `(summary_v2, paired_diff)` -- see `DEFAULT_SUMMARY_V2_OUT_PATH` /
        `DEFAULT_PAIRED_DIFF_OUT_PATH` docstrings-by-name for their exact schemas.
    """
    origins = identify_lightgbm_origins(panel)
    origin_weeks = [o.origin_week for o in origins]

    baseline_per_origin = run_backtest(panel, origins)

    lgbm_per_origin, _config, _grid_results = run_lightgbm_walk_forward(
        panel, generate_origin_schedule(panel)
    )

    predictions = build_predictions_frame(panel, origin_weeks)
    rf_per_seed = score_random_floor_per_origin(predictions, origins)
    rf_per_origin = aggregate_random_floor_over_seeds(rf_per_seed)

    combined = pl.concat([baseline_per_origin, lgbm_per_origin, rf_per_origin], how="vertical")

    summary = summarize_backtest(combined)
    summary_v2 = add_signal_flags(summary)
    paired_diff = paired_diff_table(combined)
    return summary_v2, paired_diff


def main() -> None:
    """CLI entry point: run the v2 backtest rebuild, write the summary + paired-diff CSVs."""
    panel = pl.read_parquet(DEFAULT_PANEL_PATH)
    summary_v2, paired_diff = build_backtest_v2(panel)

    DEFAULT_SUMMARY_V2_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_v2.write_csv(DEFAULT_SUMMARY_V2_OUT_PATH)
    paired_diff.write_csv(DEFAULT_PAIRED_DIFF_OUT_PATH)
    print(f"Wrote {DEFAULT_SUMMARY_V2_OUT_PATH} ({summary_v2.height} rows)")
    print(f"Wrote {DEFAULT_PAIRED_DIFF_OUT_PATH} ({paired_diff.height} rows)")


if __name__ == "__main__":
    main()
