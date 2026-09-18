"""Diagnostic: how much does the TRUE top-K style ranking turn over between consecutive backtest
origins, and does the EB-shrunk `intensity_shrunk` target reduce that turnover relative to the raw
`units_per_active_article` target?

MOTIVATION: `nss.models.metrics`'s headline metric is Precision@3 (Precision@10 as a companion). If
the realized top-3/top-10 styles by forward target value turn over almost completely between one
rolling origin and the next (`nss.models.backtest.generate_origin_schedule`, step 4 weeks), then
Precision@3 is close to unlearnable regardless of model quality -- there is no stable signal a
model could pick up even in principle. This module measures that turnover directly, gating whether
switching the forecast target from raw `units_per_active_article` to EB-shrunk `intensity_shrunk`
(via `nss.features.targets.compute_forward_target`'s `target_column` parameter) is worth doing on
stability grounds.

METHOD: for each of the 20 real-panel backtest origins (chronological order), compute the TRUE
top-3 and top-10 style_keys by realized `compute_forward_target` value -- separately for
`target_column="units_per_active_article"` and `target_column="intensity_shrunk"` -- restricted to
styles with a non-null target at that origin (see `compute_forward_target`'s WINDOWING /
NULL-HANDLING RULE). Ties are broken by `style_key` ascending, for determinism. For each of the 19
CONSECUTIVE origin pairs, the overlap fraction `|top-K(origin_i) intersect top-K(origin_i+1)| / K`
is computed for K in {3, 10}, for both raw and shrunk. `compute_turnover_table` reports the mean and
sample std (ddof=1) of that overlap fraction across the 19 pairs, per (target_column, K) cell -- a
4-cell table. This is a genuine measurement, not a hypothesis-confirming one: the report states
whichever way the raw-vs-shrunk comparison falls, including a "shrunk is not meaningfully more
stable, keep raw" verdict if that's what the numbers show.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from nss.features.targets import compute_forward_target
from nss.models.backtest import Origin, generate_origin_schedule

TARGET_COLUMNS: tuple[str, ...] = ("units_per_active_article", "intensity_shrunk")
TOP_KS: tuple[int, ...] = (3, 10)

DEFAULT_PANEL_PATH = Path("data/processed/style_week_panel.parquet")
DEFAULT_OUT_PATH = Path("reports/tables/target_turnover.csv")
DEFAULT_CORRECTED_OUT_PATH = Path("reports/tables/target_turnover_corrected.csv")


def ranked_style_keys(panel: pl.DataFrame, origin_week: date, target_column: str) -> list[str]:
    """TRUE style_keys at `origin_week`, ranked by realized `compute_forward_target` value
    (descending), restricted to non-null targets. Ties broken by `style_key` ascending.

    Args:
        panel: The dense style-week panel.
        origin_week: The origin week to rank styles as of.
        target_column: Which panel column `compute_forward_target` aggregates -- see that
            function's own `target_column` docstring.

    Returns:
        `style_key`s in descending-target order, non-null targets only. The top-K prefix of this
        list is the TRUE top-K ranking at this origin, for any K.
    """
    targets = compute_forward_target(panel, origin_week, target_column=target_column)
    ranked = targets.filter(pl.col("target").is_not_null()).sort(
        ["target", "style_key"], descending=[True, False]
    )
    return ranked["style_key"].to_list()


def overlap_fraction(a: set[str], b: set[str], k: int) -> float:
    """`|a intersect b| / k` -- fraction overlap between two top-k style_key sets.

    Args:
        a: First top-k style_key set.
        b: Second top-k style_key set.
        k: The K both sets were truncated to (the fraction's denominator).

    Returns:
        Overlap fraction in `[0, 1]`.
    """
    return len(a & b) / k


def compute_pair_overlaps(
    panel: pl.DataFrame, origins: Sequence[Origin]
) -> dict[tuple[str, int], list[float]]:
    """Per-consecutive-origin-pair overlap fractions, keyed by `(target_column, top_k)`.

    Shared by `compute_turnover_table` (independent-sample aggregate) and
    `compute_turnover_paired_diff_table` (paired raw-vs-shrunk difference) so both consume the same
    underlying per-pair overlap series -- see module docstring METHOD.

    Args:
        panel: The dense style-week panel.
        origins: Backtest origins in chronological order. Overlap is computed only for the
            `len(origins) - 1` CONSECUTIVE pairs `(origins[i], origins[i+1])`.

    Returns:
        `{(target_column, top_k): [overlap_fraction, ...]}`, one list of `len(origins) - 1` values
        per cell, in origin-pair chronological order (so index `i` means the SAME origin-pair
        across every `target_column`/`top_k` cell -- required for the paired comparison).
    """
    origin_weeks = [o.origin_week for o in origins]
    ranked_by_origin: dict[tuple[date, str], list[str]] = {
        (ow, target_column): ranked_style_keys(panel, ow, target_column)
        for ow in origin_weeks
        for target_column in TARGET_COLUMNS
    }

    result: dict[tuple[str, int], list[float]] = {}
    for target_column in TARGET_COLUMNS:
        for k in TOP_KS:
            top_k_sets = [set(ranked_by_origin[(ow, target_column)][:k]) for ow in origin_weeks]
            result[(target_column, k)] = [
                overlap_fraction(top_k_sets[i], top_k_sets[i + 1], k)
                for i in range(len(top_k_sets) - 1)
            ]
    return result


def compute_turnover_table(panel: pl.DataFrame, origins: Sequence[Origin]) -> pl.DataFrame:
    """Compute the 4-cell (target_column x top_k) mean/std consecutive-origin overlap table.

    NOTE: this treats each `(target_column, top_k)` cell's 19 overlap fractions as an independent
    sample -- valid for describing each cell on its own, but NOT valid for a raw-vs-shrunk
    significance comparison, since raw and shrunk overlap at the SAME origin-pair are paired
    observations, not independent draws. See `compute_turnover_paired_diff_table` for the correct
    paired comparison (`reports/tables/target_turnover_corrected.csv`).

    Args:
        panel: The dense style-week panel.
        origins: Backtest origins in chronological order (see
            `nss.models.backtest.generate_origin_schedule`). Overlap is computed only for the
            `len(origins) - 1` CONSECUTIVE pairs `(origins[i], origins[i+1])`.

    Returns:
        One row per `(target_column, top_k)`, columns `target_column`, `top_k`, `n_pairs`,
        `mean_overlap`, `std_overlap` (sample std, ddof=1; `0.0` if `n_pairs < 2`).
    """
    overlaps = compute_pair_overlaps(panel, origins)
    rows: list[dict[str, object]] = []
    for target_column in TARGET_COLUMNS:
        for k in TOP_KS:
            arr = np.asarray(overlaps[(target_column, k)], dtype=float)
            rows.append(
                {
                    "target_column": target_column,
                    "top_k": k,
                    "n_pairs": arr.shape[0],
                    "mean_overlap": float(arr.mean()) if arr.size else float("nan"),
                    "std_overlap": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                }
            )
    return pl.DataFrame(rows)


def compute_turnover_paired_diff_table(
    panel: pl.DataFrame, origins: Sequence[Origin]
) -> pl.DataFrame:
    """PAIRED raw-vs-shrunk overlap comparison, per top_k -- the statistically correct correction
    to `compute_turnover_table`'s independent-sample means/stds.

    Raw and shrunk overlap fractions at each of the 19 consecutive origin-pairs are two
    measurements of the SAME origin-pair (paired observations), not independent samples. This
    computes the per-pair difference `diff = shrunk_overlap - raw_overlap` and reports its own
    mean/std plus a one-sample t-test against `popmean=0`
    (`scipy.stats.ttest_1samp(diffs, popmean=0)`) -- mathematically equivalent to a paired t-test on
    (shrunk, raw) here, since a paired t-test IS a one-sample t-test on the difference series.

    Args:
        panel: The dense style-week panel.
        origins: Backtest origins in chronological order.

    Returns:
        One row per `top_k`, columns `top_k`, `n_pairs`, `raw_mean_overlap`, `raw_std_overlap`,
        `shrunk_mean_overlap`, `shrunk_std_overlap` (independent-sample stats, for reference/
        continuity with `target_turnover.csv`), `mean_diff`, `std_diff` (sample std, ddof=1, of the
        difference series), `t_statistic`, `p_value` (two-sided, `H0: mean_diff == 0`).
    """
    overlaps = compute_pair_overlaps(panel, origins)
    rows: list[dict[str, object]] = []
    for k in TOP_KS:
        raw = np.asarray(overlaps[("units_per_active_article", k)], dtype=float)
        shrunk = np.asarray(overlaps[("intensity_shrunk", k)], dtype=float)
        diffs = shrunk - raw
        t_result = stats.ttest_1samp(diffs, popmean=0.0)
        rows.append(
            {
                "top_k": k,
                "n_pairs": diffs.shape[0],
                "raw_mean_overlap": float(raw.mean()),
                "raw_std_overlap": float(raw.std(ddof=1)) if raw.size > 1 else 0.0,
                "shrunk_mean_overlap": float(shrunk.mean()),
                "shrunk_std_overlap": float(shrunk.std(ddof=1)) if shrunk.size > 1 else 0.0,
                "mean_diff": float(diffs.mean()),
                "std_diff": float(diffs.std(ddof=1)) if diffs.size > 1 else 0.0,
                "t_statistic": float(t_result.statistic),
                "p_value": float(t_result.pvalue),
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    """CLI entry point: run the diagnostic over the real panel's origin schedule, write the CSV,
    then write the paired-diff correction (see `compute_turnover_paired_diff_table`) -- both are
    kept: `target_turnover.csv` as the historical record, `target_turnover_corrected.csv` as the
    statistically correct table for reporting purposes."""
    panel = pl.read_parquet(DEFAULT_PANEL_PATH)
    origins = generate_origin_schedule(panel)
    print(f"Origin schedule: {len(origins)} origins, {len(origins) - 1} consecutive pairs")

    table = compute_turnover_table(panel, origins)
    DEFAULT_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(DEFAULT_OUT_PATH)
    print(f"Wrote {DEFAULT_OUT_PATH} ({table.height} rows)")
    for row in table.iter_rows(named=True):
        print(
            f"  {row['target_column']} top-{row['top_k']}: "
            f"mean_overlap={row['mean_overlap']:.3f} std={row['std_overlap']:.3f} "
            f"(n_pairs={row['n_pairs']})"
        )

    corrected = compute_turnover_paired_diff_table(panel, origins)
    corrected.write_csv(DEFAULT_CORRECTED_OUT_PATH)
    print(f"Wrote {DEFAULT_CORRECTED_OUT_PATH} ({corrected.height} rows)")
    for row in corrected.iter_rows(named=True):
        print(
            f"  top-{row['top_k']}: mean_diff={row['mean_diff']:.3f} "
            f"std_diff={row['std_diff']:.3f} t={row['t_statistic']:.3f} p={row['p_value']:.4f} "
            f"(n_pairs={row['n_pairs']})"
        )


if __name__ == "__main__":
    main()
