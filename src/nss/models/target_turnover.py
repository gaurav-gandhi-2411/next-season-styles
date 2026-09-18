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

from nss.features.targets import compute_forward_target
from nss.models.backtest import Origin, generate_origin_schedule

TARGET_COLUMNS: tuple[str, ...] = ("units_per_active_article", "intensity_shrunk")
TOP_KS: tuple[int, ...] = (3, 10)

DEFAULT_PANEL_PATH = Path("data/processed/style_week_panel.parquet")
DEFAULT_OUT_PATH = Path("reports/tables/target_turnover.csv")


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


def compute_turnover_table(panel: pl.DataFrame, origins: Sequence[Origin]) -> pl.DataFrame:
    """Compute the 4-cell (target_column x top_k) mean/std consecutive-origin overlap table.

    Args:
        panel: The dense style-week panel.
        origins: Backtest origins in chronological order (see
            `nss.models.backtest.generate_origin_schedule`). Overlap is computed only for the
            `len(origins) - 1` CONSECUTIVE pairs `(origins[i], origins[i+1])`.

    Returns:
        One row per `(target_column, top_k)`, columns `target_column`, `top_k`, `n_pairs`,
        `mean_overlap`, `std_overlap` (sample std, ddof=1; `0.0` if `n_pairs < 2`).
    """
    origin_weeks = [o.origin_week for o in origins]
    ranked_by_origin: dict[tuple[date, str], list[str]] = {
        (ow, target_column): ranked_style_keys(panel, ow, target_column)
        for ow in origin_weeks
        for target_column in TARGET_COLUMNS
    }

    rows: list[dict[str, object]] = []
    for target_column in TARGET_COLUMNS:
        for k in TOP_KS:
            top_k_sets = [set(ranked_by_origin[(ow, target_column)][:k]) for ow in origin_weeks]
            fracs = [
                overlap_fraction(top_k_sets[i], top_k_sets[i + 1], k)
                for i in range(len(top_k_sets) - 1)
            ]
            arr = np.asarray(fracs, dtype=float)
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


def main() -> None:
    """CLI entry point: run the diagnostic over the real panel's origin schedule, write the CSV."""
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


if __name__ == "__main__":
    main()
