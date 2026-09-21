"""G1: redesign of the emerging ranking, evaluated against the deployed one.

Rules are pre-registered in `reports/v3/PREREGISTRATION.md` (G1/G2, commit `e22a0ed`), committed
before this code existed. Three scores, each computed for the model and for every baseline's own
forecast, all judged against the SAME realised growth `expm1(y_true) / trailing` on the SAME
population P3 (the deployed emerging pool restricted to styles with a seasonal-naive forecast):

- `ratio`    `expm1(f) / trailing`                    (the deployed score, mean-reversion prone)
- `excess`   `f - sn` on the log1p scale              (the candidate: model minus seasonal naive)
- `residual` OLS residuals of `ratio` on `trailing`   (per origin; comparison only)

Primary (both origin sets): redesigned Hit@3-in-top20 minus seasonal naive's `ratio` Hit@3-in-top20,
interval lower bound above 0. Mandatory diagnostic: the constant global-mean forecast under `excess`
must be within 0.10 of the random floor and not demonstrably above it. Adopt only if both pass.

    uv run python -m nss.models.emerging_redesign
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from nss.models import growth_backtest as gb
from nss.models.backtest import Origin, block_bootstrap_ci
from nss.models.metrics import METRIC_KEYS, score_predictions
from nss.models.random_floor import (
    RANDOM_FLOOR_METHOD,
    aggregate_random_floor_over_seeds,
    score_random_floor_per_origin,
)

MODEL = gb.MODEL
SN = "seasonal_naive"
GM = "global_mean"
PRIMARY = "hit_at_3_in_top20"
TOL = 1e-9
FLOOR_TOL = 0.10
CONSTRUCTIONS = ("ratio", "excess", "residual")
OUT = "reports/tables"


def p3(frame: pl.DataFrame) -> pl.DataFrame:
    """Deployed emerging pool (P2) restricted to styles with a seasonal-naive forecast."""
    return gb.population(frame, "P2").filter(pl.col(f"y_pred_{SN}").is_not_null())


def realised_growth(pop: pl.DataFrame) -> pl.DataFrame:
    """Add `g_true` (the 2a realised growth) to a population frame."""
    return pop.with_columns(
        ((pl.col("y_true").exp() - 1.0) / pl.col("trailing_13w_mean_intensity")).alias("g_true")
    )


def score_vector(sub: pl.DataFrame, construction: str, method: str) -> np.ndarray | None:
    """One origin's scores for one (construction, method); None when it is degenerate."""
    f = sub[f"y_pred_{method}"].to_numpy()
    trailing = sub["trailing_13w_mean_intensity"].to_numpy()
    if construction == "excess":
        if method == SN:
            return None  # identically zero: a constant, not a ranking
        return f - sub[f"y_pred_{SN}"].to_numpy()
    ratio = np.expm1(f) / trailing
    if construction == "ratio":
        return ratio
    slope, intercept = np.polyfit(trailing, ratio, 1)
    return ratio - (slope * trailing + intercept)


def per_origin_table(pop: pl.DataFrame, origins: Sequence[Origin]) -> pl.DataFrame:
    """Per-origin metrics for every (construction, method) plus the random floor."""
    g = realised_growth(pop)
    rows: list[dict] = []
    for o in origins:
        sub = g.filter(pl.col("origin_week") == o.origin_week)
        if sub.height == 0:  # non-empty P3 only (pre-registered)
            continue
        for construction in CONSTRUCTIONS:
            for method in gb.METHODS:
                v = score_vector(sub, construction, method)
                if v is None:
                    continue
                m = score_predictions(sub["g_true"].to_numpy(), v, np.ones(sub.height))
                n_eval = m.pop("n_eval")
                rows.append(
                    {
                        "origin_week": o.origin_week,
                        "construction": construction,
                        "method": method,
                        "n_styles": sub.height,
                        "n_eval": int(n_eval),
                        **m,
                    }
                )
    table = pl.DataFrame(rows)
    used = [o for o in origins if o.origin_week in set(table["origin_week"].to_list())]
    frame = g.select(
        "style_key",
        "origin_week",
        pl.col("g_true").alias("y_true"),
        pl.lit(1.0).alias("weight"),
    )
    floor = aggregate_random_floor_over_seeds(score_random_floor_per_origin(frame, used))
    floor = floor.rename({"method": "construction"}).with_columns(
        pl.lit(RANDOM_FLOOR_METHOD).alias("method"),
        pl.lit(RANDOM_FLOOR_METHOD).alias("construction"),
        pl.col("n_eval_set").alias("n_styles"),
    )
    floor = floor.select(table.columns)
    return pl.concat([table, floor], how="vertical")


def series(table: pl.DataFrame, construction: str, method: str, metric: str) -> pl.DataFrame:
    """`origin_week`, value for one (construction, method, metric), sorted by origin."""
    return (
        table.filter((pl.col("construction") == construction) & (pl.col("method") == method))
        .select("origin_week", pl.col(metric).alias("v"))
        .sort("origin_week")
    )


def paired(
    table: pl.DataFrame, a: tuple[str, str], b: tuple[str, str], metric: str, block: int
) -> dict[str, float]:
    """Paired per-origin difference a - b with a moving-block bootstrap interval."""
    sa, sb = series(table, *a, metric), series(table, *b, metric)
    j = sa.join(sb, on="origin_week", suffix="_b")
    d = (j["v"] - j["v_b"]).drop_nans().drop_nulls().to_list()
    mean, lo, hi = block_bootstrap_ci(d, block_size=block)
    return {
        "n_origins": len(d),
        "mean_a": float(j["v"].mean()),
        "mean_b": float(j["v_b"].mean()),
        "diff": mean,
        "ci_lo": lo,
        "ci_hi": hi,
    }


def summary_table(table: pl.DataFrame, block: int) -> pl.DataFrame:
    """Pooled mean and block-bootstrap interval per (construction, method, metric)."""
    rows = []
    for (construction, method), _ in table.group_by("construction", "method"):
        for metric in METRIC_KEYS:
            vals = series(table, construction, method, metric)["v"].drop_nans().to_list()
            if not vals:
                continue
            mean, lo, hi = block_bootstrap_ci(vals, block_size=block)
            rows.append(
                {
                    "construction": construction,
                    "method": method,
                    "metric": metric,
                    "n_origins": len(vals),
                    "mean": mean,
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
    return pl.DataFrame(rows).sort("construction", "method", "metric")


def decide(grid: pl.DataFrame, weekly: pl.DataFrame) -> pl.DataFrame:
    """The pre-registered primary, diagnostic and adoption rule."""
    rows = []
    cand = ("excess", MODEL)
    for label, table, block in (("grid_10", grid, 4), ("weekly", weekly, 13)):
        prim = paired(table, cand, ("ratio", SN), PRIMARY, block)
        gm = ("excess", GM)
        floor = (RANDOM_FLOOR_METHOD, RANDOM_FLOOR_METHOD)
        gm_vs_floor = paired(table, gm, floor, PRIMARY, block)
        near_floor = abs(gm_vs_floor["mean_a"] - gm_vs_floor["mean_b"]) <= FLOOR_TOL + TOL
        not_above = gm_vs_floor["ci_lo"] <= TOL
        rows.append(
            {
                "origin_set": label,
                "block": block,
                "primary_n_origins": prim["n_origins"],
                "redesigned_hit": prim["mean_a"],
                "seasonal_naive_ratio_hit": prim["mean_b"],
                "primary_diff": prim["diff"],
                "primary_ci_lo": prim["ci_lo"],
                "primary_ci_hi": prim["ci_hi"],
                "primary_pass": bool(prim["ci_lo"] > TOL),
                "diag_global_mean_excess_hit": gm_vs_floor["mean_a"],
                "diag_floor_hit": gm_vs_floor["mean_b"],
                "diag_diff_vs_floor": gm_vs_floor["diff"],
                "diag_ci_lo": gm_vs_floor["ci_lo"],
                "diag_ci_hi": gm_vs_floor["ci_hi"],
                "diag_within_0.10_of_floor": bool(near_floor),
                "diag_not_demonstrably_above_floor": bool(not_above),
                "diagnostic_pass": bool(near_floor and not_above),
            }
        )
    out = pl.DataFrame(rows)
    primary = bool(out["primary_pass"].all())
    diagnostic = bool(out["diagnostic_pass"].all())
    return out.with_columns(
        pl.lit(primary).alias("primary_passes_on_both"),
        pl.lit(diagnostic).alias("diagnostic_passes_on_both"),
        pl.lit(primary and diagnostic).alias("ADOPT_REDESIGNED_SCORE"),
    )


def lost_to_no_naive(frame: pl.DataFrame) -> pl.DataFrame:
    """Per origin: P2 size, P3 size, and the styles the seasonal-naive requirement drops."""
    p2 = gb.population(frame, "P2").group_by("origin_week").len().rename({"len": "n_p2"})
    p3n = p3(frame).group_by("origin_week").len().rename({"len": "n_p3"})
    return (
        p2.join(p3n, on="origin_week", how="left")
        .with_columns(pl.col("n_p3").fill_null(0))
        .with_columns((pl.col("n_p2") - pl.col("n_p3")).alias("n_dropped"))
        .sort("origin_week")
    )


def run_origin_set(
    frame: pl.DataFrame, origins: Sequence[Origin]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(per-origin table, population-size table) for one origin set."""
    return per_origin_table(p3(frame), origins), lost_to_no_naive(
        frame.filter(pl.col("origin_week").is_in([o.origin_week for o in origins]))
    )


def main() -> None:
    """Run the grid (10 evaluable origins) and weekly sets and write every table."""
    from nss.models.backtest import generate_origin_schedule
    from nss.models.backtest_v2 import identify_lightgbm_origins
    from nss.models.eval_power import LAST_WEEKLY_ORIGIN
    from nss.models.lightgbm_model import INITIAL_POOL_SIZE

    panel = pl.read_parquet("data/processed/style_week_panel.parquet")
    grid = generate_origin_schedule(panel)
    grid_origins = identify_lightgbm_origins(panel)
    weekly_origins = [
        o
        for o in generate_origin_schedule(panel, step_weeks=1)
        if grid_origins[0].origin_week <= o.origin_week <= LAST_WEEKLY_ORIGIN
    ]
    frame = gb.predictions_for_origins(
        panel, grid, [o.origin_week for o in weekly_origins], INITIAL_POOL_SIZE
    )
    grid_table, grid_sizes = run_origin_set(frame, grid_origins)
    weekly_table, weekly_sizes = run_origin_set(frame, weekly_origins)
    grid_table.write_csv(f"{OUT}/v3_redesign_per_origin_grid.csv")
    weekly_table.write_csv(f"{OUT}/v3_redesign_per_origin_weekly.csv")
    grid_sizes.write_csv(f"{OUT}/v3_redesign_population_grid.csv")
    weekly_sizes.write_csv(f"{OUT}/v3_redesign_population_weekly.csv")
    summary_table(grid_table, 4).write_csv(f"{OUT}/v3_redesign_summary_grid.csv")
    summary_table(weekly_table, 13).write_csv(f"{OUT}/v3_redesign_summary_weekly.csv")
    decision = decide(grid_table, weekly_table)
    decision.write_csv(f"{OUT}/v3_redesign_decision.csv")
    _print(decision, grid_table, weekly_table, grid_sizes, weekly_sizes)


def _print(
    decision: pl.DataFrame,
    grid: pl.DataFrame,
    weekly: pl.DataFrame,
    grid_sizes: pl.DataFrame,
    weekly_sizes: pl.DataFrame,
) -> None:
    def hit(table: pl.DataFrame, block: int) -> pl.DataFrame:
        s = summary_table(table, block).filter(pl.col("metric") == PRIMARY)
        return s.select("construction", "method", "n_origins", "mean", "ci_lo", "ci_hi")

    with pl.Config(
        tbl_rows=60, tbl_width_chars=200, tbl_formatting="ASCII_FULL", float_precision=3
    ):
        print("=== BASE-EFFECT DIAGNOSTIC (global_mean under `excess` vs floor) ===")
        print(
            decision.select(
                "origin_set",
                "diag_global_mean_excess_hit",
                "diag_floor_hit",
                "diag_diff_vs_floor",
                "diag_ci_lo",
                "diag_ci_hi",
                "diagnostic_pass",
            )
        )
        print("=== PRIMARY (redesigned excess model vs seasonal naive ratio) ===")
        print(
            decision.select(
                "origin_set",
                "primary_n_origins",
                "redesigned_hit",
                "seasonal_naive_ratio_hit",
                "primary_diff",
                "primary_ci_lo",
                "primary_ci_hi",
                "primary_pass",
            )
        )
        print("=== Hit@3-in-top20, grid (10 origins) ===")
        print(hit(grid, 4))
        print("=== Hit@3-in-top20, weekly ===")
        print(hit(weekly, 13))
        p2_med, p3_med = grid_sizes["n_p2"].median(), grid_sizes["n_p3"].median()
        print(f"grid populations: median P2 {p2_med}, P3 {p3_med}")
        print(f"weekly evaluable origins (non-empty P3): {weekly['origin_week'].n_unique()}")
    print("ADOPT_REDESIGNED_SCORE:", bool(decision["ADOPT_REDESIGNED_SCORE"][0]))


if __name__ == "__main__":
    main()
