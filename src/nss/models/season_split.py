"""H2: per-season split of the intensity backtest (evaluation only).

Rule pre-registered in `reports/v3/PREREGISTRATION.md` (H2), committed before this ran. Season of an
origin = calendar month of the midpoint of its 13-week target window (origin + 7 weeks), with
`final_forecast.SEASON_MONTHS`. Each season occurs once in the test period, so a cell is one
contiguous stretch of heavily overlapping origins: intervals describe that stretch, not the season
in general. Thin cells (fewer than 8 origins with both methods defined, or `ess_ac` < 5) are
flagged.

    uv run python -m nss.models.season_split
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from nss.models.backtest import block_bootstrap_ci
from nss.models.eval_power import ess_ac, ess_boot
from nss.models.final_forecast import SEASON_MONTHS
from nss.models.metrics import METRIC_KEYS
from nss.models.random_floor import RANDOM_FLOOR_METHOD

MODEL = "lightgbm"
COMPARATORS = (
    "seasonal_naive",
    "ewma_persistence",
    "parent_category_mean",
    "global_mean",
    RANDOM_FLOOR_METHOD,
)
BLOCK = 4
MIN_ORIGINS = 8
MIN_ESS = 5.0
SEASON_ORDER = ("autumn", "winter", "spring", "summer")  # chronological in the test period
SOURCE = "reports/tables/v3_power_per_origin_weekly.csv"
OUT = "reports/tables"


def season_of(origin: date, horizon_weeks: int = 13) -> str:
    """Season of the target window's midpoint (origin + 7 weeks)."""
    mid = origin + timedelta(weeks=(horizon_weeks + 1) // 2)
    for name, months in SEASON_MONTHS.items():
        if mid.month in months:
            return name
    raise ValueError(mid)


def paired_cell(
    table: pl.DataFrame, season: str, comparator: str, metric: str
) -> dict[str, object]:
    """One (season, comparator, metric) cell: means, paired difference, interval, ESS, flags."""
    sub = table.filter(pl.col("season") == season)
    a = sub.filter(pl.col("method") == MODEL).select("origin_week", pl.col(metric).alias("a"))
    b = sub.filter(pl.col("method") == comparator).select("origin_week", pl.col(metric).alias("b"))
    j = a.join(b, on="origin_week").sort("origin_week")
    ok = j.filter(pl.col("a").is_not_nan() & pl.col("b").is_not_nan()).drop_nulls()
    diff = (ok["a"] - ok["b"]).to_list()
    n_all, n_def = a.height, ok.height
    if n_def:
        mean, lo, hi = block_bootstrap_ci(diff, block_size=BLOCK)
        e_ac, e_boot = ess_ac(diff), ess_boot(diff, BLOCK)
    else:
        mean = lo = hi = e_ac = e_boot = float("nan")
    thin = n_def < MIN_ORIGINS or not (e_ac >= MIN_ESS)
    return {
        "season": season,
        "comparator": comparator,
        "metric": metric,
        "n_origins": n_all,
        "n_both_defined": n_def,
        "model_mean": float(ok["a"].mean()) if n_def else float("nan"),
        "comparator_mean": float(ok["b"].mean()) if n_def else float("nan"),
        "mean_diff": mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "ess_ac": e_ac,
        "ess_boot": e_boot,
        "THIN": bool(thin),
    }


def build(table: pl.DataFrame) -> pl.DataFrame:
    """The full per-season table."""
    rows = [
        paired_cell(table, s, c, m) for s in SEASON_ORDER for c in COMPARATORS for m in METRIC_KEYS
    ]
    return pl.DataFrame(rows)


def season_summary(table: pl.DataFrame) -> pl.DataFrame:
    """Origins per season, date range, and the model's own per-season means."""
    rows = []
    for s in SEASON_ORDER:
        sub = table.filter((pl.col("season") == s) & (pl.col("method") == MODEL))
        rows.append(
            {
                "season": s,
                "n_origins": sub.height,
                "first_origin": sub["origin_week"].min(),
                "last_origin": sub["origin_week"].max(),
                "n_covid_flagged": int(sub["is_covid"].sum()),
                **{f"model_{m}": float(sub[m].mean()) for m in METRIC_KEYS},
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    """Write the per-season tables and print the headline metric."""
    table = pl.read_csv(SOURCE, try_parse_dates=True).with_columns(
        pl.col("origin_week").map_elements(season_of, return_dtype=pl.String).alias("season")
    )
    cells = build(table)
    cells.write_csv(f"{OUT}/v3_season_split.csv")
    summary = season_summary(table)
    summary.write_csv(f"{OUT}/v3_season_summary.csv")
    hit = cells.filter(pl.col("metric") == "hit_at_3_in_top20")
    with pl.Config(
        tbl_rows=40, tbl_width_chars=220, tbl_formatting="ASCII_FULL", float_precision=3
    ):
        print(
            summary.select("season", "n_origins", "first_origin", "last_origin", "n_covid_flagged")
        )
        print(
            hit.select(
                "season",
                "comparator",
                "n_origins",
                "n_both_defined",
                "model_mean",
                "comparator_mean",
                "mean_diff",
                "ci_lo",
                "ci_hi",
                "ess_ac",
                "THIN",
            )
        )
    print(f"cells: {cells.height}, THIN: {int(cells['THIN'].sum())}")


if __name__ == "__main__":
    main()
