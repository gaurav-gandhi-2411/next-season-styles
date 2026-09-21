"""2b: the same embargoed model at a 1-week origin step, with effective sample size.

Rules are pre-registered in `reports/v3/PREREGISTRATION.md` (2b), committed before this ran. This is
the same model measured with more origins, not a stronger model: a weekly origin uses the model of
its 4-week grid block (`growth_backtest.predictions_for_origins`), because under the 16-week
embargo the training set of a weekly origin equals that of its block.

Adjacent weekly origins share 12 of their 13 forecast weeks, so the raw origin count overstates the
information. Two effective-sample-size estimators are reported for the paired difference series:

- `ess_ac`:   n / (1 + 2 * sum_{k=1..L} (1 - k/(L+1)) * rho_k), L = 12, floored at 1, capped at n;
- `ess_boot`: n * Var_iid(mean) / Var_block(mean), block-bootstrap variance of the mean.

    uv run python -m nss.models.eval_power
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import polars as pl

from nss.models import growth_backtest
from nss.models.backtest import (
    BASELINE_METHODS,
    BOOTSTRAP_N_RESAMPLES,
    BOOTSTRAP_SEED,
    Origin,
    build_predictions_frame,
    generate_origin_schedule,
    run_backtest,
)
from nss.models.backtest_v2 import paired_diff_table
from nss.models.lightgbm_model import INITIAL_POOL_SIZE
from nss.models.metrics import METRIC_KEYS, score_predictions
from nss.models.random_floor import (
    RANDOM_FLOOR_METHOD,
    aggregate_random_floor_over_seeds,
    score_random_floor_per_origin,
)

MODEL = growth_backtest.MODEL
PRIMARY = "hit_at_3_in_top20"
ACF_MAX_LAG = 12
HORIZON_BLOCK = 13  # weekly origins spanning the 13-week forecast horizon
GRID_BLOCK = 4
LAST_WEEKLY_ORIGIN = date(2020, 6, 22)  # last origin whose 13-week outcome fits in the panel
REPRO_PATH = "reports/tables/backtest_embargo_per_origin.csv"
OUT = "reports/tables"


def ess_ac(series: Sequence[float], max_lag: int = ACF_MAX_LAG) -> float:
    """Bartlett-weighted autocorrelation effective sample size, floored at 1, capped at n."""
    x = np.asarray([v for v in series if not np.isnan(v)], dtype=float)
    n = x.size
    if n < 3 or np.var(x) == 0.0:
        return float(n)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    lag = min(max_lag, n - 2)
    acc = 0.0
    for k in range(1, lag + 1):
        rho = float(np.dot(x[:-k], x[k:])) / denom
        acc += (1.0 - k / (lag + 1)) * rho
    return float(min(n, max(1.0, n / (1.0 + 2.0 * acc))))


def ess_boot(
    series: Sequence[float],
    block_size: int,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> float:
    """n * Var_iid(mean) / Var_block(mean), same moving-block resampling as `block_bootstrap_ci`."""
    x = np.asarray([v for v in series if not np.isnan(v)], dtype=float)
    n = x.size
    if n < 3 or np.var(x) == 0.0:
        return float(n)
    rng = np.random.default_rng(seed)
    blocks_needed = -(-n // block_size)
    max_start = max(n - block_size, 0)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        starts = rng.integers(0, max_start + 1, size=blocks_needed)
        means[i] = np.concatenate([x[s : s + block_size] for s in starts])[:n].mean()
    var_block = float(np.var(means, ddof=1))
    var_iid = float(np.var(x, ddof=1)) / n
    return float(min(n, max(1.0, n * var_iid / var_block))) if var_block > 0 else float(n)


def model_rows(frame: pl.DataFrame, origins: Sequence[Origin]) -> pl.DataFrame:
    """Per-origin model metrics on the FULL eval set (intensity ranking), backtest schema."""
    rows = []
    for o in origins:
        sub = frame.filter(pl.col("origin_week") == o.origin_week)
        metrics = score_predictions(
            sub["y_true"].to_numpy(),
            sub[f"y_pred_{MODEL}"].to_numpy(),
            sub["n_active_articles_level"].to_numpy(),
        )
        n_eval = metrics.pop("n_eval")
        rows.append(
            {
                "origin_week": o.origin_week,
                "method": MODEL,
                "has_52w_lag": o.has_52w_lag,
                "is_covid": o.is_covid,
                "n_eval_set": sub.height,
                "n_eval": int(n_eval),
                **metrics,
            }
        )
    return pl.DataFrame(rows)


def reproduction_gate(model_table: pl.DataFrame, grid_weeks: Sequence[date]) -> int:
    """The model's per-origin metrics at the grid origins must equal the committed table."""
    committed = pl.read_csv(REPRO_PATH, try_parse_dates=True).filter(pl.col("method") == MODEL)
    ours = model_table.filter(pl.col("origin_week").is_in(list(grid_weeks))).sort("origin_week")
    ref = committed.filter(pl.col("origin_week").is_in(list(grid_weeks))).sort("origin_week")
    assert ours.height == ref.height == len(grid_weeks), (ours.height, ref.height)
    for m in METRIC_KEYS:
        a, b = ours[m].to_numpy(), ref[m].to_numpy()
        if not np.allclose(a, b, rtol=0, atol=1e-9):
            raise SystemExit(f"reproduction gate failed on {m}: max diff {np.abs(a - b).max()}")
    return ours.height


def paired_with_ess(
    combined: pl.DataFrame, density: str, block: int, origins_used: int
) -> pl.DataFrame:
    """Paired model-minus-comparator rows (pooled) with n, ESS and the block length."""
    paired = paired_diff_table(
        combined,
        treatment_method=MODEL,
        baseline_methods=(*BASELINE_METHODS, RANDOM_FLOOR_METHOD),
        block_size=block,
    ).filter(pl.col("split") == "pooled")
    rows = []
    for r in paired.iter_rows(named=True):
        a = combined.filter(pl.col("method") == MODEL).sort("origin_week")
        b = combined.filter(pl.col("method") == r["method_b"]).sort("origin_week")
        j = a.join(b, on="origin_week", suffix="_b")
        diff = (j[r["metric"]] - j[f"{r['metric']}_b"]).drop_nulls().to_list()
        rows.append(
            {
                "density": density,
                "block_length": block,
                "comparator": r["method_b"],
                "metric": r["metric"],
                "n_origins": r["n_origins"],
                "ess_ac": ess_ac(diff),
                "ess_boot": ess_boot(diff, block),
                "mean_diff": r["mean_diff"],
                "ci_lo": r["diff_ci_low"],
                "ci_hi": r["diff_ci_high"],
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    """Run the weekly-origin backtest and write the side-by-side tables."""
    panel = pl.read_parquet("data/processed/style_week_panel.parquet")
    grid = generate_origin_schedule(panel)
    grid_test = grid[INITIAL_POOL_SIZE:]
    grid_weeks = [o.origin_week for o in grid_test]
    weekly = [
        o
        for o in generate_origin_schedule(panel, step_weeks=1)
        if grid_weeks[0] <= o.origin_week <= LAST_WEEKLY_ORIGIN
    ]
    weekly_weeks = [o.origin_week for o in weekly]
    assert set(grid_weeks) <= set(weekly_weeks) and len(weekly) == 48, len(weekly)

    frame = growth_backtest.predictions_for_origins(panel, grid, weekly_weeks, INITIAL_POOL_SIZE)
    model_table = model_rows(frame, weekly)
    n_gate = reproduction_gate(model_table, grid_weeks)
    print(f"reproduction gate OK at {n_gate} grid origins (1e-9)")

    baselines = run_backtest(panel, weekly)
    preds = build_predictions_frame(panel, weekly_weeks)
    floor = aggregate_random_floor_over_seeds(score_random_floor_per_origin(preds, weekly))
    cols = baselines.columns
    combined = pl.concat([baselines, model_table.select(cols), floor.select(cols)], how="vertical")
    combined.write_csv(f"{OUT}/v3_power_per_origin_weekly.csv")

    on_grid = combined.filter(pl.col("origin_week").is_in(grid_weeks))
    out = pl.concat(
        [
            paired_with_ess(on_grid, "grid_4w", GRID_BLOCK, len(grid_weeks)),
            paired_with_ess(combined, "weekly_1w", HORIZON_BLOCK, len(weekly_weeks)),
            paired_with_ess(combined, "weekly_1w_block4", GRID_BLOCK, len(weekly_weeks)),
        ]
    )
    out.write_csv(f"{OUT}/v3_power_paired.csv")
    with pl.Config(
        tbl_rows=40, tbl_width_chars=200, tbl_formatting="ASCII_FULL", float_precision=3
    ):
        print(
            out.filter((pl.col("metric") == PRIMARY) & (pl.col("comparator") == "seasonal_naive"))
        )
        print(out.filter(pl.col("metric") == PRIMARY).drop("metric"))


if __name__ == "__main__":
    main()
