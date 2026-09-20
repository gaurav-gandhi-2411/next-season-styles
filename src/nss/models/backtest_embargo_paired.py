"""Post-embargo PAIRED comparison of LightGBM against every baseline (task P1).

`backtest_embargo_check` showed the shipped walk-forward trained on labels overlapping the test
window and that Hit@3-in-top20 falls from 0.722 to 0.528 once a 13-week gap is enforced. It did not
say whether the embargoed model still beats the baselines: the pre-embargo paired differences
(`backtest_paired_diff.csv`) were computed with the leaked model. This module re-does the paired
comparison with the EMBARGOED model as the treatment, on the same 12 origins, using the project's
standard paired block bootstrap (block 4, 2,000 resamples, seed 42; `paired_diff_table`).

Baselines (seasonal naive, EWMA persistence, global mean, parent-category mean) are unaffected by
the embargo: they use only data before the origin. The random floor is included for reference.
Seasonal naive is undefined at origins without a 52-week lag (2 of the 12), so its paired rows use
10 origins although `n_origins` reads 12.

Outputs (new files; nothing committed is overwritten):
    reports/tables/backtest_embargo_summary.csv   per-method means with block-bootstrap CIs
    reports/tables/backtest_embargo_paired_diff.csv   embargoed LightGBM minus each baseline

Usage:
    uv run python -m nss.models.backtest_embargo_paired
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from nss.models.backtest import (
    build_predictions_frame,
    generate_origin_schedule,
    run_backtest,
    summarize_backtest,
)
from nss.models.backtest_embargo_check import (
    DEFAULT_PANEL_PATH,
    run_embargoed_walk_forward,
)
from nss.models.backtest_v2 import identify_lightgbm_origins, paired_diff_table
from nss.models.lightgbm_model import METHOD_NAME, run_lightgbm_walk_forward
from nss.models.random_floor import (
    aggregate_random_floor_over_seeds,
    score_random_floor_per_origin,
)

SUMMARY_OUT = Path("reports/tables/backtest_embargo_summary.csv")
PAIRED_OUT = Path("reports/tables/backtest_embargo_paired_diff.csv")
PER_ORIGIN_OUT = Path("reports/tables/backtest_embargo_per_origin.csv")
BASELINES = ("seasonal_naive", "ewma_persistence", "global_mean", "parent_category_mean")


def main() -> None:
    """Run baselines + embargoed LightGBM on the shared origins; write summary and paired diffs."""
    panel = pl.read_parquet(DEFAULT_PANEL_PATH)
    origins = identify_lightgbm_origins(panel)
    baseline = run_backtest(panel, origins)
    _shipped, config, _grid = run_lightgbm_walk_forward(panel, generate_origin_schedule(panel))
    embargoed, skipped = run_embargoed_walk_forward(panel, generate_origin_schedule(panel), config)
    assert not skipped, f"embargoed arm skipped origins {skipped}"
    predictions = build_predictions_frame(panel, [o.origin_week for o in origins])
    floor = aggregate_random_floor_over_seeds(score_random_floor_per_origin(predictions, origins))
    combined = pl.concat([baseline, embargoed.select(baseline.columns), floor], how="vertical")
    assert combined.filter(pl.col("method") == METHOD_NAME).height == len(origins)
    combined.write_csv(PER_ORIGIN_OUT)
    summary = summarize_backtest(combined)
    summary.write_csv(SUMMARY_OUT)
    paired = paired_diff_table(combined, baseline_methods=BASELINES)
    paired.write_csv(PAIRED_OUT)
    with pl.Config(
        tbl_rows=80, tbl_width_chars=200, tbl_formatting="ASCII_FULL", float_precision=3
    ):
        keep = ["hit_at_3_in_top20", "hit_at_3_in_top10", "ndcg_at_10", "spearman_rho", "wmape"]
        print(paired.filter((pl.col("split") == "pooled") & pl.col("metric").is_in(keep)))


if __name__ == "__main__":
    main()
