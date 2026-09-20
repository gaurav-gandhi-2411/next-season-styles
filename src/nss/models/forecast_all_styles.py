"""Persist the frozen model's forecast for EVERY eligible style (task N8's lookup table).

The closed loop ("score a generated concept through the same predictor") needs the forecast and
rank of an arbitrary style_key, but only the top-10 leaderboards were ever persisted. This module
re-derives the full ranking with the SAME frozen code path `final_forecast.main` uses
(`FINAL_MODEL_CONFIG`, wide training origins, seed 42, deterministic) -- no new modelling choice, no
tuning -- and writes it once, so `concept_forecast` can look styles up instead of retraining.

Sanity check built in: every row of the committed T1/T2 leaderboards must be reproduced (predicted
intensity within 1e-4 relative), otherwise this raises rather than persist a table that disagrees
with the shipped forecast.

Usage:
    uv run python -m nss.models.forecast_all_styles
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from nss.models import final_forecast as ff

OUT_PATH = Path("reports/tables/forecast_all_styles.csv")
COMMITTED_TABLES = (
    Path("reports/tables/top_styles_t1_incumbent.csv"),
    Path("reports/tables/top_styles_t2_emerging.csv"),
)


def main() -> None:
    """Train the frozen model, write the full ranking, and verify it against `top_styles.csv`."""
    panel = pl.read_parquet(ff.DEFAULT_PANEL_PATH)
    ff.verify_forecast_origin(panel)
    model, _frame, columns = ff.train_final_model(panel)
    ranking = ff.build_ranking_frame(panel, model, columns)
    ranking = ranking.with_columns(
        (pl.col("guard1_pass") & pl.col("guard2_pass") & pl.col("guard3_pass")).alias(
            "all_guards_pass"
        )
    )
    # Verify against EVERY persisted D3a-fixed leaderboard row (T1 + T2): same style, same value.
    # (`top_styles.csv` is the pre-D3a raw top-10 and is deliberately not the reference.)
    committed = pl.concat(
        [pl.read_csv(p).select("style_key", "predicted_intensity") for p in COMMITTED_TABLES]
    )
    got = dict(ranking.select("style_key", "predicted_intensity").iter_rows())
    worst = 0.0
    for key, want in committed.iter_rows():
        rel = abs(got[key] - want) / want
        worst = max(worst, rel)
        if rel > 1e-4:
            raise RuntimeError(f"re-derived forecast for {key} disagrees: {got[key]} vs {want}")
    print(f"verified {committed.height} leaderboard rows; worst relative diff {worst:.2e}")
    ranking.write_csv(OUT_PATH)
    print(f"wrote {OUT_PATH}: {ranking.height} eligible styles")


if __name__ == "__main__":
    main()
