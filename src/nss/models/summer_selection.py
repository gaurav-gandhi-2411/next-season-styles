"""Summer style selection under the final-three rules, with a 13-week training embargo.

The earlier Summer forecast (`seasonal_concept.summer_forecast`) trained on EVERY origin before
2020-06-01, including origins in March-May whose 13-week label windows run into the June-August
window being forecast (the same missing embargo found in the walk-forward backtest, see
`backtest_embargo_check`). Here the frozen config is trained only on origins whose label window ends
on or before the forecast origin (`o + 13w <= origin`), in memory, nothing saved.

Selection (rules fixed before running, same editorial constraints as `reselect_final_three`):
guards at the origin (mean active articles >= 10, price index >= 0.85, active >= 26 of 52 weeks),
then the category exclusion (intimates/underwear/nightwear) and the visual-ambiguity exclusion
(swimwear BOTTOMS, hosiery/leg base layers), then the first style by growth ratio (the emerging
table, as for autumn/winter). Swimwear TOPS
and summer dresses stay eligible.

Usage:
    uv run python -m nss.models.summer_selection
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from nss.features.style_panel import STYLE_KEY_COLS
from nss.models import diversity_forecast as dm
from nss.models import final_forecast as ff
from nss.models import reselect_final_three as rs
from nss.models.backtest import generate_origin_schedule
from nss.models.lightgbm_model import build_model_frame, feature_columns, train_lightgbm

ORIGIN = date(2020, 6, 1)
HORIZON_WEEKS = 13
OUT = Path("reports/tables/summer_selection_log.csv")
ALL_OUT = Path("reports/tables/forecast_all_styles_summer.csv")
TOP_N = 15


def main() -> None:
    """Train the embargoed model, rank guard-passing styles, apply exclusions, write the table."""
    panel = pl.read_parquet(ff.DEFAULT_PANEL_PATH)
    weeks = [o.origin_week for o in generate_origin_schedule(panel)]
    assert ORIGIN in weeks
    frame = build_model_frame(panel, weeks)
    cols = feature_columns(frame)
    train_weeks = [w for w in weeks if w + timedelta(weeks=HORIZON_WEEKS) <= ORIGIN]
    print(f"embargoed training origins: {len(train_weeks)} (last {train_weeks[-1]})")
    model = train_lightgbm(
        frame.filter(pl.col("origin_week").is_in(train_weeks)), ff.FINAL_MODEL_CONFIG, cols
    )
    ranking = ff.build_ranking_frame(panel, model, cols, forecast_origin=ORIGIN)
    at_origin = frame.filter(pl.col("origin_week") == ORIGIN).select(
        "style_key", pl.col("y_true").alias("y_true_log")
    )
    ranking = ranking.join(at_origin, on="style_key", how="left").with_columns(
        pl.col("y_true_log")
        .map_elements(lambda v: float(np.expm1(v)), return_dtype=pl.Float64)
        .alias("realised_intensity")
    )
    # Full ranking at the summer origin, for the closed loop (`concept_forecast`).
    ranking.write_csv(ALL_OUT)
    # The final-three rules applied to Summer: the EMERGING table (guards + median-intensity floor,
    # ranked by growth ratio), then the editorial exclusions, then the first eligible style.
    candidates = dm.build_t2_candidate_frame(panel, ranking, ORIGIN).join(
        ranking.select("style_key", "realised_intensity"), on="style_key", how="left"
    )
    articles = pl.read_csv(rs.ARTICLES_PATH)
    intimate = rs.intimate_product_types(articles)
    rows = []
    for rank, row in enumerate(candidates.to_dicts(), 1):
        why = rs.exclusion_reason(row, intimate)
        rows.append(
            {
                **{
                    k: row[k]
                    for k in (
                        "style_key",
                        *STYLE_KEY_COLS,
                        "predicted_intensity",
                        "trailing_13w_mean_intensity",
                        "growth_ratio",
                        "realised_intensity",
                    )
                },
                "emerging_rank": rank,
                "excluded": why,
            }
        )
        if rank >= 200:
            break
    table = pl.DataFrame(rows)
    table.write_csv(OUT)
    with pl.Config(tbl_rows=TOP_N + 5, tbl_width_chars=220, fmt_str_lengths=60):
        print(
            table.head(TOP_N).select(
                "emerging_rank",
                "style_key",
                "predicted_intensity",
                "growth_ratio",
                "realised_intensity",
                "excluded",
            )
        )
        print(
            "FIRST ELIGIBLE:",
            table.filter(pl.col("excluded").is_null())
            .head(3)
            .select(
                "emerging_rank",
                "style_key",
                "predicted_intensity",
                "growth_ratio",
                "realised_intensity",
            ),
        )


if __name__ == "__main__":
    main()
