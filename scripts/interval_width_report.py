"""Relative width of the champion's calibrated interval, for the model card.

Relative width = median interval width / median point forecast, both in raw intensity (units per
active article per week), over the 48 embargoed weekly origins. Reported for every style and for
the top-20 shortlist (the model's own top 20 by point forecast at each origin), with the
shortlist's coverage. Reads the Section S artifacts; writes
`reports/tables/c1_interval_width.csv`.

    uv run python scripts/interval_width_report.py
"""

from __future__ import annotations

import numpy as np
import polars as pl

from nss.models.phase_a_measure import weekly_origins
from nss.models.phase_b import load_champion
from nss.models.phase_s_calibration import calibrate, calibration_origins, quantile_path

OUT = "reports/tables/c1_interval_width.csv"


def main() -> None:
    panel = pl.read_parquet("data/processed/style_week_panel.parquet")
    _, weekly = weekly_origins(panel)
    test_weeks = [o.origin_week for o in weekly]
    weeks = [*(o.origin_week for o in calibration_origins(panel, test_weeks[0])), *test_weeks]
    lo = pl.read_parquet(quantile_path(10, weeks)).rename({"y_pred": "q10"})
    hi = pl.read_parquet(quantile_path(90, weeks)).select(
        "style_key", "origin_week", pl.col("y_pred").alias("q90")
    )
    rows = calibrate(lo.join(hi, on=["style_key", "origin_week"]), weekly)
    j = rows.join(
        load_champion().select("style_key", "origin_week", "y_pred"),
        on=["style_key", "origin_week"],
    ).with_columns(
        pl.col("y_pred").rank("ordinal", descending=True).over("origin_week").alias("rank")
    )
    out = []
    for population, sub in (("all_styles", j), ("top20_shortlist", j.filter(pl.col("rank") <= 20))):
        pred = np.expm1(sub["y_pred"].to_numpy())
        for interval in ("uncalibrated", "cqr_asymmetric"):
            width = sub[f"wraw_{interval}"].to_numpy()
            out.append(
                {
                    "population": population,
                    "interval": interval,
                    "n_rows": sub.height,
                    "median_width": float(np.median(width)),
                    "median_point_forecast": float(np.median(pred)),
                    "relative_width": float(np.median(width) / np.median(pred)),
                    "coverage": float(sub[f"cov_{interval}"].mean()),
                }
            )
    table = pl.DataFrame(out)
    table.write_csv(OUT)
    with pl.Config(tbl_width_chars=200, float_precision=3):
        print(table)


if __name__ == "__main__":
    main()
