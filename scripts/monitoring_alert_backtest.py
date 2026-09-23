"""How often the pre-stated coverage alert would have fired over the 48-origin backtest.

Applies `nss.prod.monitoring.coverage_alert` (4 most recent forecast weeks, pooled, band
[0.70, 0.90]) to the champion's backtest coverage (`reports/tables/phase_s_coverage_per_origin.csv`,
asymmetric CQR) at every point where 4 weeks of outcomes exist. Writes
`reports/tables/c1_coverage_alert_backtest.csv`.

    uv run python scripts/monitoring_alert_backtest.py
"""

from __future__ import annotations

import polars as pl

from nss.prod.monitoring import COVERAGE_BAND, COVERAGE_WINDOW_WEEKS, coverage_alert

SOURCE = "reports/tables/phase_s_coverage_per_origin.csv"
OUT = "reports/tables/c1_coverage_alert_backtest.csv"


def main() -> None:
    cov = pl.read_csv(SOURCE, try_parse_dates=True).sort("origin_week")
    per_week = cov.select(
        pl.col("origin_week").alias("as_of"),
        pl.col("n"),
        (pl.col("cov_cqr_asymmetric") * pl.col("n")).round(0).cast(pl.Int64).alias("n_covered"),
    )
    rows = []
    for i in range(COVERAGE_WINDOW_WEEKS, per_week.height + 1):
        res = coverage_alert(per_week.head(i))
        rows.append(
            {
                "window_end": per_week["as_of"][i - 1],
                **{k: res[k] for k in ("coverage", "status", "alert")},
            }
        )
    table = pl.DataFrame(rows)
    table.write_csv(OUT)
    n, fired = table.height, int(table["alert"].sum())
    print(f"band {COVERAGE_BAND}, window {COVERAGE_WINDOW_WEEKS}: {fired} of {n} windows alert "
          f"({fired / n:.0%}); low {int((table['status'] == 'low').sum())}, "
          f"high {int((table['status'] == 'high').sum())}")  # fmt: skip


if __name__ == "__main__":
    main()
