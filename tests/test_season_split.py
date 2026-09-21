from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from nss.models import season_split as ss


def test_season_is_the_month_of_the_window_midpoint() -> None:
    # origin + 7 weeks: 2019-07-29 -> 2019-09-16 (autumn); 2020-01-13 -> 2020-03-02 (spring)
    assert ss.season_of(date(2019, 7, 29)) == "autumn"
    assert ss.season_of(date(2020, 1, 13)) == "spring"
    assert ss.season_of(date(2019, 10, 21)) == "winter"  # midpoint 2019-12-09
    assert ss.season_of(date(2020, 4, 6)) == "spring"  # midpoint 2020-05-25
    assert ss.season_of(date(2020, 6, 22)) == "summer"  # midpoint 2020-08-10


def test_every_weekly_origin_gets_a_season() -> None:
    d = date(2019, 7, 29)
    seen = {ss.season_of(d + timedelta(weeks=i)) for i in range(48)}
    assert seen == set(ss.SEASON_ORDER)


def _table(n: int, defined: int, season: str = "winter") -> pl.DataFrame:
    rows = []
    for i in range(n):
        wk = date(2020, 1, 6) + timedelta(weeks=i)
        rows.append(
            {
                "origin_week": wk,
                "method": "lightgbm",
                "season": season,
                "hit_at_3_in_top20": 0.5 + 0.02 * (i % 3),
            }
        )
        b = 0.4 + 0.03 * (i % 4) if i < defined else float("nan")
        rows.append(
            {
                "origin_week": wk,
                "method": "seasonal_naive",
                "season": season,
                "hit_at_3_in_top20": b,
            }
        )
    return pl.DataFrame(rows)


def test_a_cell_with_few_defined_origins_is_thin() -> None:
    cell = ss.paired_cell(_table(12, 5), "winter", "seasonal_naive", "hit_at_3_in_top20")
    assert cell["n_origins"] == 12 and cell["n_both_defined"] == 5 and cell["THIN"] is True


def test_undefined_comparator_gives_nan_not_zero() -> None:
    cell = ss.paired_cell(_table(12, 0), "winter", "seasonal_naive", "hit_at_3_in_top20")
    assert cell["n_both_defined"] == 0 and cell["mean_diff"] != cell["mean_diff"] and cell["THIN"]
