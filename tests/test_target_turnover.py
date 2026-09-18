from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from nss.models.backtest import Origin
from nss.models.target_turnover import (
    compute_turnover_table,
    overlap_fraction,
    ranked_style_keys,
)

_WEEK0 = date(2018, 1, 1)


def _weeks(n: int, start: date = _WEEK0) -> list[date]:
    """`n` consecutive Monday week_start dates starting at `start`."""
    return [start + timedelta(weeks=i) for i in range(n)]


def test_overlap_fraction_hand_computed() -> None:
    """`|a intersect b| / k` -- 2 of 3 shared elements, k=3 -> 2/3."""
    a = {"A", "B", "C"}
    b = {"B", "C", "D"}
    assert overlap_fraction(a, b, k=3) == 2 / 3
    assert overlap_fraction(a, a, k=3) == 1.0
    assert overlap_fraction(a, {"X", "Y", "Z"}, k=3) == 0.0


def test_ranked_style_keys_orders_descending_with_style_key_tiebreak() -> None:
    """3 styles, distinct forward-window means -> ranked descending; a tie breaks by style_key asc.

    Styles B, A, C have constant `units_per_active_article` 10, 20, 20 respectively across all
    weeks, so the forward mean for any origin equals that constant. Expected order: A and C tie at
    20 (A before C, alphabetical tiebreak), then B at 10.
    """
    weeks = _weeks(20)
    n = len(weeks)
    panel = pl.concat(
        [
            pl.DataFrame(
                {
                    "style_key": [style_key] * n,
                    "week_start": weeks,
                    "units_per_active_article": [value] * n,
                }
            )
            for style_key, value in [("B", 10.0), ("A", 20.0), ("C", 20.0)]
        ]
    )
    origin_week = weeks[5]

    ranked = ranked_style_keys(panel, origin_week, target_column="units_per_active_article")

    assert ranked == ["A", "C", "B"]


def test_ranked_style_keys_excludes_null_targets() -> None:
    """A style with an incomplete forward window (null target) is excluded from the ranking."""
    full_weeks = _weeks(20)
    short_weeks = _weeks(18)  # only 18 weeks total -> origin index 5 leaves just 12 future weeks
    panel = pl.concat(
        [
            pl.DataFrame(
                {
                    "style_key": ["FULL"] * len(full_weeks),
                    "week_start": full_weeks,
                    "units_per_active_article": [5.0] * len(full_weeks),
                }
            ),
            pl.DataFrame(
                {
                    "style_key": ["SHORT"] * len(short_weeks),
                    "week_start": short_weeks,
                    "units_per_active_article": [5.0] * len(short_weeks),
                }
            ),
        ]
    )
    origin_week = full_weeks[5]

    ranked = ranked_style_keys(panel, origin_week, target_column="units_per_active_article")

    assert ranked == ["FULL"]


def _constant_value_panel(
    style_values: dict[str, float], n_weeks: int, column: str = "units_per_active_article"
) -> pl.DataFrame:
    """One row per (style, week) for `n_weeks` consecutive weeks; each style's `column` value is
    constant across all weeks (so any forward window's mean equals that constant)."""
    weeks = _weeks(n_weeks)
    return pl.concat(
        [
            pl.DataFrame(
                {
                    "style_key": [style_key] * n_weeks,
                    "week_start": weeks,
                    column: [value] * n_weeks,
                }
            )
            for style_key, value in style_values.items()
        ]
    )


def test_compute_turnover_table_full_overlap_when_ranking_is_stable_over_time() -> None:
    """15 styles with time-invariant per-style values -> identical ranking at every origin, so the
    consecutive-pair overlap fraction is exactly 1.0 for every (target_column, top_k) cell."""
    style_values = {f"S{i}": float(100 - i) for i in range(15)}
    raw = _constant_value_panel(style_values, n_weeks=40, column="units_per_active_article")
    shrunk = _constant_value_panel(style_values, n_weeks=40, column="intensity_shrunk")
    panel = raw.join(
        shrunk.select("style_key", "week_start", "intensity_shrunk"),
        on=["style_key", "week_start"],
    )
    weeks = _weeks(40)
    origins = [
        Origin(origin_week=weeks[13], has_52w_lag=False, is_covid=False),
        Origin(origin_week=weeks[17], has_52w_lag=False, is_covid=False),
        Origin(origin_week=weeks[21], has_52w_lag=False, is_covid=False),
    ]

    table = compute_turnover_table(panel, origins)

    assert table.height == 4  # 2 target columns x 2 top_ks
    for row in table.iter_rows(named=True):
        assert row["n_pairs"] == 2
        assert row["mean_overlap"] == 1.0
        assert row["std_overlap"] == 0.0


def test_compute_turnover_table_hand_computed_overlap_when_ranking_reverses() -> None:
    """12 styles, 2 origins with DISJOINT forward windows whose per-style values completely
    reverse rank order between window 1 and window 2.

    Window 1 (origin index 20, weeks 21-33): style_i value = 100 - i (i=1..12) -> descending rank
    order S1 > S2 > ... > S12. Window 2 (origin index 60, weeks 61-73): style_i value = i -> rank
    order S12 > S11 > ... > S1 (fully reversed). Hand-computed expected overlap:
    - top-3: window1={S1,S2,S3}, window2={S12,S11,S10} -> 0/3 = 0.0.
    - top-10: window1={S1..S10}, window2={S12,S11,S10,S9,S8,S7,S6,S5,S4,S3} ->
      intersection={S3..S10} (8 styles) -> 8/10 = 0.8.
    """
    n_weeks = 74  # weeks 0..73, so window2 (weeks 61-73) fits fully inside the panel
    weeks = _weeks(n_weeks)
    style_keys = [f"S{i}" for i in range(1, 13)]

    rows: list[dict[str, object]] = []
    for style_index, style_key in enumerate(style_keys, start=1):
        for week_index, week_start in enumerate(weeks):
            if 21 <= week_index <= 33:
                value = 100.0 - style_index
            elif 61 <= week_index <= 73:
                value = float(style_index)
            else:
                value = 0.0
            rows.append(
                {
                    "style_key": style_key,
                    "week_start": week_start,
                    "units_per_active_article": value,
                    "intensity_shrunk": value,
                }
            )
    panel = pl.DataFrame(rows)
    origins = [
        Origin(origin_week=weeks[20], has_52w_lag=False, is_covid=False),
        Origin(origin_week=weeks[60], has_52w_lag=False, is_covid=False),
    ]

    table = compute_turnover_table(panel, origins)

    for target_column in ("units_per_active_article", "intensity_shrunk"):
        top3 = table.filter(
            (pl.col("target_column") == target_column) & (pl.col("top_k") == 3)
        ).row(0, named=True)
        top10 = table.filter(
            (pl.col("target_column") == target_column) & (pl.col("top_k") == 10)
        ).row(0, named=True)
        assert top3["n_pairs"] == 1
        assert top3["mean_overlap"] == 0.0
        assert top10["mean_overlap"] == 0.8
