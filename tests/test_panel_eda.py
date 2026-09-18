from __future__ import annotations

import datetime

import polars as pl
import pytest

import nss.viz.panel_eda as panel_eda_module
from nss.features.style_panel import STYLE_KEY_COLS
from nss.viz.panel_eda import (
    build_intensity_comparison_table,
    detect_stockout_signature,
    mean_price_index_by_style,
    mean_shrunk_intensity_by_style,
    stockout_signature_summary,
)


def _weekly_series(style_key: str, units: list[int]) -> pl.DataFrame:
    """Build a synthetic single-style weekly units series starting 2018-01-01 (a Monday)."""
    start = datetime.date(2018, 1, 1)
    weeks = [start + datetime.timedelta(weeks=i) for i in range(len(units))]
    return pl.DataFrame(
        {"style_key": [style_key] * len(units), "week_start": weeks, "units": units}
    )


def test_detect_stockout_signature_finds_known_drop_and_recover() -> None:
    """A >70% week-over-week drop followed by a >70% recovery (vs. the dropped week) is flagged.

    Series: 100, 100, 20, 100, 100.
    - Week index 2 (units=20): prev=100, 20 < 0.3*100=30 -> drop.
    - next=100 > 1.7*20=34 -> recovery. So index 2 must be flagged True.
    - No other index in this series should match.
    """
    panel = _weekly_series("MATCH", [100, 100, 20, 100, 100])
    flagged = detect_stockout_signature(panel)

    matches = flagged.filter(pl.col("is_stockout_signature"))
    assert matches.height == 1
    assert matches["units"].item() == 20
    assert matches["week_start"].item() == datetime.date(2018, 1, 1) + datetime.timedelta(weeks=2)


def test_detect_stockout_signature_drop_without_recovery_is_not_flagged() -> None:
    """A qualifying drop that does NOT recover by >70% the following week is not flagged."""
    # Week index 2 (units=20) drops from 100 (20 < 30), but week index 3 (units=25) does not
    # clear 1.7 * 20 = 34, so this is a drop without a qualifying recovery.
    panel = _weekly_series("NO_RECOVER", [100, 100, 20, 25, 100])
    flagged = detect_stockout_signature(panel)

    assert flagged["is_stockout_signature"].sum() == 0


def test_detect_stockout_signature_ignores_zero_prior_week() -> None:
    """A transition FROM a zero-unit prior week is never a 'drop' (prev_units > 0 required)."""
    # Week index 1 (units=5) has prev_units=0 -- restocking, not a drop, regardless of what
    # follows. Confirms the prev_units > 0 guard in the drop definition.
    panel = _weekly_series("ZERO_PRIOR", [0, 5, 5, 5, 5])
    flagged = detect_stockout_signature(panel)

    week1 = flagged.filter(pl.col("units") == 5).sort("week_start").head(1)
    assert week1["is_stockout_signature"].item() is False


def test_detect_stockout_signature_boundary_rows_are_ineligible() -> None:
    """The first and last week of a style's panel window have no prior/next week -- ineligible."""
    panel = _weekly_series("BOUNDS", [100, 20, 100])
    flagged = detect_stockout_signature(panel).sort("week_start")

    assert flagged["eligible"].to_list() == [False, True, False]


def test_stockout_signature_summary_reports_numerator_denominator_rate() -> None:
    """Summary counts match a hand-computed tally across two synthetic styles."""
    panel = pl.concat(
        [
            _weekly_series("MATCH", [100, 100, 20, 100, 100]),  # 1 match out of 3 eligible rows
            _weekly_series("NO_RECOVER", [100, 100, 20, 25, 100]),  # 0 matches, 3 eligible rows
        ]
    )
    flagged = detect_stockout_signature(panel)
    summary = stockout_signature_summary(flagged)

    assert summary["n_matched"] == 1
    assert summary["n_eligible"] == 6
    assert summary["rate_pct"] == 100.0 * 1 / 6


_DUMMY_STYLE_COLS = dict.fromkeys(STYLE_KEY_COLS, "x")


def _intensity_comparison_fixture() -> pl.DataFrame:
    """3 styles, 1 active week each, with deliberately distinct rankings on each metric.

    units (lifetime_units) descending:            A=100, B=50, C=10
    units_per_active_article (raw intensity) desc: C=20,  B=8,  A=5
    intensity_shrunk descending:                   A=9,   B=7,  C=6
    price_index: A=1.2, B=0.8, C=null (simulates an early-life style, <52 weeks history).
    """
    week = datetime.date(2018, 1, 1)
    rows = [
        {
            "style_key": "A",
            "units": 100,
            "n_active_articles": 1,
            "units_per_active_article": 5.0,
            "intensity_shrunk": 9.0,
            "price_index": 1.2,
        },
        {
            "style_key": "B",
            "units": 50,
            "n_active_articles": 1,
            "units_per_active_article": 8.0,
            "intensity_shrunk": 7.0,
            "price_index": 0.8,
        },
        {
            "style_key": "C",
            "units": 10,
            "n_active_articles": 1,
            "units_per_active_article": 20.0,
            "intensity_shrunk": 6.0,
            "price_index": None,
        },
    ]
    return pl.DataFrame([{**row, "week_start": week, **_DUMMY_STYLE_COLS} for row in rows])


def test_mean_shrunk_intensity_by_style_ranks_descending() -> None:
    """Mean intensity_shrunk per style, sorted descending, matches the fixture's known order."""
    panel = _intensity_comparison_fixture()
    result = mean_shrunk_intensity_by_style(panel)
    assert result["style_key"].to_list() == ["A", "B", "C"]
    assert result["mean_intensity_shrunk"].to_list() == [9.0, 7.0, 6.0]


def test_mean_price_index_by_style_excludes_null_rows() -> None:
    """A style with no non-null price_index observations is absent, not zero or NaN."""
    panel = _intensity_comparison_fixture()
    result = mean_price_index_by_style(panel)
    assert set(result["style_key"]) == {"A", "B"}  # C excluded: its only row has price_index=null
    assert result.filter(pl.col("style_key") == "A")["mean_price_index"].item() == 1.2
    assert result.filter(pl.col("style_key") == "B")["mean_price_index"].item() == 0.8


def test_build_intensity_comparison_table_ranks_and_overlaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rankings, union-of-top-N row set, price_index annotation, and pairwise overlaps.

    TOP_N is monkeypatched to 2 so the 3-style fixture actually produces PARTIAL top-N lists
    (with the real TOP_N=20, all 3 styles trivially appear in every list and every overlap count
    would trivially be 3 -- not a meaningful test of the overlap logic itself). With TOP_N=2:
    - lifetime top2 (units):        {A, B}  (C=10 excluded)
    - raw intensity top2:           {C, B}  (A=5 excluded)
    - shrunk intensity top2:        {A, B}  (C=6 excluded)
    => raw_vs_shrunk = |{C,B} & {A,B}| = 1 (B only)
       raw_vs_volume = |{C,B} & {A,B}| = 1 (B only)
       shrunk_vs_volume = |{A,B} & {A,B}| = 2 (A and B)
    C is still present in the final table (it's in the raw-intensity top2) with null
    rank_lifetime_units / rank_shrunk_intensity and a null mean_price_index (never observed).
    """
    monkeypatch.setattr(panel_eda_module, "TOP_N", 2)
    panel = _intensity_comparison_fixture()

    table, overlaps = build_intensity_comparison_table(panel)

    assert overlaps == {"raw_vs_shrunk": 1, "raw_vs_volume": 1, "shrunk_vs_volume": 2}
    assert set(table["style_key"]) == {"A", "B", "C"}

    row_c = table.filter(pl.col("style_key") == "C").row(0, named=True)
    assert row_c["rank_raw_intensity"] == 1
    assert row_c["rank_lifetime_units"] is None
    assert row_c["rank_shrunk_intensity"] is None
    assert row_c["mean_price_index"] is None

    row_a = table.filter(pl.col("style_key") == "A").row(0, named=True)
    assert row_a["rank_lifetime_units"] == 1
    assert row_a["rank_raw_intensity"] is None  # A=5.0 is 3rd/last in raw intensity, excluded
    assert row_a["rank_shrunk_intensity"] == 1
    assert row_a["mean_price_index"] == 1.2
