from __future__ import annotations

import datetime

import polars as pl

from nss.viz.panel_eda import detect_stockout_signature, stockout_signature_summary


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
