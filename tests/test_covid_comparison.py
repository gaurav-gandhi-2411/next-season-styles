from __future__ import annotations

from datetime import date, timedelta

from nss.models.covid_comparison import (
    drop_covid_origins,
    horizon_overlaps_window,
    purge_overlapping_origins,
    qualify_verdict,
    verdict_for,
)

_START = date(2020, 3, 1)
_END = date(2020, 6, 30)


def test_horizon_overlap_boundaries_are_inclusive() -> None:
    # horizon = [o+1w, o+13w]. o+13w == 2020-03-01 -> touches window start -> overlap.
    last_overlapping_early = _START - timedelta(weeks=13)
    assert horizon_overlaps_window(last_overlapping_early, _START, _END)
    # One week earlier: horizon ends 2020-02-23, no overlap.
    assert not horizon_overlaps_window(last_overlapping_early - timedelta(weeks=1), _START, _END)
    # o+1w == 2020-06-30 -> touches window end -> overlap; one day later does not.
    late = _END - timedelta(weeks=1)
    assert horizon_overlaps_window(late, _START, _END)
    assert not horizon_overlaps_window(late + timedelta(days=1), _START, _END)


def test_horizon_overlap_matches_harness_is_covid_flag() -> None:
    import polars as pl

    from nss.models.backtest import generate_origin_schedule

    weeks = [date(2018, 9, 17) + timedelta(weeks=i) for i in range(107)]
    panel = pl.DataFrame({"week_start": weeks})
    for origin in generate_origin_schedule(panel):
        assert horizon_overlaps_window(origin.origin_week) == origin.is_covid


def test_drop_covid_origins_keeps_only_non_overlapping() -> None:
    origins = [date(2019, 11, 18), date(2019, 11, 25), date(2019, 12, 2), date(2020, 7, 6)]
    # 2019-11-25 + 13w = 2020-02-24 (no overlap); 2019-12-02 + 13w = 2020-03-02 (overlap);
    # 2020-07-06 + 1w = 2020-07-13 > window end (no overlap).
    assert drop_covid_origins(origins) == [date(2019, 11, 18), date(2019, 11, 25), date(2020, 7, 6)]


def test_purge_removes_only_target_window_overlaps() -> None:
    test_week = date(2019, 10, 21)
    weeks = [test_week + timedelta(weeks=k) for k in range(-14, 15)]
    kept = purge_overlapping_origins(weeks, test_week)
    # |o - t| < 13 weeks overlaps (target windows share weeks); >= 13 weeks is disjoint.
    assert all(abs(o - test_week) >= timedelta(weeks=13) for o in kept)
    assert test_week not in kept
    assert test_week + timedelta(weeks=12) not in kept
    assert test_week + timedelta(weeks=13) in kept
    assert test_week - timedelta(weeks=13) in kept
    assert len(kept) == len(weeks) - 25


def test_verdict_is_direction_aware_and_fails_closed() -> None:
    # diff = B - A, B excludes COVID data.
    assert verdict_for("precision_at_3", 0.1, 0.05, 0.2) == "covid_data_hurt"
    assert verdict_for("precision_at_3", -0.1, -0.2, -0.05) == "covid_data_helped"
    assert verdict_for("precision_at_3", 0.1, -0.05, 0.2) == "no_difference"
    # WMAPE: lower is better, so B's error going UP significantly means COVID data helped.
    assert verdict_for("wmape", 0.1, 0.05, 0.2) == "covid_data_helped"
    assert verdict_for("wmape", -0.1, -0.2, -0.05) == "covid_data_hurt"
    assert verdict_for("wmape", 0.0, 0.0, 0.0) == "no_difference"
    nan = float("nan")
    assert verdict_for("spearman_rho", nan, nan, nan) == "no_difference"


def test_verdict_ignores_float_noise_and_qualifies_weak_evidence() -> None:
    assert verdict_for("precision_at_10", 1e-17, 5e-18, 1.6e-17) == "no_difference"
    assert qualify_verdict("covid_data_hurt", "purged", 13) == "covid_data_hurt"
    assert qualify_verdict("covid_data_hurt", "purged", 5) == "covid_data_hurt_directional_only"
    assert qualify_verdict("covid_data_helped", "frozen_in_sample", 13) == (
        "covid_data_helped_in_sample_only"
    )
    assert qualify_verdict("no_difference", "frozen_in_sample", 5) == "no_difference"
