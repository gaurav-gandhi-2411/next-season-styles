from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from nss.models.backtest import (
    BASELINE_METHODS,
    _add_trailing_baseline_means,
    _predict_seasonal_naive,
    block_bootstrap_ci,
    build_predictions_frame,
    generate_origin_schedule,
    run_backtest,
    summarize_backtest,
)
from nss.models.metrics import METRIC_KEYS

_WEEK0 = date(2018, 1, 1)


def _weeks(n: int, start: date = _WEEK0) -> list[date]:
    """`n` consecutive Monday week_start dates starting at `start`."""
    return [start + timedelta(weeks=i) for i in range(n)]


# ---------------------------------------------------------------------------
# generate_origin_schedule
# ---------------------------------------------------------------------------


def test_generate_origin_schedule_count_and_dates_hand_computed() -> None:
    """30 weeks (index 0-29), burn_in=5, horizon=5, step=4.

    first_origin = week[0] + 5w = week[5]; last_origin = week[29] - 5w = week[24].
    Origins: index 5, 9, 13, 17, 21 (index 25 would exceed last_origin=24).
    """
    weeks = _weeks(30)
    panel = pl.DataFrame({"week_start": weeks})

    origins = generate_origin_schedule(panel, step_weeks=4, burn_in_weeks=5, horizon_weeks=5)

    expected = [weeks[5], weeks[9], weeks[13], weeks[17], weeks[21]]
    assert [o.origin_week for o in origins] == expected


def test_generate_origin_schedule_matches_real_panel_extent() -> None:
    """Hand/script-verified against the actual style_week_panel.parquet extent (2018-09-17 ..
    2020-09-21) with the production defaults (step=4, burn_in=13, horizon=13): 20 origins,
    2018-12-17 .. 2020-06-01.
    """
    panel = pl.DataFrame({"week_start": [date(2018, 9, 17), date(2020, 9, 21)]})

    origins = generate_origin_schedule(panel)

    assert len(origins) == 20
    assert origins[0].origin_week == date(2018, 12, 17)
    assert origins[-1].origin_week == date(2020, 6, 1)


def test_generate_origin_schedule_is_covid_tag_hand_verified() -> None:
    """Script-verified: horizon = origin+1..origin+13 weeks, overlap with 2020-03-01..2020-06-30."""
    panel = pl.DataFrame({"week_start": [date(2018, 9, 17), date(2020, 9, 21)]})

    origins = generate_origin_schedule(panel)
    tags = {o.origin_week: o.is_covid for o in origins}

    # horizon 2019-11-25..2020-02-17 -- ends before the COVID window starts.
    assert tags[date(2019, 11, 18)] is False
    # horizon 2019-12-23..2020-03-16 -- overlaps the COVID window's start.
    assert tags[date(2019, 12, 16)] is True
    # horizon 2020-06-08..2020-08-31 -- overlaps the COVID window's end.
    assert tags[date(2020, 6, 1)] is True


def test_generate_origin_schedule_has_52w_lag_tag_hand_verified() -> None:
    """has_52w_lag = origin_week >= panel_start + 52 weeks = 2018-09-17 + 52w = 2019-09-16."""
    panel = pl.DataFrame({"week_start": [date(2018, 9, 17), date(2020, 9, 21)]})

    origins = generate_origin_schedule(panel)
    tags = {o.origin_week: o.has_52w_lag for o in origins}

    assert tags[date(2019, 8, 26)] is False
    assert tags[date(2019, 9, 23)] is True


# ---------------------------------------------------------------------------
# block_bootstrap_ci
# ---------------------------------------------------------------------------


def test_block_bootstrap_ci_reproducible_under_fixed_seed() -> None:
    values = [0.1, 0.2, 0.15, 0.3, 0.25, 0.2, 0.18, 0.22]

    result1 = block_bootstrap_ci(values, block_size=3, n_resamples=500, seed=42)
    result2 = block_bootstrap_ci(values, block_size=3, n_resamples=500, seed=42)

    assert result1 == result2


def test_block_bootstrap_ci_different_seed_can_give_a_different_ci() -> None:
    """Over a large-enough, less-discretized sample, different seeds should not coincidentally
    land on the exact same percentile boundaries (unlike the earlier tiny 8-value fixture, whose
    small discrete resample space made an exact coincidence plausible)."""
    values = [0.05 * i + 0.001 * (i % 7) for i in range(40)]

    _, lo1, hi1 = block_bootstrap_ci(values, block_size=4, n_resamples=500, seed=42)
    _, lo2, hi2 = block_bootstrap_ci(values, block_size=4, n_resamples=500, seed=7)

    assert (lo1, hi1) != (lo2, hi2)


def test_block_bootstrap_ci_point_estimate_matches_plain_mean() -> None:
    values = [1.0, 2.0, 3.0, 4.0]

    mean, ci_low, ci_high = block_bootstrap_ci(values, block_size=2, n_resamples=1000, seed=42)

    assert mean == pytest.approx(2.5)
    assert ci_low <= mean <= ci_high


def test_block_bootstrap_ci_drops_nan_values() -> None:
    values = [0.1, float("nan"), 0.2, 0.3]

    mean, _ci_low, _ci_high = block_bootstrap_ci(values, block_size=2, n_resamples=500, seed=42)

    assert mean == pytest.approx((0.1 + 0.2 + 0.3) / 3)


def test_block_bootstrap_ci_empty_is_nan() -> None:
    mean, ci_low, ci_high = block_bootstrap_ci([], block_size=2)

    assert math.isnan(mean)
    assert math.isnan(ci_low)
    assert math.isnan(ci_high)


# ---------------------------------------------------------------------------
# _predict_seasonal_naive
# ---------------------------------------------------------------------------


def test_predict_seasonal_naive_uses_the_correct_year_earlier_window() -> None:
    """65 weeks, units_per_active_article == week index. Origin = index 60.

    Lookback origin = index 60 - 52 = index 8; compute_forward_target's window from there is
    index 9..21 (13 weeks) = EXACTLY "the same 13 calendar weeks one year earlier" as origin's own
    window would be (origin+1..origin+13 = index 61..73) shifted back 52 weeks.
    mean(range(9, 22)) = sum(9..21) / 13 = 195 / 13 = 15.0.
    """
    n_weeks = 65
    weeks = _weeks(n_weeks)
    panel = pl.DataFrame(
        {
            "style_key": ["A"] * n_weeks,
            "week_start": weeks,
            "units_per_active_article": [float(i) for i in range(n_weeks)],
        }
    )
    origin_week = weeks[60]

    out = _predict_seasonal_naive(panel, origin_week).row(0, named=True)

    expected_mean = sum(range(9, 22)) / 13
    assert expected_mean == 15.0
    assert out["y_pred_seasonal_naive"] == pytest.approx(math.log1p(expected_mean))


def test_predict_seasonal_naive_null_when_year_earlier_window_incomplete() -> None:
    """Only 30 weeks of panel history -- an origin at index 20 needs weeks -31..-19 relative to
    week 0, i.e. entirely before the panel starts -> null prediction, not imputed."""
    n_weeks = 30
    weeks = _weeks(n_weeks)
    panel = pl.DataFrame(
        {
            "style_key": ["A"] * n_weeks,
            "week_start": weeks,
            "units_per_active_article": [float(i) for i in range(n_weeks)],
        }
    )
    origin_week = weeks[20]

    out = _predict_seasonal_naive(panel, origin_week).row(0, named=True)

    assert out["y_pred_seasonal_naive"] is None


# ---------------------------------------------------------------------------
# _add_trailing_baseline_means (parent_category_mean / global_mean) -- hand-computed + causality
# ---------------------------------------------------------------------------


def test_trailing_means_hand_computed_with_shared_group() -> None:
    """2 styles A, B sharing index_group 'G'. units_per_active_article: A=[10,11,12], B=[1,2,3].

    At origin = week index 2, trailing (strictly-before) values use only weeks 0-1:
    A_trailing_sum=21 (count 2), B_trailing_sum=3 (count 2).
    global_trailing_mean = (21+3)/(2+2) = 6.0 -- same for both styles (no exclusion for global).
    parent_category_mean (exclude self) for A = B's trailing mean = 3/2 = 1.5.
    parent_category_mean (exclude self) for B = A's trailing mean = 21/2 = 10.5.
    """
    weeks = _weeks(3)
    panel = pl.DataFrame(
        {
            "style_key": ["A", "A", "A", "B", "B", "B"],
            "index_group_name": ["G"] * 6,
            "week_start": [weeks[0], weeks[1], weeks[2], weeks[0], weeks[1], weeks[2]],
            "units_per_active_article": [10.0, 11.0, 12.0, 1.0, 2.0, 3.0],
        }
    )
    origin_week = weeks[2]

    out = _add_trailing_baseline_means(panel).filter(pl.col("week_start") == origin_week)
    row_a = out.filter(pl.col("style_key") == "A").row(0, named=True)
    row_b = out.filter(pl.col("style_key") == "B").row(0, named=True)

    assert row_a["y_pred_global_mean"] == pytest.approx(math.log1p(6.0))
    assert row_b["y_pred_global_mean"] == pytest.approx(math.log1p(6.0))
    assert row_a["y_pred_parent_category_mean"] == pytest.approx(math.log1p(1.5))
    assert row_b["y_pred_parent_category_mean"] == pytest.approx(math.log1p(10.5))


def test_trailing_means_null_at_the_very_first_week() -> None:
    """No trailing history exists at all at a style's own first observed week -> both null."""
    weeks = _weeks(3)
    panel = pl.DataFrame(
        {
            "style_key": ["A", "A", "A"],
            "index_group_name": ["G"] * 3,
            "week_start": weeks,
            "units_per_active_article": [10.0, 11.0, 12.0],
        }
    )

    out = (
        _add_trailing_baseline_means(panel)
        .filter(pl.col("week_start") == weeks[0])
        .row(0, named=True)
    )

    assert out["y_pred_global_mean"] is None
    assert out["y_pred_parent_category_mean"] is None


def _trailing_means_panel(n_weeks: int = 30) -> pl.DataFrame:
    """3 styles: A, B share index_group 'G1'; C is alone in 'G2'. Deterministic, distinct-per-week
    values so shuffling a row's value is actually detectable downstream.
    """
    weeks = _weeks(n_weeks)

    def _style(style_key: str, group: str, base: float) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "style_key": [style_key] * n_weeks,
                "index_group_name": [group] * n_weeks,
                "week_start": weeks,
                "units_per_active_article": [float(base + i) for i in range(n_weeks)],
            }
        )

    return pl.concat(
        [
            _style("A", "G1", base=10.0),
            _style("B", "G1", base=100.0),
            _style("C", "G2", base=1000.0),
        ]
    )


def _shuffle_future_units(panel: pl.DataFrame, origin_week: date, seed: int) -> pl.DataFrame:
    """Permute `units_per_active_article` among all rows with `week_start > origin_week`."""
    past = panel.filter(pl.col("week_start") <= origin_week)
    future = panel.filter(pl.col("week_start") > origin_week)
    key_cols = [c for c in panel.columns if c != "units_per_active_article"]
    values = future.select("units_per_active_article").sample(fraction=1.0, shuffle=True, seed=seed)
    shuffled_future = pl.concat([future.select(key_cols), values], how="horizontal_extend").select(
        panel.columns
    )
    return pl.concat([past, shuffled_future]).select(panel.columns)


def _shuffle_same_style_past_units(
    panel: pl.DataFrame, style_key: str, origin_week: date, seed: int
) -> pl.DataFrame:
    """Permute `units_per_active_article` among a single style's own pre-origin rows only."""
    is_target = (pl.col("style_key") == style_key) & (pl.col("week_start") <= origin_week)
    target_rows = panel.filter(is_target)
    other_rows = panel.filter(~is_target)
    key_cols = [c for c in panel.columns if c != "units_per_active_article"]
    values = target_rows.select("units_per_active_article").sample(
        fraction=1.0, shuffle=True, seed=seed
    )
    shuffled_target = pl.concat(
        [target_rows.select(key_cols), values], how="horizontal_extend"
    ).select(panel.columns)
    return pl.concat([other_rows, shuffled_target]).select(panel.columns)


def test_trailing_baseline_means_seed_actually_permutes() -> None:
    """Sanity check on the shuffle helpers: the permutation is not (by chance) the identity --
    otherwise the causality test below could pass vacuously."""
    panel = _trailing_means_panel()
    origin_week = _weeks(30)[20]

    future_shuffled = _shuffle_future_units(panel, origin_week, seed=42)
    future = panel.filter(pl.col("week_start") > origin_week).sort("style_key", "week_start")
    future_after = future_shuffled.filter(pl.col("week_start") > origin_week).sort(
        "style_key", "week_start"
    )
    assert not future.select("units_per_active_article").equals(
        future_after.select("units_per_active_article")
    )

    past_shuffled = _shuffle_same_style_past_units(panel, "A", origin_week, seed=7)
    past_a = panel.filter(
        (pl.col("style_key") == "A") & (pl.col("week_start") <= origin_week)
    ).sort("week_start")
    past_a_after = past_shuffled.filter(
        (pl.col("style_key") == "A") & (pl.col("week_start") <= origin_week)
    ).sort("week_start")
    assert not past_a.select("units_per_active_article").equals(
        past_a_after.select("units_per_active_article")
    )


def test_trailing_baseline_means_are_causally_safe() -> None:
    """Positive control: shuffling ALL future values leaves both trailing-mean baselines unchanged
    at the origin. Negative control: shuffling style A's own PRE-origin history changes A's own
    `global_mean` (A contributes to the global trailing sum) and B's `parent_category_mean` (B's
    exclude-self group mean, in this 2-style group, is exactly A's trailing mean) -- but must NOT
    change A's OWN `parent_category_mean` (it excludes A's own history by construction -- shuffling
    a quantity a formula doesn't read is the correct null result, not a test bug) or C's
    `parent_category_mean` (C is in a different index_group -- proves the group boundary is
    respected, not just "some cross-style effect exists").
    """
    panel = _trailing_means_panel()
    origin_week = _weeks(30)[20]

    baseline = (
        _add_trailing_baseline_means(panel)
        .filter(pl.col("week_start") == origin_week)
        .sort("style_key")
    )

    future_shuffled = _shuffle_future_units(panel, origin_week, seed=42)
    future_shuffled_out = (
        _add_trailing_baseline_means(future_shuffled)
        .filter(pl.col("week_start") == origin_week)
        .sort("style_key")
    )
    assert baseline.equals(future_shuffled_out)

    past_shuffled = _shuffle_same_style_past_units(panel, "A", origin_week, seed=7)
    past_shuffled_out = (
        _add_trailing_baseline_means(past_shuffled)
        .filter(pl.col("week_start") == origin_week)
        .sort("style_key")
    )

    row_a_baseline = baseline.filter(pl.col("style_key") == "A").row(0, named=True)
    row_a_shuffled = past_shuffled_out.filter(pl.col("style_key") == "A").row(0, named=True)
    assert row_a_baseline["y_pred_global_mean"] != row_a_shuffled["y_pred_global_mean"]
    # A's own parent_category_mean EXCLUDES A's own history by construction -- unaffected.
    assert (
        row_a_baseline["y_pred_parent_category_mean"]
        == row_a_shuffled["y_pred_parent_category_mean"]
    )

    row_b_baseline = baseline.filter(pl.col("style_key") == "B").row(0, named=True)
    row_b_shuffled = past_shuffled_out.filter(pl.col("style_key") == "B").row(0, named=True)
    assert (
        row_b_baseline["y_pred_parent_category_mean"]
        != row_b_shuffled["y_pred_parent_category_mean"]
    )

    row_c_baseline = baseline.filter(pl.col("style_key") == "C").row(0, named=True)
    row_c_shuffled = past_shuffled_out.filter(pl.col("style_key") == "C").row(0, named=True)
    assert (
        row_c_baseline["y_pred_parent_category_mean"]
        == row_c_shuffled["y_pred_parent_category_mean"]
    )


# ---------------------------------------------------------------------------
# build_predictions_frame -- eval-set membership (a) feature row + (b) non-null target
# ---------------------------------------------------------------------------

_N_WEEKS = 40


def _full_synthetic_panel(n_weeks: int = _N_WEEKS) -> pl.DataFrame:
    """2 styles (A, B), both present for the full `n_weeks`, with every column
    `build_features`/`compute_forward_target`/the baselines need."""
    weeks = _weeks(n_weeks)
    first_seen = weeks[0]
    last_seen = weeks[-1]

    def _style(style_key: str, index_group: str, base: float) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "style_key": [style_key] * n_weeks,
                "index_group_name": [index_group] * n_weeks,
                "product_type_name": ["Trousers"] * n_weeks,
                "garment_group_name": ["GG"] * n_weeks,
                "perceived_colour_master_name": ["Black"] * n_weeks,
                "graphical_appearance_name": ["Solid"] * n_weeks,
                "week_start": weeks,
                "first_week_seen": [first_seen] * n_weeks,
                "last_week_seen": [last_seen] * n_weeks,
                "units": [float(base + i) for i in range(n_weeks)],
                "n_active_articles": [float(1 + (i % 5)) for i in range(n_weeks)],
                "price_index": [1.0 + 0.01 * i for i in range(n_weeks)],
                "intensity_shrunk": [base + 0.5 * i for i in range(n_weeks)],
                "units_per_active_article": [base + 0.3 * i for i in range(n_weeks)],
            }
        )

    return pl.concat([_style("A", "G1", base=10.0), _style("B", "G1", base=100.0)])


def test_build_predictions_frame_includes_styles_with_full_forward_window() -> None:
    panel = _full_synthetic_panel()
    origin_week = _weeks(_N_WEEKS)[20]  # 19 forward weeks remain -- comfortably >= 13

    out = build_predictions_frame(panel, [origin_week])

    assert set(out["style_key"].to_list()) == {"A", "B"}
    assert out["y_true"].null_count() == 0


def test_build_predictions_frame_excludes_styles_without_full_forward_window() -> None:
    panel = _full_synthetic_panel()
    origin_week = _weeks(_N_WEEKS)[35]  # only 4 forward weeks remain -- short of 13

    out = build_predictions_frame(panel, [origin_week])

    assert out.height == 0


def test_build_predictions_frame_excludes_styles_not_yet_launched() -> None:
    weeks = _weeks(_N_WEEKS)
    early = _full_synthetic_panel()

    late_weeks = weeks[25:]
    late = pl.DataFrame(
        {
            "style_key": ["C"] * len(late_weeks),
            "index_group_name": ["G2"] * len(late_weeks),
            "product_type_name": ["Trousers"] * len(late_weeks),
            "garment_group_name": ["GG"] * len(late_weeks),
            "perceived_colour_master_name": ["Black"] * len(late_weeks),
            "graphical_appearance_name": ["Solid"] * len(late_weeks),
            "week_start": late_weeks,
            "first_week_seen": [late_weeks[0]] * len(late_weeks),
            "last_week_seen": [late_weeks[-1]] * len(late_weeks),
            "units": [500.0 + i for i in range(len(late_weeks))],
            "n_active_articles": [2.0] * len(late_weeks),
            "price_index": [1.0] * len(late_weeks),
            "intensity_shrunk": [500.0 + i for i in range(len(late_weeks))],
            "units_per_active_article": [500.0 + i for i in range(len(late_weeks))],
        }
    )
    panel = pl.concat([early, late])
    origin_week = weeks[10]  # before C's first_week_seen (index 25)

    out = build_predictions_frame(panel, [origin_week])

    assert "C" not in set(out["style_key"].to_list())


def test_build_predictions_frame_has_one_pred_column_per_baseline_method() -> None:
    panel = _full_synthetic_panel()
    origin_week = _weeks(_N_WEEKS)[20]

    out = build_predictions_frame(panel, [origin_week])

    for method in BASELINE_METHODS:
        assert f"y_pred_{method}" in out.columns


# ---------------------------------------------------------------------------
# run_backtest / summarize_backtest -- end-to-end smoke
# ---------------------------------------------------------------------------


def test_run_backtest_and_summarize_end_to_end() -> None:
    panel = _full_synthetic_panel(n_weeks=60)
    origins = generate_origin_schedule(panel, step_weeks=4, burn_in_weeks=13, horizon_weeks=13)
    assert len(origins) >= 2  # sanity: the fixture actually produces a usable schedule

    per_origin = run_backtest(panel, origins)
    assert per_origin.height == len(origins) * len(BASELINE_METHODS)
    for metric in METRIC_KEYS:
        assert metric in per_origin.columns

    summary = summarize_backtest(per_origin)
    assert summary.height == len(BASELINE_METHODS) * 3  # pooled / covid / non_covid
    for metric in METRIC_KEYS:
        assert f"{metric}_mean" in summary.columns
        assert f"{metric}_ci_low" in summary.columns
        assert f"{metric}_ci_high" in summary.columns
