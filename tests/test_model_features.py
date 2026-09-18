from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from nss.features.model_features import (
    EWMA_HALF_LIFE_LONG_WEEKS,
    EWMA_HALF_LIFE_SHORT_WEEKS,
    LAG_WEEKS,
    build_features,
)

_WEEK0 = date(2018, 1, 1)
_N_WEEKS = 60


def _weeks(n: int, start: date = _WEEK0) -> list[date]:
    """`n` consecutive Monday week_start dates starting at `start`."""
    return [start + timedelta(weeks=i) for i in range(n)]


def _synthetic_panel(n_weeks: int = _N_WEEKS) -> pl.DataFrame:
    """Two style_keys ('A' in group G1/GG1, 'B' in group G2/GG2), `n_weeks` dense weeks each.

    Values are deterministic, distinct-per-week sequences (not constant) so that shuffling a
    row's value is actually detectable downstream -- a constant series would make the causality
    test vacuous even if shuffling were implemented correctly.
    """
    weeks = _weeks(n_weeks)
    first_seen = weeks[0]
    last_seen = weeks[-1]

    def _style(style_key: str, index_group: str, garment_group: str, base: float) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "style_key": [style_key] * n_weeks,
                "index_group_name": [index_group] * n_weeks,
                "product_type_name": ["Trousers"] * n_weeks,
                "garment_group_name": [garment_group] * n_weeks,
                "perceived_colour_master_name": ["Black"] * n_weeks,
                "graphical_appearance_name": ["Solid"] * n_weeks,
                "week_start": weeks,
                "first_week_seen": [first_seen] * n_weeks,
                "last_week_seen": [last_seen] * n_weeks,
                "units": [float(base + i) for i in range(n_weeks)],
                "n_active_articles": [float(1 + (i % 5)) for i in range(n_weeks)],
                "price_index": [1.0 + 0.01 * i for i in range(n_weeks)],
                "intensity_shrunk": [base + 0.5 * i for i in range(n_weeks)],
            }
        )

    return pl.concat(
        [
            _style("A", "G1", "GG1", base=10.0),
            _style("B", "G2", "GG2", base=100.0),
        ]
    )


def test_build_features_produces_one_row_per_style_at_a_valid_origin() -> None:
    """One row per (style_key, origin_week) for every style with a panel row at that origin."""
    panel = _synthetic_panel()
    origin_week = _weeks(_N_WEEKS)[40]

    out = build_features(panel, [origin_week])

    assert out.height == 2
    assert set(out["style_key"]) == {"A", "B"}
    assert set(out["origin_week"].to_list()) == {origin_week}


def test_build_features_excludes_styles_without_a_row_at_the_origin() -> None:
    """A style whose lifetime does not cover the requested origin is absent from the output."""
    panel = _synthetic_panel()
    # An origin_week far beyond both styles' last_week_seen -- no panel row exists there.
    future_origin = _weeks(_N_WEEKS)[-1] + timedelta(weeks=10)

    out = build_features(panel, [future_origin])

    assert out.height == 0


def test_lag_values_match_hand_computation() -> None:
    """lag_1 == value at origin; lag_L == value at origin - (L-1) weeks. Hand-verified for style A.

    Style A's intensity_shrunk at week index i is `10.0 + 0.5*i`. Origin = index 40.
    lag_1 = value at index 40 = 10 + 20.0 = 30.0.
    lag_4 = value at index 40-3=37 = 10 + 18.5 = 28.5.
    lag_13 = value at index 40-12=28 = 10 + 14.0 = 24.0.
    """
    panel = _synthetic_panel()
    origin_week = _weeks(_N_WEEKS)[40]

    out = build_features(panel, [origin_week]).filter(pl.col("style_key") == "A").row(0, named=True)

    assert out["lag_1"] == pytest.approx(10.0 + 0.5 * 40)
    assert out["lag_4"] == pytest.approx(10.0 + 0.5 * 37)
    assert out["lag_13"] == pytest.approx(10.0 + 0.5 * 28)


def test_lag_52_is_nullable_and_no_rows_dropped() -> None:
    """A style with < 52 weeks of history gets lag_52 = null, and is NOT dropped from output."""
    panel = _synthetic_panel(n_weeks=30)  # fewer than 52 weeks total
    origin_week = _weeks(30)[20]

    out = build_features(panel, [origin_week])

    assert out.height == 2  # both styles still present
    assert out["lag_52"].null_count() == 2  # neither has 52 prior weeks


def test_ewma_matches_hand_computed_recursive_formula() -> None:
    """EWMA (polars adjust=True convention) at half-life 4, hand-recomputed for a short series."""
    weeks = _weeks(5)
    panel = pl.DataFrame(
        {
            "style_key": ["A"] * 5,
            "index_group_name": ["G1"] * 5,
            "product_type_name": ["Trousers"] * 5,
            "garment_group_name": ["GG1"] * 5,
            "perceived_colour_master_name": ["Black"] * 5,
            "graphical_appearance_name": ["Solid"] * 5,
            "week_start": weeks,
            "first_week_seen": [weeks[0]] * 5,
            "last_week_seen": [weeks[-1]] * 5,
            "units": [1.0, 2.0, 3.0, 4.0, 5.0],
            "n_active_articles": [1.0, 1.0, 1.0, 1.0, 1.0],
            "price_index": [1.0, 1.0, 1.0, 1.0, 1.0],
            "intensity_shrunk": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    origin_week = weeks[4]

    out = build_features(panel, [origin_week]).row(0, named=True)

    # adjust=True EWMA: w_i = (1-alpha)^i, applied to the REVERSED series (most recent = i=0).
    alpha = 1 - 0.5 ** (1 / EWMA_HALF_LIFE_SHORT_WEEKS)
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    weights = [(1 - alpha) ** i for i in range(len(values))]
    expected = sum(w * v for w, v in zip(weights, reversed(values), strict=True)) / sum(weights)

    assert out["ewma_halflife_4w"] == pytest.approx(expected)


def test_slope_matches_two_point_finite_difference() -> None:
    """slope_4w == (value_now - value_4_weeks_ago) / 4, hand-verified."""
    panel = _synthetic_panel()
    origin_week = _weeks(_N_WEEKS)[40]

    out = build_features(panel, [origin_week]).filter(pl.col("style_key") == "A").row(0, named=True)

    # intensity_shrunk increases by 0.5/week -> slope should be exactly 0.5 regardless of window.
    assert out["slope_4w"] == pytest.approx(0.5)
    assert out["slope_13w"] == pytest.approx(0.5)


def test_weeks_since_first_seen_matches_hand_computation() -> None:
    panel = _synthetic_panel()
    origin_week = _weeks(_N_WEEKS)[40]

    out = build_features(panel, [origin_week]).filter(pl.col("style_key") == "A").row(0, named=True)

    assert out["weeks_since_first_seen"] == 40


def test_fourier_terms_match_hand_computation() -> None:
    panel = _synthetic_panel()
    origin_week = _weeks(_N_WEEKS)[40]
    doy = origin_week.timetuple().tm_yday

    out = build_features(panel, [origin_week]).filter(pl.col("style_key") == "A").row(0, named=True)

    expected_sin_1 = math.sin(2 * math.pi * 1 * doy / 365.25)
    expected_cos_1 = math.cos(2 * math.pi * 1 * doy / 365.25)
    assert out["fourier_sin_1"] == pytest.approx(expected_sin_1)
    assert out["fourier_cos_1"] == pytest.approx(expected_cos_1)


def test_share_of_parent_group_is_full_when_style_is_sole_member() -> None:
    """Each style is the sole member of its own index/garment group in the fixture -> share == 1.0
    once any trailing history exists (numerator == denominator, both equal the style's own trailing
    units, since no other style contributes to that group).
    """
    panel = _synthetic_panel()
    origin_week = _weeks(_N_WEEKS)[40]

    out = build_features(panel, [origin_week]).filter(pl.col("style_key") == "A").row(0, named=True)

    assert out["share_index_group"] == pytest.approx(1.0)
    assert out["share_garment_group"] == pytest.approx(1.0)


def test_share_of_parent_group_null_at_the_very_first_week() -> None:
    """No trailing group history exists at a style's own first observed week -> share is null."""
    panel = _synthetic_panel()
    origin_week = _weeks(_N_WEEKS)[0]

    out = build_features(panel, [origin_week]).filter(pl.col("style_key") == "A").row(0, named=True)

    assert out["share_index_group"] is None
    assert out["share_garment_group"] is None


def test_share_of_parent_group_reflects_relative_size_with_shared_group() -> None:
    """Two styles sharing one group: trailing share matches a hand-computed ratio.

    Style A: units = [10, 11, 12] across weeks 0-2. Style B: units = [1, 2, 3], same group.
    At origin = week index 2 (the 3rd week), trailing (strictly-before) totals are weeks 0-1 only:
    A_trailing = 10 + 11 = 21, B_trailing = 1 + 2 = 3, group_trailing = 24.
    share_A = 21/24 = 0.875, share_B = 3/24 = 0.125.
    """
    weeks = _weeks(3)
    panel = pl.DataFrame(
        {
            "style_key": ["A", "A", "A", "B", "B", "B"],
            "index_group_name": ["G"] * 6,
            "product_type_name": ["Trousers"] * 6,
            "garment_group_name": ["GG"] * 6,
            "perceived_colour_master_name": ["Black"] * 6,
            "graphical_appearance_name": ["Solid"] * 6,
            "week_start": [weeks[0], weeks[1], weeks[2], weeks[0], weeks[1], weeks[2]],
            "first_week_seen": [weeks[0]] * 6,
            "last_week_seen": [weeks[2]] * 6,
            "units": [10.0, 11.0, 12.0, 1.0, 2.0, 3.0],
            "n_active_articles": [1.0] * 6,
            "price_index": [1.0] * 6,
            "intensity_shrunk": [1.0] * 6,
        }
    )
    origin_week = weeks[2]

    out = build_features(panel, [origin_week])
    row_a = out.filter(pl.col("style_key") == "A").row(0, named=True)
    row_b = out.filter(pl.col("style_key") == "B").row(0, named=True)

    assert row_a["share_index_group"] == pytest.approx(21 / 24)
    assert row_b["share_index_group"] == pytest.approx(3 / 24)


def _shuffle_future_value_columns(
    panel: pl.DataFrame, origin_week: date, seed: int
) -> pl.DataFrame:
    """Randomly permute the VALUE columns (units, n_active_articles, price_index,
    intensity_shrunk) among all rows with `week_start > origin_week`, keeping every key/group
    column (style_key, week_start, first/last_week_seen, the 5 attribute cols) fixed in place --
    this preserves frame validity (no duplicate/missing keys) while genuinely scrambling every
    quantity any feature could possibly read from the future.
    """
    value_cols = ["units", "n_active_articles", "price_index", "intensity_shrunk"]
    past = panel.filter(pl.col("week_start") <= origin_week)
    future = panel.filter(pl.col("week_start") > origin_week)

    key_cols = [c for c in panel.columns if c not in value_cols]
    values = future.select(value_cols).sample(fraction=1.0, shuffle=True, seed=seed)
    shuffled_future = pl.concat([future.select(key_cols), values], how="horizontal_extend").select(
        panel.columns
    )
    return pl.concat([past, shuffled_future]).select(panel.columns)


def _shuffle_same_style_past_value_columns(
    panel: pl.DataFrame, style_key: str, origin_week: date, seed: int
) -> pl.DataFrame:
    """Randomly permute the VALUE columns among a single style's own PRE-origin rows (week_start
    <= origin_week), keeping every other row untouched -- the negative control.
    """
    value_cols = ["units", "n_active_articles", "price_index", "intensity_shrunk"]
    is_target = (pl.col("style_key") == style_key) & (pl.col("week_start") <= origin_week)
    target_rows = panel.filter(is_target)
    other_rows = panel.filter(~is_target)

    key_cols = [c for c in panel.columns if c not in value_cols]
    values = target_rows.select(value_cols).sample(fraction=1.0, shuffle=True, seed=seed)
    shuffled_target = pl.concat(
        [target_rows.select(key_cols), values], how="horizontal_extend"
    ).select(panel.columns)
    return pl.concat([other_rows, shuffled_target]).select(panel.columns)


def test_build_features_is_causally_safe() -> None:
    """Shuffling ALL future data leaves every feature unchanged (positive control: nothing reads
    the future). Shuffling SAME-STYLE pre-origin history DOES change lag/EWMA/slope features
    (negative control: proves the test is actually sensitive, not vacuously passing).
    """
    panel = _synthetic_panel()
    origin_week = _weeks(_N_WEEKS)[40]

    baseline = build_features(panel, [origin_week]).sort("style_key")

    # Positive control: scramble every value column for every row strictly after the origin,
    # across BOTH style_keys ("across all style_keys -- genuinely scramble the future").
    future_shuffled = _shuffle_future_value_columns(panel, origin_week, seed=42)
    future_shuffled_features = build_features(future_shuffled, [origin_week]).sort("style_key")

    assert baseline.equals(future_shuffled_features)

    # Negative control: scramble style A's own PRE-origin history -- lag/EWMA/slope features for
    # style A must change (they depend on that history); style B's features must NOT change.
    past_shuffled = _shuffle_same_style_past_value_columns(
        panel, style_key="A", origin_week=origin_week, seed=7
    )
    past_shuffled_features = build_features(past_shuffled, [origin_week]).sort("style_key")

    row_a_baseline = baseline.filter(pl.col("style_key") == "A").row(0, named=True)
    row_a_shuffled = past_shuffled_features.filter(pl.col("style_key") == "A").row(0, named=True)
    assert row_a_baseline["lag_4"] != row_a_shuffled["lag_4"]
    assert row_a_baseline["ewma_halflife_4w"] != row_a_shuffled["ewma_halflife_4w"]
    assert row_a_baseline["slope_4w"] != row_a_shuffled["slope_4w"]

    row_b_baseline = baseline.filter(pl.col("style_key") == "B").row(0, named=True)
    row_b_shuffled = past_shuffled_features.filter(pl.col("style_key") == "B").row(0, named=True)
    assert row_b_baseline == row_b_shuffled


def test_causally_safe_test_seed_actually_permutes() -> None:
    """Sanity check on the shuffle helpers themselves: with this fixture size and seed, the
    permutation is not (by chance) the identity -- otherwise the causality test's positive/negative
    controls could pass vacuously without the shuffle having done anything.
    """
    panel = _synthetic_panel()
    origin_week = _weeks(_N_WEEKS)[40]

    future_shuffled = _shuffle_future_value_columns(panel, origin_week, seed=42)
    future = panel.filter(pl.col("week_start") > origin_week).sort("style_key", "week_start")
    future_after = future_shuffled.filter(pl.col("week_start") > origin_week).sort(
        "style_key", "week_start"
    )
    assert not future.select("intensity_shrunk").equals(future_after.select("intensity_shrunk"))

    past_shuffled = _shuffle_same_style_past_value_columns(
        panel, style_key="A", origin_week=origin_week, seed=7
    )
    past_a = panel.filter(
        (pl.col("style_key") == "A") & (pl.col("week_start") <= origin_week)
    ).sort("week_start")
    past_a_after = past_shuffled.filter(
        (pl.col("style_key") == "A") & (pl.col("week_start") <= origin_week)
    ).sort("week_start")
    assert not past_a.select("intensity_shrunk").equals(past_a_after.select("intensity_shrunk"))


def test_lag_weeks_and_ewma_half_lives_are_the_documented_values() -> None:
    """Guard against silent drift of the documented conventions (module docstring)."""
    assert LAG_WEEKS == [1, 2, 4, 8, 13, 52]
    assert EWMA_HALF_LIFE_SHORT_WEEKS == 4
    assert EWMA_HALF_LIFE_LONG_WEEKS == 13
