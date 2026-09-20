from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from nss.features.neighbourhood_features import (
    NEIGHBOURHOOD_FEATURE_COLS,
    add_neighbourhood_features,
    build_neighbourhood_features,
)

_WEEK0 = date(2018, 1, 1)
_N_WEEKS = 40
_VALUE_COLS = ["units", "n_active_articles", "price_index", "intensity_shrunk"]

# (index_group, product_type, garment_group, colour, appearance, slope, level offset)
# S0 is the focal style. S1: same product_type+garment_group, other colour (ptgg sibling).
# S2: same colour+appearance, other product type (colgfx sibling). S3: same index+garment group
# only (igg sibling). S4: unrelated in every neighbourhood. S5: same colour as S0 in the ptgg
# group but different appearance (NOT a ptgg sibling: excluded by colour).
_STYLES = {
    "S0": ("IG", "Knit", "GGK", "Beige", "Solid", 1.0, 10.0),
    "S1": ("IG", "Knit", "GGK", "Black", "Solid", 2.0, 20.0),
    "S2": ("IG", "Dress", "GGD", "Beige", "Solid", 3.0, 30.0),
    "S3": ("IG", "Top", "GGK", "White", "Dot", 4.0, 40.0),
    "S4": ("OTHER", "Shoe", "GGS", "Red", "Stripe", 5.0, 50.0),
    "S5": ("IG", "Knit", "GGK", "Beige", "Dot", 6.0, 60.0),
}


def _weeks(n: int = _N_WEEKS) -> list[date]:
    return [_WEEK0 + timedelta(weeks=i) for i in range(n)]


def _panel(n_weeks: int = _N_WEEKS) -> pl.DataFrame:
    """Six styles with distinct, linear-in-week intensity so every statistic is hand-computable."""
    weeks = _weeks(n_weeks)
    frames = []
    for key, (ig, pt, gg, col, gfx, slope, base) in _STYLES.items():
        frames.append(
            pl.DataFrame(
                {
                    "style_key": [key] * n_weeks,
                    "index_group_name": [ig] * n_weeks,
                    "product_type_name": [pt] * n_weeks,
                    "garment_group_name": [gg] * n_weeks,
                    "perceived_colour_master_name": [col] * n_weeks,
                    "graphical_appearance_name": [gfx] * n_weeks,
                    "week_start": weeks,
                    "first_week_seen": [weeks[0]] * n_weeks,
                    "last_week_seen": [weeks[-1]] * n_weeks,
                    "units": [float(base + i) for i in range(n_weeks)],
                    "n_active_articles": [float(1 + (i * int(slope)) % 7) for i in range(n_weeks)],
                    "price_index": [1.0 + 0.01 * i for i in range(n_weeks)],
                    "intensity_shrunk": [base + slope * i for i in range(n_weeks)],
                }
            )
        )
    return pl.concat(frames)


def _row(frame: pl.DataFrame, key: str) -> dict[str, object]:
    return frame.filter(pl.col("style_key") == key).row(0, named=True)


def test_columns_and_row_keys() -> None:
    origin = _weeks()[30]
    out = build_neighbourhood_features(_panel(), [origin])
    assert out.columns == ["style_key", "origin_week", *NEIGHBOURHOOD_FEATURE_COLS]
    assert out.height == len(_STYLES)
    assert set(out["origin_week"].to_list()) == {origin}
    assert len(NEIGHBOURHOOD_FEATURE_COLS) == 15


def test_sibling_sets_exclude_the_style_itself_and_follow_the_definitions() -> None:
    i = 30
    out = build_neighbourhood_features(_panel(), [_weeks()[i]])
    s0 = _row(out, "S0")
    # ptgg siblings of S0 = {S1} (S5 shares S0's colour -> excluded; S0 excluded as itself)
    assert s0["nb_ptgg_level_mean"] == pytest.approx(20.0 + 2.0 * i)
    assert s0["nb_ptgg_slope_4w"] == pytest.approx(2.0)
    assert s0["nb_ptgg_slope_13w"] == pytest.approx(2.0)
    assert s0["nb_ptgg_breadth"] == 1
    # colgfx siblings of S0 = {S2} (S5 has a different appearance)
    assert s0["nb_colgfx_level_mean"] == pytest.approx(30.0 + 3.0 * i)
    assert s0["nb_colgfx_slope_13w"] == pytest.approx(3.0)
    assert s0["nb_colgfx_breadth"] == 1
    # igg siblings of S0 = every other style with index_group IG and garment_group GGK: S1, S3, S5
    assert s0["nb_igg_level_mean"] == pytest.approx(
        ((20 + 2 * i) + (40 + 4 * i) + (60 + 6 * i)) / 3
    )
    assert s0["nb_igg_slope_4w"] == pytest.approx((2.0 + 4.0 + 6.0) / 3)
    assert s0["nb_igg_breadth"] == 3
    # ratio = own intensity / cohort mean
    assert s0["nb_ptgg_ratio"] == pytest.approx((10.0 + 1.0 * i) / (20.0 + 2.0 * i))


def test_style_with_no_siblings_gets_null_stats_and_zero_breadth() -> None:
    out = build_neighbourhood_features(_panel(), [_weeks()[30]])
    s4 = _row(out, "S4")
    for tag in ("ptgg", "colgfx", "igg"):
        assert s4[f"nb_{tag}_level_mean"] is None
        assert s4[f"nb_{tag}_slope_4w"] is None
        assert s4[f"nb_{tag}_ratio"] is None
        assert s4[f"nb_{tag}_breadth"] == 0


def test_own_values_never_enter_own_neighbourhood() -> None:
    """Perturbing ONLY the focal style's intensity leaves its own neighbourhood features unchanged
    and changes its siblings' (exclusion of the style itself, in both directions)."""
    panel = _panel()
    origin = _weeks()[30]
    bumped = panel.with_columns(
        pl.when(pl.col("style_key") == "S0")
        .then(pl.col("intensity_shrunk") + 1000.0)
        .otherwise(pl.col("intensity_shrunk"))
        .alias("intensity_shrunk")
    )
    base = build_neighbourhood_features(panel, [origin])
    after = build_neighbourhood_features(bumped, [origin])
    cols = [c for c in NEIGHBOURHOOD_FEATURE_COLS if not c.endswith("_ratio")]
    for c in cols:
        assert _row(base, "S0")[c] == pytest.approx(_row(after, "S0")[c])
    assert _row(base, "S1")["nb_ptgg_level_mean"] != _row(after, "S1")["nb_ptgg_level_mean"]


def test_breadth_counts_only_active_siblings_with_a_row_at_the_origin() -> None:
    panel = _panel()
    origin = _weeks()[30]
    # make S1 inactive (n_active_articles == 0) and drop S3's row at the origin week entirely
    panel = panel.with_columns(
        pl.when((pl.col("style_key") == "S1") & (pl.col("week_start") == origin))
        .then(0.0)
        .otherwise(pl.col("n_active_articles"))
        .alias("n_active_articles")
    ).filter(~((pl.col("style_key") == "S3") & (pl.col("week_start") == origin)))
    s0 = _row(build_neighbourhood_features(panel, [origin]), "S0")
    assert s0["nb_ptgg_breadth"] == 0  # S1 inactive
    assert s0["nb_igg_breadth"] == 1  # S3 absent, S1 inactive, only S5 counts


def _shuffle_future_values(panel: pl.DataFrame, origin: date, seed: int) -> pl.DataFrame:
    """Permute the value columns among rows with week_start > origin (keys stay in place)."""
    past = panel.filter(pl.col("week_start") <= origin)
    future = panel.filter(pl.col("week_start") > origin)
    key_cols = [c for c in panel.columns if c not in _VALUE_COLS]
    values = future.select(_VALUE_COLS).sample(fraction=1.0, shuffle=True, seed=seed)
    shuffled = pl.concat([future.select(key_cols), values], how="horizontal_extend")
    return pl.concat([past, shuffled.select(panel.columns)]).select(panel.columns)


def test_neighbourhood_features_are_causally_safe() -> None:
    """Shuffling every post-origin value leaves every neighbourhood feature unchanged; shuffling a
    SIBLING's pre-origin history does change the focal style's features (negative control)."""
    panel = _panel()
    origin = _weeks()[30]
    base = build_neighbourhood_features(panel, [origin]).sort("style_key")

    future_shuffled = _shuffle_future_values(panel, origin, seed=42)
    assert not future_shuffled.equals(panel)  # the shuffle really permuted something
    after = build_neighbourhood_features(future_shuffled, [origin]).sort("style_key")
    assert base.equals(after)

    # negative control: scramble S1's PRE-origin intensity (S1 is S0's ptgg sibling)
    is_s1_past = (pl.col("style_key") == "S1") & (pl.col("week_start") <= origin)
    s1_vals = panel.filter(is_s1_past).sort("week_start")["intensity_shrunk"].reverse()
    scrambled = pl.concat(
        [
            panel.filter(~is_s1_past),
            panel.filter(is_s1_past)
            .sort("week_start")
            .with_columns(s1_vals.alias("intensity_shrunk")),
        ]
    ).select(panel.columns)
    changed = build_neighbourhood_features(scrambled, [origin]).sort("style_key")
    assert _row(base, "S0")["nb_ptgg_slope_4w"] != _row(changed, "S0")["nb_ptgg_slope_4w"]
    # S4 has no siblings and is untouched
    assert _row(base, "S4") == _row(changed, "S4")


def test_features_at_earlier_origin_ignore_later_origins_in_the_request() -> None:
    """A row's values do not depend on which OTHER origin weeks were requested."""
    panel = _panel()
    a, b = _weeks()[25], _weeks()[35]
    solo = build_neighbourhood_features(panel, [a]).sort("style_key")
    both = build_neighbourhood_features(panel, [a, b]).filter(pl.col("origin_week") == a)
    assert solo.equals(both.sort("style_key"))


def test_add_neighbourhood_features_is_additive_and_preserves_order() -> None:
    panel = _panel()
    origin = _weeks()[30]
    frame = pl.DataFrame(
        {
            "style_key": ["S3", "S0", "S1"],
            "origin_week": [origin] * 3,
            "y_true": [1.0, 2.0, 3.0],
        }
    )
    joined = add_neighbourhood_features(frame, panel)
    assert joined.columns == [*frame.columns, *NEIGHBOURHOOD_FEATURE_COLS]
    assert joined["style_key"].to_list() == ["S3", "S0", "S1"]
    assert joined.select(frame.columns).equals(frame)
    assert joined.filter(pl.col("style_key") == "S0")["nb_ptgg_breadth"].item() == 1


def test_missing_column_raises() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        build_neighbourhood_features(_panel().drop("intensity_shrunk"), [_weeks()[30]])
