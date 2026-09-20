from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from nss.features.price_features import (
    FULL_PRICE_THRESHOLD,
    PRICE_FEATURE_COLS,
    WINDOW_WEEKS,
    add_price_features,
    build_price_features,
)

_WEEK0 = date(2018, 1, 1)
_N_WEEKS = 70
_VALUE_COLS = ["mean_price", "price_index", "intensity_shrunk"]
_ORIGIN_INDEX = 60
_ELASTICITY = -1.5


def _weeks(n: int = _N_WEEKS) -> list[date]:
    return [_WEEK0 + timedelta(weeks=i) for i in range(n)]


def _style_rows(key: str, n: int, price_fn, index_fn, intensity_fn) -> pl.DataFrame:
    idx = range(n)
    return pl.DataFrame(
        {
            "style_key": [key] * n,
            "week_start": _weeks(n),
            "mean_price": [price_fn(i) for i in idx],
            "price_index": [index_fn(i) for i in idx],
            "intensity_shrunk": [intensity_fn(i) for i in idx],
        }
    )


def _price(i: int) -> float:
    return 0.05 * (1.0 + 0.1 * math.sin(i))


def _panel(n: int = _N_WEEKS) -> pl.DataFrame:
    """S0: intensity = 3 * price^-1.5 exactly (elasticity -1.5), price_index dips every 4th week.
    S1: flat price (no elasticity defined). S2: price_index null throughout."""
    return pl.concat(
        [
            _style_rows(
                "S0",
                n,
                _price,
                lambda i: 0.8 if i % 4 == 0 else 1.0,
                lambda i: 3.0 * _price(i) ** _ELASTICITY,
            ),
            _style_rows("S1", n, lambda i: 0.04, lambda i: 1.0, lambda i: 2.0 + 0.1 * (i % 5)),
            _style_rows("S2", n, _price, lambda i: None, lambda i: 1.0 + 0.2 * (i % 3)),
        ]
    )


def _row(frame: pl.DataFrame, key: str) -> dict[str, object]:
    return frame.filter(pl.col("style_key") == key).row(0, named=True)


def test_columns_and_row_keys() -> None:
    origin = _weeks()[_ORIGIN_INDEX]
    out = build_price_features(_panel(), [origin])
    assert out.columns == ["style_key", "origin_week", *PRICE_FEATURE_COLS]
    assert out.height == 3
    assert set(out["origin_week"].to_list()) == {origin}
    assert len(PRICE_FEATURE_COLS) == 6


def test_elasticity_recovers_the_constructed_slope_and_is_null_without_price_variation() -> None:
    out = build_price_features(_panel(), [_weeks()[_ORIGIN_INDEX]])
    assert _row(out, "S0")["pr_elasticity_13w"] == pytest.approx(_ELASTICITY, abs=1e-9)
    assert _row(out, "S1")["pr_elasticity_13w"] is None  # flat price
    assert _row(out, "S2")["pr_elasticity_13w"] is not None  # elasticity needs no price_index


def test_discount_and_full_price_features_match_a_python_reference() -> None:
    t = _ORIGIN_INDEX
    window = range(t - WINDOW_WEEKS + 1, t + 1)
    idx = [0.8 if i % 4 == 0 else 1.0 for i in window]
    intens = [3.0 * _price(i) ** _ELASTICITY for i in window]
    full = [v >= FULL_PRICE_THRESHOLD for v in idx]
    full_int = sum(x for x, f in zip(intens, full, strict=True) if f) / sum(full)
    overall = sum(intens) / len(intens)
    row = _row(build_price_features(_panel(), [_weeks()[t]]), "S0")
    assert row["pr_discount_freq_13w"] == pytest.approx(1 - sum(full) / len(full))
    assert row["pr_discount_depth_13w"] == pytest.approx(sum(max(0.0, 1 - v) for v in idx) / 13)
    assert row["pr_full_price_intensity_13w"] == pytest.approx(full_int)
    assert row["pr_full_price_holding_ratio_13w"] == pytest.approx(full_int / overall)
    assert row["pr_price_index_slope_4w"] == pytest.approx(
        ((0.8 if t % 4 == 0 else 1.0) - (0.8 if (t - 4) % 4 == 0 else 1.0)) / 4
    )


def test_style_without_price_index_gets_null_discount_features() -> None:
    row = _row(build_price_features(_panel(), [_weeks()[_ORIGIN_INDEX]]), "S2")
    for col in PRICE_FEATURE_COLS:
        if col != "pr_elasticity_13w":
            assert row[col] is None


def test_early_origin_uses_partial_window_but_needs_enough_points() -> None:
    early = build_price_features(_panel(), [_weeks()[3]])  # only 4 weeks of history
    assert _row(early, "S0")["pr_elasticity_13w"] is None  # < MIN_ELASTICITY_POINTS


def _shuffle_future_values(panel: pl.DataFrame, origin: date, seed: int) -> pl.DataFrame:
    """Permute each value column independently among rows with week_start > origin."""
    past = panel.filter(pl.col("week_start") <= origin)
    future = panel.filter(pl.col("week_start") > origin)
    key_cols = [c for c in panel.columns if c not in _VALUE_COLS]
    values = future.select(_VALUE_COLS).sample(fraction=1.0, shuffle=True, seed=seed)
    shuffled = pl.concat([future.select(key_cols), values], how="horizontal_extend")
    return pl.concat([past, shuffled.select(panel.columns)]).select(panel.columns)


def _invariant_to_future(**leak: int) -> bool:
    """The causality check: shuffling every post-origin value leaves every feature unchanged."""
    panel = _panel()
    origin = _weeks()[_ORIGIN_INDEX]
    base = build_price_features(panel, [origin], **leak).sort("style_key")
    shuffled = _shuffle_future_values(panel, origin, seed=42)
    assert not shuffled.equals(panel)  # the shuffle really permuted something
    after = build_price_features(shuffled, [origin], **leak).sort("style_key")
    return base.equals(after)


def test_price_features_are_causally_safe() -> None:
    assert _invariant_to_future()


def test_negative_control_leaky_window_fails_the_causality_check() -> None:
    """A window that reaches into the future (the deliberately leaky variant) must FAIL the very
    same check the real features pass -- otherwise the check could not catch a real leak."""
    assert not _invariant_to_future(_window_lead_weeks=3)


def test_negative_control_pre_origin_history_does_change_the_features() -> None:
    panel = _panel()
    origin = _weeks()[_ORIGIN_INDEX]
    base = build_price_features(panel, [origin]).sort("style_key")
    bumped = panel.with_columns(
        pl.when(
            (pl.col("style_key") == "S0") & (pl.col("week_start") == origin - timedelta(weeks=2))
        )
        .then(pl.col("price_index") * 0.5)
        .otherwise(pl.col("price_index"))
        .alias("price_index")
    )
    changed = build_price_features(bumped, [origin]).sort("style_key")
    assert _row(base, "S0")["pr_discount_depth_13w"] != _row(changed, "S0")["pr_discount_depth_13w"]
    assert _row(base, "S1") == _row(changed, "S1")  # other styles untouched


def test_features_do_not_depend_on_which_other_origins_are_requested() -> None:
    panel = _panel()
    a, b = _weeks()[50], _weeks()[65]
    solo = build_price_features(panel, [a]).sort("style_key")
    both = build_price_features(panel, [a, b]).filter(pl.col("origin_week") == a)
    assert solo.equals(both.sort("style_key"))


def test_add_price_features_is_additive_and_preserves_order() -> None:
    panel = _panel()
    origin = _weeks()[_ORIGIN_INDEX]
    frame = pl.DataFrame(
        {"style_key": ["S2", "S0", "S1"], "origin_week": [origin] * 3, "y_true": [1.0, 2.0, 3.0]}
    )
    joined = add_price_features(frame, panel)
    assert joined.columns == [*frame.columns, *PRICE_FEATURE_COLS]
    assert joined["style_key"].to_list() == ["S2", "S0", "S1"]
    assert joined.select(frame.columns).equals(frame)


def test_missing_column_raises() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        build_price_features(_panel().drop("mean_price"), [_weeks()[_ORIGIN_INDEX]])
