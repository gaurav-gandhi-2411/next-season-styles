from __future__ import annotations

import math
import random
from datetime import date, timedelta

import polars as pl
import pytest

from nss.features.signal_features import (
    LEAD_FEATURE_COLS,
    SIGNAL_FEATURE_COLS,
    add_signal_features,
    build_signal_features,
    usable_terms,
)

_WEEK0 = date(2018, 1, 1)  # a Monday
_N = 80
_ORIGIN_INDEX = 70


def _weeks(n: int = _N) -> list[date]:
    return [_WEEK0 + timedelta(weeks=i) for i in range(n)]


def _series(i: int) -> float:
    """A rising, wobbling reading; strictly positive."""
    return 20.0 + 1.5 * i + 4.0 * math.sin(i)


def _weekly(scale: float = 1.0) -> pl.DataFrame:
    """T0: normal; T1: mostly zero (unusable)."""
    weeks = _weeks()
    t0 = pl.DataFrame(
        {"term": ["T0"] * _N, "week_start": weeks, "value": [scale * _series(i) for i in range(_N)]}
    )
    t1 = pl.DataFrame(
        {
            "term": ["T1"] * _N,
            "week_start": weeks,
            "value": [scale * (5.0 if i % 5 == 0 else 0.0) for i in range(_N)],
        }
    )
    return pl.concat([t0, t1])


def _panel() -> pl.DataFrame:
    weeks = _weeks()
    rows = []
    for key in ("S0", "S1", "S2", "S3"):
        for i, w in enumerate(weeks):
            rows.append(
                {
                    "style_key": key,
                    "week_start": w,
                    "intensity_shrunk": 3.0 + 0.2 * i + (1.0 if key == "S1" else 0.0),
                }
            )
    return pl.DataFrame(rows)


def _terms() -> pl.DataFrame:
    """S0, S1 share T0; S2 has the unusable T1; S3 is unmapped."""
    return pl.DataFrame(
        {"style_key": ["S0", "S1", "S2", "S3"], "term": ["T0", "T0", "T1", None]},
        schema={"style_key": pl.String, "term": pl.String},
    )


def _row(frame: pl.DataFrame, key: str) -> dict[str, object]:
    return frame.filter(pl.col("style_key") == key).row(0, named=True)


def _ln_ratio_of_mean(x: list[float], t: int, a: int, b: int) -> float:
    """ln(mean of the `a` values ending at t / mean of the `a` values ending at t-b)."""
    num = sum(x[t - a + 1 : t + 1]) / a
    den = sum(x[t - b - a + 1 : t - b + 1]) / a
    return math.log(num / den)


def test_columns_and_row_keys() -> None:
    origin = _weeks()[_ORIGIN_INDEX]
    out = build_signal_features(_panel(), _terms(), _weekly(), [origin])
    assert out.columns == ["style_key", "origin_week", *SIGNAL_FEATURE_COLS]
    assert sorted(out["style_key"].to_list()) == ["S0", "S1", "S2", "S3"]


def test_values_match_a_python_reference() -> None:
    x = [_series(i) for i in range(_N)]
    t = _ORIGIN_INDEX
    out = build_signal_features(_panel(), _terms(), _weekly(), [_weeks()[t]])
    r = _row(out, "S0")
    m4 = sum(x[t - 3 : t + 1]) / 4
    m13 = sum(x[t - 12 : t + 1]) / 13
    b52 = sum(x[t - 51 : t + 1]) / 52
    assert r["sg_level_rel13"] == pytest.approx(math.log(m4 / m13))
    assert r["sg_slope_4w"] == pytest.approx(_ln_ratio_of_mean(x, t, 4, 4) / 4)
    assert r["sg_slope_13w"] == pytest.approx(_ln_ratio_of_mean(x, t, 4, 13) / 13)
    assert r["sg_rel_52w"] == pytest.approx(math.log(m4 / b52))
    for lag in (2, 4, 8):
        assert r[f"sg_lead{lag}"] == pytest.approx(_ln_ratio_of_mean(x, t - lag, 4, 4) / 4)


def test_divergence_is_signal_slope_minus_sales_slope() -> None:
    t = _ORIGIN_INDEX
    x = [_series(i) for i in range(_N)]
    s = [3.0 + 0.2 * i for i in range(_N)]  # S0's intensity_shrunk

    def s4(k: int) -> float:
        return sum(s[k - 3 : k + 1]) / 4

    out = build_signal_features(_panel(), _terms(), _weekly(), [_weeks()[t]])
    sales_slope4 = (math.log1p(s4(t)) - math.log1p(s4(t - 4))) / 4
    expected = _ln_ratio_of_mean(x, t, 4, 4) / 4 - sales_slope4
    assert _row(out, "S0")["sg_div_4w"] == pytest.approx(expected)


def test_styles_sharing_a_term_share_signal_but_not_divergence() -> None:
    out = build_signal_features(_panel(), _terms(), _weekly(), [_weeks()[_ORIGIN_INDEX]])
    s0, s1 = _row(out, "S0"), _row(out, "S1")
    assert s0["sg_slope_4w"] == s1["sg_slope_4w"]  # same term, same signal
    assert s0["sg_div_4w"] != s1["sg_div_4w"]  # different sales history, different divergence


def test_unmapped_and_unusable_styles_get_all_null_features() -> None:
    out = build_signal_features(_panel(), _terms(), _weekly(), [_weeks()[_ORIGIN_INDEX]])
    for key in ("S2", "S3"):  # S2: sparse term; S3: no term
        assert all(_row(out, key)[c] is None for c in SIGNAL_FEATURE_COLS)
    usable = usable_terms(_weekly()).filter(pl.col("usable"))["term"].to_list()
    assert usable == ["T0"]


def test_early_origin_leaves_long_windows_null_not_imputed() -> None:
    out = build_signal_features(_panel(), _terms(), _weekly(), [_weeks()[10]])
    r = _row(out, "S0")
    assert r["sg_slope_4w"] is not None
    assert r["sg_slope_13w"] is None  # needs 4 + 13 weeks of history
    assert r["sg_lead8"] is None
    assert r["sg_rel_52w"] is None  # fewer than 26 weeks of baseline


def _shuffle_future(frame: pl.DataFrame, origin: date, value_col: str, seed: int) -> pl.DataFrame:
    """Permute `value_col` among rows dated after `origin`, within each group (first column)."""
    rng = random.Random(seed)
    parts = []
    for _, g in frame.group_by(frame.columns[0], maintain_order=True):
        past = g.filter(pl.col("week_start") <= origin)
        fut = g.filter(pl.col("week_start") > origin)
        vals = fut[value_col].to_list()
        rng.shuffle(vals)
        parts.append(pl.concat([past, fut.with_columns(pl.Series(value_col, vals))]))
    return pl.concat(parts)


def _invariant_to_future(**leak: int) -> bool:
    panel, weekly, origin = _panel(), _weekly(), _weeks()[_ORIGIN_INDEX]
    base = build_signal_features(panel, _terms(), weekly, [origin], **leak).sort("style_key")
    panel_s = _shuffle_future(panel, origin, "intensity_shrunk", 42)
    weekly_s = _shuffle_future(weekly, origin, "value", 42)
    assert not weekly_s.equals(weekly) and not panel_s.equals(panel)  # really permuted
    after = build_signal_features(panel_s, _terms(), weekly_s, [origin], **leak).sort("style_key")
    return base.equals(after)


def test_signal_features_are_causally_safe() -> None:
    """Shuffling every signal and sales value dated after the origin changes nothing."""
    assert _invariant_to_future()


def test_negative_control_leaky_signal_fails_the_causality_check() -> None:
    """A signal slid into the future must FAIL the same check, or it could not catch a leak."""
    assert not _invariant_to_future(_lead_weeks=3)


def test_negative_control_pre_origin_history_does_change_the_features() -> None:
    weekly, origin = _weekly(), _weeks()[_ORIGIN_INDEX]
    base = build_signal_features(_panel(), _terms(), weekly, [origin]).sort("style_key")
    bumped = weekly.with_columns(
        pl.when((pl.col("term") == "T0") & (pl.col("week_start") == origin - timedelta(weeks=2)))
        .then(pl.col("value") * 3.0)
        .otherwise(pl.col("value"))
        .alias("value")
    )
    changed = build_signal_features(_panel(), _terms(), bumped, [origin]).sort("style_key")
    assert _row(base, "S0")["sg_slope_4w"] != _row(changed, "S0")["sg_slope_4w"]
    assert _row(base, "S2") == _row(changed, "S2")  # a style on another term is untouched


def test_features_are_invariant_to_rescaling_the_series() -> None:
    """Trends rescales each series by its own future-inclusive maximum. Multiplying the whole
    series by a positive constant must leave every feature unchanged, or that normalisation
    would leak the future into the features."""
    origin = _weeks()[_ORIGIN_INDEX]
    a = build_signal_features(_panel(), _terms(), _weekly(1.0), [origin]).sort("style_key")
    b = build_signal_features(_panel(), _terms(), _weekly(3.7), [origin]).sort("style_key")
    for col in SIGNAL_FEATURE_COLS:
        for va, vb in zip(a[col].to_list(), b[col].to_list(), strict=True):
            if va is None:
                assert vb is None
            else:
                assert vb == pytest.approx(va, rel=1e-9, abs=1e-12)


def test_features_do_not_depend_on_which_other_origins_are_requested() -> None:
    a, b = _weeks()[60], _weeks()[72]
    solo = build_signal_features(_panel(), _terms(), _weekly(), [a]).sort("style_key")
    both = build_signal_features(_panel(), _terms(), _weekly(), [a, b])
    assert solo.equals(both.filter(pl.col("origin_week") == a).sort("style_key"))


def test_add_signal_features_is_additive_and_preserves_order() -> None:
    origin = _weeks()[_ORIGIN_INDEX]
    frame = pl.DataFrame(
        {"style_key": ["S2", "S0", "S1"], "origin_week": [origin] * 3, "y_true": [1.0, 2.0, 3.0]}
    )
    joined = add_signal_features(frame, _panel(), _terms(), _weekly())
    assert joined.columns == [*frame.columns, *SIGNAL_FEATURE_COLS]
    assert joined["style_key"].to_list() == ["S2", "S0", "S1"]
    assert joined.select(frame.columns).equals(frame)


def test_lead_columns_are_the_documented_three() -> None:
    assert LEAD_FEATURE_COLS == ["sg_lead2", "sg_lead4", "sg_lead8"]
