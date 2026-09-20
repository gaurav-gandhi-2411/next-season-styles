from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from nss.features.customer_features import (
    BASE_STATS,
    CUSTOMER_FEATURE_COLS,
    MIN_BUYERS,
    REPEAT_LOOKBACK_WEEKS,
    WINDOW_WEEKS,
    add_customer_features,
    build_customer_features,
)

_WEEK0 = date(2018, 1, 1)
_N_WEEKS = 48
_ORIGIN_INDEX = 40  # window weeks 28..40: the 26-week repeat lookback (weeks 2..) fits in the data
_N_CUSTOMERS = 60
_STYLES = {"A": (0, 30), "B": (20, 50), "C": (30, 60)}  # style -> customer id pool [lo, hi)
_BUYERS_PER_WEEK = 8
_FIRST_LATE_CUSTOMER = 54  # customers 54..59 appear only after the origin (future-only customers)


def _weeks(n: int = _N_WEEKS) -> list[date]:
    return [_WEEK0 + timedelta(weeks=i) for i in range(n)]


def _customers() -> pl.DataFrame:
    rng = np.random.default_rng(42)
    ages = rng.integers(16, 70, _N_CUSTOMERS).tolist()
    ages[3] = None  # a customer with an unknown age
    club = rng.choice(["ACTIVE", "PRE-CREATE", "LEFT CLUB"], _N_CUSTOMERS).tolist()
    club[5] = None
    news = rng.choice(["NONE", "Regularly", "Monthly"], _N_CUSTOMERS).tolist()
    postal = [f"P{p}" for p in rng.integers(0, 12, _N_CUSTOMERS)]
    return pl.DataFrame(
        {
            "customer_id": list(range(_N_CUSTOMERS)),
            "age": ages,
            "club_member_status": club,
            "fashion_news_frequency": news,
            "postal_code": postal,
        }
    )


def _purchases() -> pl.DataFrame:
    """Each week every style gets `_BUYERS_PER_WEEK` distinct buyers from its pool, plus one
    duplicated purchase line (same customer, same week) that must count once."""
    rng = np.random.default_rng(7)
    rows: list[tuple[str, date, int]] = []
    for i, week in enumerate(_weeks()):
        for style, (lo, hi) in _STYLES.items():
            if i <= _ORIGIN_INDEX:  # customers >= _FIRST_LATE_CUSTOMER first buy after the origin
                hi = min(hi, _FIRST_LATE_CUSTOMER)
            buyers = rng.choice(np.arange(lo, hi), _BUYERS_PER_WEEK, replace=False).tolist()
            rows.extend((style, week, int(c)) for c in buyers)
            rows.append((style, week, int(buyers[0])))  # duplicate line
    return pl.DataFrame(
        rows, schema=["style_key", "week_start", "customer_id"], orient="row"
    ).with_columns(pl.col("week_start").cast(pl.Date))


def _panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "style_key": [s for s in _STYLES for _ in _weeks()],
            "week_start": [w for _ in _STYLES for w in _weeks()],
        }
    )


def _row(frame: pl.DataFrame, key: str) -> dict[str, object]:
    return frame.filter(pl.col("style_key") == key).row(0, named=True)


def _reference(style: str, t: int) -> dict[str, float | None]:
    """Independent, loop-based computation of every base statistic for `style` at week index `t`."""
    cust = {r["customer_id"]: r for r in _customers().to_dicts()}
    purchases = _purchases().filter(pl.col("style_key") == style)
    week_idx = {w: i for i, w in enumerate(_weeks())}
    bw = {(r["customer_id"], week_idx[r["week_start"]]) for r in purchases.to_dicts()}
    window = [(c, w) for c, w in sorted(bw) if t - WINDOW_WEEKS + 1 <= w <= t]
    ages = sorted(cust[c]["age"] for c, _ in window if cust[c]["age"] is not None)
    out: dict[str, float | None] = {}
    n = len(ages)
    mean = sum(ages) / n
    out["cust_age_mean"] = mean
    out["cust_age_median"] = float(ages[int(np.ceil(0.5 * n)) - 1])
    out["cust_age_p25"] = float(ages[int(np.ceil(0.25 * n)) - 1])
    out["cust_age_p75"] = float(ages[int(np.ceil(0.75 * n)) - 1])
    out["cust_age_std"] = float(np.std(ages))
    out["cust_age_gini"] = sum(abs(a - b) for a in ages for b in ages) / (2 * n * n * mean)
    known = [cust[c]["club_member_status"] for c, _ in window if cust[c]["club_member_status"]]
    out["cust_club_active_share"] = known.count("ACTIVE") / len(known)
    out["cust_club_precreate_share"] = known.count("PRE-CREATE") / len(known)
    news = [cust[c]["fashion_news_frequency"] for c, _ in window]
    out["cust_fn_engaged_share"] = sum(x in ("Regularly", "Monthly") for x in news) / len(news)
    repeats = sum(
        any((c, w2) in bw for w2 in range(w - REPEAT_LOOKBACK_WEEKS, w)) for c, w in window
    )
    out["cust_repeat_share"] = repeats / len(window)
    postal = [cust[c]["postal_code"] for c, _ in window]
    counts = np.array([postal.count(p) for p in set(postal)], dtype=float)
    out["cust_geo_distinct_postal"] = float(len(counts))
    out["cust_geo_hhi"] = float(((counts / counts.sum()) ** 2).sum())
    return out


def test_columns_and_row_keys() -> None:
    origin = _weeks()[_ORIGIN_INDEX]
    out = build_customer_features(_purchases(), _customers(), _panel(), [origin])
    assert out.columns == ["style_key", "origin_week", *CUSTOMER_FEATURE_COLS]
    assert out.height == len(_STYLES)
    assert set(out["origin_week"].to_list()) == {origin}
    assert len(BASE_STATS) == 12
    assert "cust_age_median_shift_13w" in CUSTOMER_FEATURE_COLS
    assert "cust_age_median_slope_13w" not in CUSTOMER_FEATURE_COLS
    assert len(CUSTOMER_FEATURE_COLS) == 36


@pytest.mark.parametrize("style", list(_STYLES))
def test_levels_and_slopes_match_an_independent_reference(style: str) -> None:
    t = 44  # t-4 = 40 has a repeat window starting at 28 >= 26; t-13 = 31 does not (repeat null)
    row = _row(build_customer_features(_purchases(), _customers(), _panel(), [_weeks()[t]]), style)
    now, then4, then13 = _reference(style, t), _reference(style, t - 4), _reference(style, t - 13)
    for base in BASE_STATS:
        assert row[base] == pytest.approx(now[base]), base
        assert row[f"{base}_slope_4w"] == pytest.approx((now[base] - then4[base]) / 4), base
    for base in BASE_STATS:
        if base == "cust_age_median":
            assert row["cust_age_median_shift_13w"] == pytest.approx(now[base] - then13[base])
        elif base != "cust_repeat_share":  # repeat at t-13 is inside the repeat warm-up: null
            assert row[f"{base}_slope_13w"] == pytest.approx((now[base] - then13[base]) / 13), base
    assert row["cust_repeat_share_slope_13w"] is None


def test_repeat_share_is_null_until_the_lookback_fits_and_other_stats_are_not() -> None:
    row = _row(
        build_customer_features(_purchases(), _customers(), _panel(), [_weeks()[30]]), "A"
    )  # window 18..30 starts before week 26: lookback runs off the start of the data
    assert row["cust_repeat_share"] is None
    assert row["cust_age_mean"] is not None


def test_window_that_does_not_fit_or_too_few_buyers_is_null() -> None:
    early = _row(build_customer_features(_purchases(), _customers(), _panel(), [_weeks()[5]]), "A")
    assert all(early[c] is None for c in CUSTOMER_FEATURE_COLS)
    origin = _weeks()[_ORIGIN_INDEX]
    window_start = _weeks()[_ORIGIN_INDEX - WINDOW_WEEKS + 1]
    # keep only ONE week of style-A buyers in the window: _BUYERS_PER_WEEK (8) < MIN_BUYERS (10)
    assert _BUYERS_PER_WEEK < MIN_BUYERS
    thin = _purchases().filter(
        ~((pl.col("style_key") == "A") & (pl.col("week_start") >= window_start))
        | (pl.col("week_start") == origin)
    )
    row = _row(build_customer_features(thin, _customers(), _panel(), [origin]), "A")
    assert row["cust_age_mean"] is None
    assert row["cust_geo_distinct_postal"] is None
    assert row["cust_club_active_share"] is None
    assert (
        _row(build_customer_features(thin, _customers(), _panel(), [origin]), "B")["cust_age_mean"]
        is not None
    )


def test_duplicate_purchase_lines_count_once() -> None:
    purchases = _purchases()
    doubled = pl.concat([purchases, purchases])
    origin = [_weeks()[_ORIGIN_INDEX]]
    a = build_customer_features(purchases, _customers(), _panel(), origin).sort("style_key")
    b = build_customer_features(doubled, _customers(), _panel(), origin).sort("style_key")
    assert a.equals(b)


def _future_shuffled(purchases: pl.DataFrame, origin: date, seed: int) -> pl.DataFrame:
    """Independently permute style_key and customer_id among purchases after the origin."""
    past = purchases.filter(pl.col("week_start") <= origin)
    future = purchases.filter(pl.col("week_start") > origin)
    a = future["style_key"].sample(fraction=1.0, shuffle=True, seed=seed)
    b = future["customer_id"].sample(fraction=1.0, shuffle=True, seed=seed + 1)
    shuffled = future.with_columns(a.alias("style_key"), b.alias("customer_id"))
    return pl.concat([past, shuffled.select(purchases.columns)])


def _scramble_future_only_customers(
    customers: pl.DataFrame, purchases: pl.DataFrame, origin: date
) -> pl.DataFrame:
    """Rewrite every attribute of customers who have NOT bought at or before the origin."""
    seen = set(purchases.filter(pl.col("week_start") <= origin)["customer_id"].to_list())
    return customers.with_columns(
        [
            pl.when(pl.col("customer_id").is_in(list(seen)))
            .then(pl.col(c))
            .otherwise(pl.lit(v))
            .alias(c)
            for c, v in (
                ("age", 90),
                ("club_member_status", "PRE-CREATE"),
                ("fashion_news_frequency", "Regularly"),
                ("postal_code", "ZZ"),
            )
        ]
    )


def _invariant_to_future(**leak: object) -> bool:
    """The causality check: shuffling every post-origin purchase (and rewriting customers who only
    appear after the origin) leaves every feature at the origin unchanged."""
    purchases, customers, panel = _purchases(), _customers(), _panel()
    origin = _weeks()[_ORIGIN_INDEX]
    base = build_customer_features(purchases, customers, panel, [origin], **leak).sort("style_key")
    shuffled = _future_shuffled(purchases, origin, seed=42)
    assert not shuffled.equals(purchases)  # the shuffle really permuted something
    scrambled = _scramble_future_only_customers(customers, purchases, origin)
    assert not scrambled.equals(customers)  # ... and some customer really is future-only
    after = build_customer_features(shuffled, scrambled, panel, [origin], **leak).sort("style_key")
    return base.equals(after)


def test_customer_features_are_causally_safe() -> None:
    assert _invariant_to_future()


def test_negative_control_leaky_window_fails_the_causality_check() -> None:
    """A trailing window that extends 3 weeks past the origin must FAIL the same check."""
    assert not _invariant_to_future(_window_lead_weeks=3)


def test_negative_control_forward_looking_repeat_flag_fails_the_causality_check() -> None:
    """A repeat flag defined by the NEXT purchase (instead of an earlier one) must FAIL it too."""
    assert not _invariant_to_future(_repeat_lookahead=True)


def test_negative_control_pre_origin_purchases_and_attributes_do_change_the_features() -> None:
    purchases, customers, panel = _purchases(), _customers(), _panel()
    origin = _weeks()[_ORIGIN_INDEX]
    base = build_customer_features(purchases, customers, panel, [origin]).sort("style_key")
    # (a) drop one style-A buyer-week inside the window: A changes, B does not
    week = _weeks()[_ORIGIN_INDEX - 1]
    dropped = purchases.filter(~((pl.col("style_key") == "A") & (pl.col("week_start") == week)))
    changed = build_customer_features(dropped, customers, panel, [origin]).sort("style_key")
    assert _row(base, "A")["cust_age_mean"] != _row(changed, "A")["cust_age_mean"]
    assert _row(base, "B") == _row(changed, "B")
    # (b) rewrite the age of a customer who DID buy before the origin: features change
    buyer = purchases.filter(pl.col("week_start") <= origin)["customer_id"][0]
    aged = customers.with_columns(
        pl.when(pl.col("customer_id") == buyer).then(99).otherwise(pl.col("age")).alias("age")
    )
    changed_attr = build_customer_features(purchases, aged, panel, [origin]).sort("style_key")
    assert not base.equals(changed_attr)


def test_features_at_earlier_origin_ignore_later_origins_in_the_request() -> None:
    a, b = _weeks()[36], _weeks()[44]
    solo = build_customer_features(_purchases(), _customers(), _panel(), [a]).sort("style_key")
    both = build_customer_features(_purchases(), _customers(), _panel(), [a, b])
    assert solo.equals(both.filter(pl.col("origin_week") == a).sort("style_key"))


def test_truncating_the_data_after_the_origin_changes_nothing() -> None:
    origin = _weeks()[_ORIGIN_INDEX]
    full = build_customer_features(_purchases(), _customers(), _panel(), [origin]).sort("style_key")
    cut = build_customer_features(
        _purchases().filter(pl.col("week_start") <= origin),
        _customers(),
        _panel().filter(pl.col("week_start") <= origin),
        [origin],
    ).sort("style_key")
    assert full.equals(cut)


def test_add_customer_features_is_additive_and_preserves_order() -> None:
    origin = _weeks()[_ORIGIN_INDEX]
    feats = build_customer_features(_purchases(), _customers(), _panel(), [origin])
    frame = pl.DataFrame(
        {"style_key": ["C", "A", "B"], "origin_week": [origin] * 3, "y_true": [1.0, 2.0, 3.0]}
    )
    joined = add_customer_features(frame, feats)
    assert joined.columns == [*frame.columns, *CUSTOMER_FEATURE_COLS]
    assert joined["style_key"].to_list() == ["C", "A", "B"]
    assert joined.select(frame.columns).equals(frame)
    assert joined.filter(pl.col("style_key") == "A")["cust_age_mean"].item() is not None


def test_bad_inputs_raise() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        build_customer_features(
            _purchases().drop("customer_id"), _customers(), _panel(), [_weeks()[40]]
        )
    dup = pl.concat([_customers(), _customers().head(1)])
    with pytest.raises(ValueError, match="unique"):
        build_customer_features(_purchases(), dup, _panel(), [_weeks()[40]])
