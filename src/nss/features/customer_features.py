"""Buyer-mix features from `customers.csv`: who is buying a style, and is that base broadening?

Every feature in `nss.features.model_features` derives from a style's sales history. These describe
the PEOPLE behind those sales, aggregated per style-week, with trailing windows only.

BUYER UNIT: a "buyer" is a distinct `(style_key, customer_id, week_start)` triple (a customer buying
three units of a style in one week is one buyer-week). All statistics below are over the buyer-weeks
in the trailing `WINDOW_WEEKS` (13) weeks ending at (and including) the origin week.

BASE STATISTICS (`BASE_STATS`, computed per style-week over the trailing window):

- `cust_age_mean` / `cust_age_median` / `cust_age_p25` / `cust_age_p75`: age of buyer-weeks with a
  known age (discrete quantiles: the smallest age whose cumulative share is >= q);
- `cust_age_std`, `cust_age_gini`: concentration of buyer ages (population std; Gini coefficient of
  the age distribution, 0 = every buyer the same age);
- `cust_club_active_share` / `cust_club_precreate_share`: share of buyer-weeks whose customer has
  `club_member_status` ACTIVE / PRE-CREATE, among buyer-weeks with a known status;
- `cust_fn_engaged_share`: share whose `fashion_news_frequency` is Regularly or Monthly, among
  buyer-weeks with a known value;
- `cust_repeat_share`: share of buyer-weeks where the same customer bought the same style in the
  previous `REPEAT_LOOKBACK_WEEKS` (26) weeks (repeat vs new; `1 - repeat_share` is the new-buyer
  share). NULL until the trailing window and its 26-week lookback both fit inside the data (the
  first ~38 weeks), never imputed;
- `cust_geo_distinct_postal`, `cust_geo_hhi`: number of distinct `postal_code`s among buyer-weeks
  and the Herfindahl index (sum of squared shares) of buyer-weeks over postal codes (lower = more
  geographically spread).

A base statistic is null when its own denominator has fewer than `MIN_BUYERS` (10) buyer-weeks in
the window (too few to be a statistic) or the 13-week window does not fit inside the data yet.

FEATURE COLUMNS (`CUSTOMER_FEATURE_COLS`): each base statistic (level), plus its 4-week and 13-week
slope `(now - L weeks ago) / L` -- the same 2-point finite-difference definition as
`model_features.slope_4w/13w`; null when either endpoint is null. The 13-week slope of the median
age is instead named `cust_age_median_shift_13w` and is the undivided change in median age in years
("the 13-week shift in median age"); dividing it by 13 would only re-scale it, and a tree model is
scale-invariant, so no separate slope column is kept for it.

CAUSAL CONVENTION: the value at `origin_week` uses purchases with `week_start <= origin_week` only.
The window aggregates are prefix sums over weeks, the repeat flag looks at a customer's EARLIER
purchase weeks only, and slopes read the same statistic at `origin_week - 4/13` weeks. The compute-
then-filter pattern of `model_features.build_features` is used: the statistics are computed for
every week of the panel and only then restricted to the requested origins.

KNOWN LIMITATION (documented, not fixable from the data): `customers.csv` is ONE snapshot, taken at
the end of the data (2020-09), not a history. `age` and `postal_code` are stable enough for that not
to matter (age is shifted by at most ~2 years, uniformly). `club_member_status` and
`fashion_news_frequency` are STATE variables whose value at a past purchase week is unknown: a
customer's snapshot status can encode behaviour AFTER the origin (e.g. joined the club later). The
buyer SET is strictly causal (only customers who bought at or before the origin are counted), so
this is a vintage caveat on the attribute values, not a look-ahead over the style's own future
sales, and the causality test cannot see it (attributes are time-invariant in the data). Treat the
two mix features as "snapshot-vintage" and read any result that leans on them accordingly.

PRIVACY: this module reads per-customer attributes but only ever emits per-(style, week) aggregates.
No per-customer row is written anywhere.

The two private `_...` keyword arguments of `build_customer_features` exist ONLY so the tests can
build deliberately leaky variants (a window that extends into the future; a repeat flag that looks
forward) and prove the causality test fails on them. Never pass them in production code.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from nss.features.model_features import SLOPE_WINDOWS_WEEKS

WINDOW_WEEKS = 13
REPEAT_LOOKBACK_WEEKS = 26
MIN_BUYERS = 10
# Ages in customers.csv span 16..99; 120 bins (ages 0..119) leave room for any plausible value and
# anything outside is treated as a missing age, never clipped into an edge bin.
AGE_BINS = 120

CLUB_ACTIVE = "ACTIVE"
CLUB_PRECREATE = "PRE-CREATE"
NEWS_ENGAGED: tuple[str, ...] = ("Regularly", "Monthly")

BASE_STATS: list[str] = [
    "cust_age_mean",
    "cust_age_median",
    "cust_age_p25",
    "cust_age_p75",
    "cust_age_std",
    "cust_age_gini",
    "cust_club_active_share",
    "cust_club_precreate_share",
    "cust_fn_engaged_share",
    "cust_repeat_share",
    "cust_geo_distinct_postal",
    "cust_geo_hhi",
]
_SHIFT_STAT = "cust_age_median"  # its 13w slope is exposed as an undivided "shift"


def _slope_name(base: str, window: int) -> str:
    if base == _SHIFT_STAT and window == WINDOW_WEEKS:
        return f"{base}_shift_{window}w"
    return f"{base}_slope_{window}w"


CUSTOMER_FEATURE_COLS: list[str] = [
    col
    for base in BASE_STATS
    for col in (base, *[_slope_name(base, w) for w in SLOPE_WINDOWS_WEEKS])
]

_PURCHASE_COLS = ["style_key", "week_start", "customer_id"]
_CUSTOMER_COLS = [
    "customer_id",
    "age",
    "club_member_status",
    "fashion_news_frequency",
    "postal_code",
]


def _window_bounds(t: int, n_weeks: int, lead: int) -> tuple[int, int] | None:
    """Inclusive week-index window `[t-12+lead, t+lead]`, or `None` if it leaves the data."""
    start = t - (WINDOW_WEEKS - 1) + lead
    end = t + lead
    if start < 0 or end > n_weeks - 1:
        return None
    return start, end


def _age_stats(hist: np.ndarray) -> dict[str, np.ndarray]:
    """Age statistics for rows of an age histogram `hist` with shape `(rows, AGE_BINS)`."""
    counts = hist.astype(np.float64)
    n = counts.sum(axis=1)
    ok = n >= MIN_BUYERS
    safe_n = np.where(ok, n, 1.0)
    ages = np.arange(AGE_BINS, dtype=np.float64)
    mean = counts @ ages / safe_n
    var = counts @ (ages**2) / safe_n - mean**2
    std = np.sqrt(np.maximum(var, 0.0))
    cum = np.cumsum(counts, axis=1)

    def quantile(q: float) -> np.ndarray:
        return np.argmax(cum >= q * safe_n[:, None], axis=1).astype(np.float64)

    # Gini via sorted-bin pair sums: sum_{i<j} n_i n_j (a_j - a_i) / (N^2 * mean).
    pairs = (counts * ages * (cum - counts)).sum(axis=1) - (counts * ages * (n[:, None] - cum)).sum(
        axis=1
    )
    gini = pairs / (safe_n**2 * np.where(mean > 0, mean, 1.0))
    out = {
        "cust_age_mean": mean,
        "cust_age_median": quantile(0.5),
        "cust_age_p25": quantile(0.25),
        "cust_age_p75": quantile(0.75),
        "cust_age_std": std,
        "cust_age_gini": np.where(mean > 0, gini, np.nan),
    }
    return {k: np.where(ok, v, np.nan) for k, v in out.items()}


def _share(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    ok = den >= MIN_BUYERS
    return np.where(ok, num / np.where(ok, den, 1.0), np.nan)


def _slope_columns(base: str, values: np.ndarray) -> dict[str, np.ndarray]:
    """Level plus 4w/13w slope columns from a dense `(styles, weeks)` array of a base statistic."""
    out = {base: values}
    for window in SLOPE_WINDOWS_WEEKS:
        slope = np.full_like(values, np.nan)
        diff = values[:, window:] - values[:, :-window]
        slope[:, window:] = (
            diff if _slope_name(base, window).endswith("shift_13w") else diff / window
        )
        out[_slope_name(base, window)] = slope
    return out


def build_customer_features(
    purchases: pl.DataFrame,
    customers: pl.DataFrame,
    panel: pl.DataFrame,
    origin_weeks: list[date] | None = None,
    *,
    _window_lead_weeks: int = 0,
    _repeat_lookahead: bool = False,
) -> pl.DataFrame:
    """Causal buyer-mix features for every `(style_key, origin_week)` pair. See module docstring.

    Args:
        purchases: One row per purchase line with `style_key`, `week_start` (the Monday of the
            purchase week, same convention as the panel) and `customer_id`. Duplicated
            `(style_key, customer_id, week_start)` rows are collapsed (one buyer-week).
        customers: One row per customer with unique `customer_id`, `age`, `club_member_status`,
            `fashion_news_frequency`, `postal_code` (any of the four may be null).
        panel: The dense style-week panel (needs `style_key`, `week_start`); it fixes the style set,
            the week grid (its first week is the start of history for the repeat lookback) and which
            `(style, week)` rows get a feature row.
        origin_weeks: Origin weeks to return; `None` returns every panel row.
        _window_lead_weeks, _repeat_lookahead: test-only hooks that make the features deliberately
            NON-causal (see module docstring). Leave at their defaults.

    Returns:
        One row per panel `(style_key, week_start)` at the requested origins, columns `style_key`,
        `origin_week` and every column in `CUSTOMER_FEATURE_COLS` (null where undefined).

    Raises:
        ValueError: if required columns are missing or `customer_id` is not unique in `customers`.
    """
    for name, frame, cols in (
        ("purchases", purchases, _PURCHASE_COLS),
        ("customers", customers, _CUSTOMER_COLS),
        ("panel", panel, ["style_key", "week_start"]),
    ):
        missing = set(cols) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    if customers["customer_id"].n_unique() != customers.height:
        raise ValueError("customers.customer_id must be unique")

    styles = panel["style_key"].unique().sort()
    n_styles = styles.len()
    week0: date = panel["week_start"].min()
    n_weeks = (panel["week_start"].max() - week0).days // 7 + 1
    week_dates = [week0 + timedelta(weeks=i) for i in range(n_weeks)]

    cust = customers.select(_CUSTOMER_COLS).with_row_index("_c")
    n_cust = cust.height
    age = cust["age"].cast(pl.Float64).to_numpy()  # nulls -> NaN, which fails every test below
    age_ok = (age >= 0) & (age <= AGE_BINS - 1) & (age == np.floor(age))
    age_bin = np.where(age_ok, age, -1.0).astype(np.int64)
    club = cust["club_member_status"]
    club_known = club.is_not_null().to_numpy()
    club_active = (club == CLUB_ACTIVE).fill_null(False).to_numpy()
    club_pre = (club == CLUB_PRECREATE).fill_null(False).to_numpy()
    news = cust["fashion_news_frequency"]
    news_known = news.is_not_null().to_numpy()
    news_engaged = news.is_in(list(NEWS_ENGAGED)).fill_null(False).to_numpy()
    postal = cust["postal_code"].cast(pl.String).cast(pl.Categorical).to_physical()
    postal_code = postal.cast(pl.Int64).fill_null(-1).to_numpy()
    n_postal = int(postal_code.max()) + 2 if n_cust else 1

    coded = (
        purchases.select(_PURCHASE_COLS)
        .join(pl.DataFrame({"style_key": styles}).with_row_index("_s"), on="style_key", how="inner")
        .join(cust.select("customer_id", "_c"), on="customer_id", how="inner")
        .with_columns(((pl.col("week_start") - pl.lit(week0)).dt.total_days() // 7).alias("_w"))
        .filter((pl.col("_w") >= 0) & (pl.col("_w") < n_weeks))
    )
    s = coded["_s"].to_numpy().astype(np.int64)
    c = coded["_c"].to_numpy().astype(np.int64)
    w = coded["_w"].to_numpy().astype(np.int64)
    key = np.unique((s * n_cust + c) * n_weeks + w)  # distinct buyer-weeks, sorted by (s, c, w)
    w = key % n_weeks
    sc = key // n_weeks
    c = sc % n_cust
    s = sc // n_cust

    # repeat flag: an EARLIER purchase week of the same (style, customer) within the lookback. The
    # `_repeat_lookahead` hook flips it to the NEXT purchase week -- deliberately leaky, tests only.
    repeat = np.zeros(key.shape[0], dtype=bool)
    if key.shape[0] > 1:
        same = sc[1:] == sc[:-1]
        near = (w[1:] - w[:-1]) <= REPEAT_LOOKBACK_WEEKS
        if _repeat_lookahead:
            repeat[:-1] = same & near
        else:
            repeat[1:] = same & near

    n_cells = n_styles * n_weeks
    cell = s * n_weeks + w

    def cell_count(mask: np.ndarray | None = None) -> np.ndarray:
        weights = None if mask is None else mask.astype(np.float64)
        return np.bincount(cell, weights=weights, minlength=n_cells).reshape(n_styles, n_weeks)

    repeat_n = cell_count(repeat)
    club_known_n = cell_count(club_known[c])
    club_active_n = cell_count(club_active[c])
    club_pre_n = cell_count(club_pre[c])
    news_known_n = cell_count(news_known[c])
    news_engaged_n = cell_count(news_engaged[c])

    has_age = age_bin[c] >= 0
    hist = (
        np.bincount(cell[has_age] * AGE_BINS + age_bin[c][has_age], minlength=n_cells * AGE_BINS)
        .reshape(n_styles, n_weeks, AGE_BINS)
        .astype(np.int32)
    )

    has_postal = postal_code[c] >= 0
    # (week, style, postal) -> buyer-weeks; week-major so a window is a contiguous slice.
    pkey, pcount = np.unique(
        (w[has_postal] * n_styles + s[has_postal]) * n_postal + postal_code[c][has_postal],
        return_counts=True,
    )
    p_week = pkey // (n_styles * n_postal)
    p_style = (pkey // n_postal) % n_styles
    p_code = pkey % n_postal

    def prefix(arr: np.ndarray) -> np.ndarray:
        pad = np.zeros((arr.shape[0], 1, *arr.shape[2:]), dtype=arr.dtype)
        return np.concatenate([pad, np.cumsum(arr, axis=1, dtype=arr.dtype)], axis=1)

    cums = {
        "buyers": prefix(cell_count()),
        "repeat": prefix(repeat_n),
        "club_known": prefix(club_known_n),
        "club_active": prefix(club_active_n),
        "club_pre": prefix(club_pre_n),
        "news_known": prefix(news_known_n),
        "news_engaged": prefix(news_engaged_n),
    }
    hist_cum = prefix(hist)

    stats = {name: np.full((n_styles, n_weeks), np.nan) for name in BASE_STATS}
    for t in range(n_weeks):
        bounds = _window_bounds(t, n_weeks, _window_lead_weeks)
        if bounds is None:
            continue
        lo, hi = bounds

        def win(name: str, lo: int = lo, hi: int = hi) -> np.ndarray:
            return cums[name][:, hi + 1] - cums[name][:, lo]

        for name, value in _age_stats(hist_cum[:, hi + 1] - hist_cum[:, lo]).items():
            stats[name][:, t] = value
        stats["cust_club_active_share"][:, t] = _share(win("club_active"), win("club_known"))
        stats["cust_club_precreate_share"][:, t] = _share(win("club_pre"), win("club_known"))
        stats["cust_fn_engaged_share"][:, t] = _share(win("news_engaged"), win("news_known"))
        # repeat needs the flag's own lookback to fit inside the data: first valid window start = 26
        if _repeat_lookahead or lo >= REPEAT_LOOKBACK_WEEKS:
            stats["cust_repeat_share"][:, t] = _share(win("repeat"), win("buyers"))

        a = np.searchsorted(p_week, lo, side="left")
        b = np.searchsorted(p_week, hi, side="right")
        if b > a:
            pair = p_style[a:b] * n_postal + p_code[a:b]
            uniq, inv = np.unique(pair, return_inverse=True)
            total = np.bincount(inv, weights=pcount[a:b].astype(np.float64))
            style_of = uniq // n_postal
            n_total = np.bincount(style_of, weights=total, minlength=n_styles)
            sumsq = np.bincount(style_of, weights=total**2, minlength=n_styles)
            distinct = np.bincount(style_of, minlength=n_styles).astype(np.float64)
            ok = n_total >= MIN_BUYERS
            stats["cust_geo_distinct_postal"][:, t] = np.where(ok, distinct, np.nan)
            stats["cust_geo_hhi"][:, t] = np.where(
                ok, sumsq / np.where(ok, n_total, 1.0) ** 2, np.nan
            )

    columns: dict[str, np.ndarray] = {}
    for base in BASE_STATS:
        columns.update(_slope_columns(base, stats[base]))

    grid = pl.DataFrame(
        {
            "style_key": np.repeat(styles.to_numpy(), n_weeks),
            "week_start": week_dates * n_styles,
            **{name: columns[name].reshape(-1) for name in CUSTOMER_FEATURE_COLS},
        }
    ).with_columns(pl.col("week_start").cast(pl.Date))
    grid = grid.with_columns(pl.col(CUSTOMER_FEATURE_COLS).fill_nan(None))
    out = grid.join(panel.select("style_key", "week_start"), on=["style_key", "week_start"])
    if origin_weeks is not None:
        out = out.filter(pl.col("week_start").is_in(origin_weeks))
    return out.rename({"week_start": "origin_week"}).select(
        ["style_key", "origin_week", *CUSTOMER_FEATURE_COLS]
    )


def add_customer_features(model_frame: pl.DataFrame, features: pl.DataFrame) -> pl.DataFrame:
    """Join `build_customer_features` output onto a `build_model_frame`-shaped frame (additive).

    Left join with `maintain_order="left"` so LightGBM's training-row order (and with it the
    determinism guarantee) is unchanged; the `CUSTOMER_FEATURE_COLS` are appended at the end.
    """
    return model_frame.join(
        features.select(["style_key", "origin_week", *CUSTOMER_FEATURE_COLS]),
        on=["style_key", "origin_week"],
        how="left",
        maintain_order="left",
    )
