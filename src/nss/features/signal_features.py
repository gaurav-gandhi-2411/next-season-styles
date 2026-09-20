"""External-signal features (weekly Google Trends per style term), computed strictly causally.

HYPOTHESIS (`reports/EXPERIMENT_external_signals.md`): every existing feature, and the
seasonal-naive baseline, derives from the retailer's own sales history. Public search interest is
information seasonal-naive structurally cannot have; a style rising in search before it rises in
sales would be a leading indicator at the 13-week forecast horizon.

SCALE INVARIANCE IS THE CAUSALITY CONSTRAINT SPECIFIC TO TRENDS. Google Trends rescales every
series to 0-100 by the series' OWN MAXIMUM over the requested window (2018-09 .. 2020-09). A raw
value at an early week therefore depends on how high the series climbs LATER. Any feature that
uses the raw level (or a raw difference) leaks the future through that normalisation. So every
feature here is a ratio of two values of the same series (a log-ratio of trailing means), which is
unchanged by multiplying the series by a positive constant. `tests/test_signal_features.py` checks
exactly that invariance. It is also why the requested "signal level" is implemented as the recent
level relative to the trailing quarter (`sg_level_rel13`), not the raw 0-100 number.

ALIGNMENT. A Trends week is labelled by the Sunday that starts it (Sun..Sat). It is mapped to the
Monday after (`week_start = sunday + 1 day`), the panel's week convention (Mon..Sun). The signal
week that carries `week_start = W` therefore ends on Saturday W+5, inside the origin week W whose
own sales (through Sunday W+6) the model already treats as observed. Nothing dated after the
origin week is read.

FEATURES (`SIGNAL_FEATURE_COLS`), with m4 = trailing 4-week mean of the signal (weeks t-3..t), m13
the trailing 13-week mean, b52 the trailing 52-week mean (at least 26 weeks present):
- `sg_level_rel13`   ln(m4 / m13)             recent level versus the trailing quarter;
- `sg_slope_4w`      ln(m4_t / m4_{t-4}) / 4  4-week slope of the smoothed signal;
- `sg_slope_13w`     ln(m4_t / m4_{t-13}) / 13;
- `sg_rel_52w`       ln(m4 / b52)             recent level versus the trailing-year baseline;
- `sg_lead2/4/8`     the 4-week signal slope as of 2, 4 and 8 weeks BEFORE the origin. These are
                     the lead features: the model can compare where the signal was L weeks ago with
                     where sales are now. All three lead lengths are always included; none is
                     selected on outcomes;
- `sg_div_4w/13w`    signal slope minus the style's own sales slope over the same span
                     (`ln(1+s4_t) - ln(1+s4_{t-L})`, s4 = trailing 4-week mean of
                     `intensity_shrunk`, divided by L). Positive = search rising faster than
                     sales: the "interesting case".

A feature is null when its window is incomplete or a mean is not positive; nothing is imputed. A
style with no mapped term, or whose term's series is too sparse (`MIN_NONZERO_SHARE`), gets all
signal features null.

CAUSAL CONVENTION. Everything is a trailing window / shift within a term (signal) or within a
style (sales) in week order, computed densely and filtered to the requested origins afterwards, as
in `model_features` and `price_features`. `_lead_weeks` is a test-only hook that slides the signal
into the future so the negative control can prove the causality test would catch a real leak.
"""

from __future__ import annotations

from datetime import date

import polars as pl

MIN_NONZERO_SHARE = 0.9  # a term needs a non-zero reading in >= 90% of weeks to count as usable
SLOPE_SHORT = 4
SLOPE_LONG = 13
LEAD_WEEKS: tuple[int, ...] = (2, 4, 8)
BASELINE_WEEKS = 52
BASELINE_MIN_WEEKS = 26

SIGNAL_FEATURE_COLS: list[str] = [
    "sg_level_rel13",
    "sg_slope_4w",
    "sg_slope_13w",
    "sg_rel_52w",
    *[f"sg_lead{lag}" for lag in LEAD_WEEKS],
    "sg_div_4w",
    "sg_div_13w",
]
LEAD_FEATURE_COLS: list[str] = [f"sg_lead{lag}" for lag in LEAD_WEEKS]


def usable_terms(weekly: pl.DataFrame) -> pl.DataFrame:
    """Per-term usability: share of non-zero weeks and whether it clears `MIN_NONZERO_SHARE`.

    Args:
        weekly: columns `term`, `week_start`, `value`.
    """
    return (
        weekly.group_by("term")
        .agg(
            n_weeks=pl.len(),
            nonzero_share=(pl.col("value") > 0).mean(),
            max_value=pl.col("value").max(),
        )
        .with_columns((pl.col("nonzero_share") >= MIN_NONZERO_SHARE).alias("usable"))
        .sort("term")
    )


def _log_ratio(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """ln(num/den), null unless both are positive."""
    return pl.when((num > 0) & (den > 0)).then((num / den).log()).otherwise(None)


def _signal_weekly_features(weekly: pl.DataFrame, lead: int) -> pl.DataFrame:
    """Per-(term, week) signal features, before any sales-side divergence."""
    w = weekly.select("term", "week_start", pl.col("value").cast(pl.Float64)).sort(
        "term", "week_start"
    )
    order = {"order_by": "week_start"}
    x = pl.col("value").shift(-lead).over("term", **order)  # lead=0 is the identity
    w = w.with_columns(x.alias("_x"))
    w = w.with_columns(
        pl.col("_x").rolling_mean(4, min_samples=4).over("term", **order).alias("_m4"),
        pl.col("_x").rolling_mean(13, min_samples=13).over("term", **order).alias("_m13"),
        pl.col("_x")
        .rolling_mean(BASELINE_WEEKS, min_samples=BASELINE_MIN_WEEKS)
        .over("term", **order)
        .alias("_b52"),
    )
    m4 = pl.col("_m4")

    def lag(k: int) -> pl.Expr:
        return m4.shift(k).over("term", **order)

    exprs = [
        _log_ratio(m4, pl.col("_m13")).alias("sg_level_rel13"),
        (_log_ratio(m4, lag(SLOPE_SHORT)) / SLOPE_SHORT).alias("sg_slope_4w"),
        (_log_ratio(m4, lag(SLOPE_LONG)) / SLOPE_LONG).alias("sg_slope_13w"),
        _log_ratio(m4, pl.col("_b52")).alias("sg_rel_52w"),
    ]
    exprs += [
        (_log_ratio(lag(k), lag(k + SLOPE_SHORT)) / SLOPE_SHORT).alias(f"sg_lead{k}")
        for k in LEAD_WEEKS
    ]
    return w.with_columns(exprs).select(
        "term",
        "week_start",
        *[c for c in SIGNAL_FEATURE_COLS if c not in ("sg_div_4w", "sg_div_13w")],
    )


def sales_slopes(panel: pl.DataFrame) -> pl.DataFrame:
    """Per-(style, week) log sales slopes over 4 and 13 weeks (trailing-4-week mean intensity)."""
    order = {"order_by": "week_start"}
    s = panel.select("style_key", "week_start", "intensity_shrunk").sort("style_key", "week_start")
    s = s.with_columns(
        pl.col("intensity_shrunk")
        .rolling_mean(4, min_samples=4)
        .over("style_key", **order)
        .alias("_s4")
    )
    s4 = pl.col("_s4").log1p()
    return s.with_columns(
        ((s4 - s4.shift(SLOPE_SHORT).over("style_key", **order)) / SLOPE_SHORT).alias("_ss4"),
        ((s4 - s4.shift(SLOPE_LONG).over("style_key", **order)) / SLOPE_LONG).alias("_ss13"),
    ).select("style_key", "week_start", "_ss4", "_ss13")


def build_signal_features(
    panel: pl.DataFrame,
    style_terms: pl.DataFrame,
    weekly: pl.DataFrame,
    origin_weeks: list[date],
    *,
    _lead_weeks: int = 0,
) -> pl.DataFrame:
    """Causal signal features for every `(style_key, origin_week)` with a panel row at that week.

    Args:
        panel: The dense style-week panel (`style_key`, `week_start`, `intensity_shrunk`).
        style_terms: `style_key`, `term` (null term = unmapped style).
        weekly: `term`, `week_start` (Monday, already shifted from the Trends Sunday), `value`.
            Terms that are not usable (`usable_terms`) are dropped before features are built.
        origin_weeks: Origins to build features for.
        _lead_weeks: test-only hook making the signal deliberately non-causal.

    Returns:
        `style_key`, `origin_week` and every column in `SIGNAL_FEATURE_COLS` (null where a style has
        no usable term or its window is incomplete).
    """
    usable = usable_terms(weekly).filter(pl.col("usable"))["term"]
    sig = _signal_weekly_features(weekly.filter(pl.col("term").is_in(usable)), _lead_weeks)

    base = (
        panel.select("style_key", "week_start")
        .join(style_terms.select("style_key", "term"), on="style_key", how="left")
        .join(sig, on=["term", "week_start"], how="left")
        .join(sales_slopes(panel), on=["style_key", "week_start"], how="left")
    )
    base = base.with_columns(
        (pl.col("sg_slope_4w") - pl.col("_ss4")).alias("sg_div_4w"),
        (pl.col("sg_slope_13w") - pl.col("_ss13")).alias("sg_div_13w"),
    )
    out = base.filter(pl.col("week_start").is_in(origin_weeks)).rename(
        {"week_start": "origin_week"}
    )
    return out.select("style_key", "origin_week", *SIGNAL_FEATURE_COLS)


def add_signal_features(
    model_frame: pl.DataFrame,
    panel: pl.DataFrame,
    style_terms: pl.DataFrame,
    weekly: pl.DataFrame,
) -> pl.DataFrame:
    """Join the signal columns onto a `build_model_frame`-shaped frame (additive, order kept)."""
    origin_weeks = model_frame["origin_week"].unique().sort().to_list()
    feats = build_signal_features(panel, style_terms, weekly, origin_weeks)
    return model_frame.join(
        feats, on=["style_key", "origin_week"], how="left", maintain_order="left"
    )
