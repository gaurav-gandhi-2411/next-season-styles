"""Model-ready feature set for the style-week forecasting task, built as-of a given origin week.

`build_features(panel, origin_weeks)` produces one row per `(style_key, origin_week)` pair, for
every `origin_week` requested and every `style_key` that has a densified panel row at that exact
`week_start` (i.e. `first_week_seen <= origin_week <= last_week_seen` for that style -- the panel
is dense within a style's own lifetime, see `nss.features.style_panel`, so this is equivalent to
"the style already existed as of that origin"). Styles with no row at a requested origin (not yet
launched, or already past `last_week_seen`) are simply absent from that origin's output rows --
there is no well-defined point-in-time feature vector to compute for them at that origin.

CAUSAL-SAFETY DESIGN, THE POINT OF THIS MODULE: every feature must be computable using ONLY panel
rows with `week_start <= origin_week` for that same `style_key` (plus, for the two "share of
parent group" features, contemporaneous-or-earlier data from OTHER style_keys in the same group --
see SHARE OF PARENT GROUP below for why even that is restricted to strictly-before-origin data).

The implementation strategy that makes this straightforward to get right (and to verify): every
per-row engineered column below is computed ONCE, densely, across the *entire* input panel using
window functions ordered by `week_start` within `style_key` (`.over("style_key",
order_by="week_start")`) or within a group (`.over(group_cols, order_by="week_start")`). A window
function ordered by `week_start` that only reads `shift`/`cum_sum`/`ewm_mean`/etc. of the SAME
column is, by construction, a function of that row and earlier rows only -- it cannot read a later
row's value no matter which row is later selected as "the origin row" for a given `(style_key,
origin_week)` pair. `build_features` therefore computes every column across the whole panel first,
and only at the very end filters down to the requested `origin_weeks` (`week_start.is_in(...)`).
This is the same pattern `nss.features.style_panel.add_price_index` and `add_intensity_shrunk`
already use (compute densely and causally across the whole panel, filter/select afterward) --
extended here to lags, EWMAs, slopes, and share-of-group features.

LAG CONVENTION: `lag_1` = the value of `intensity_shrunk` AT `origin_week` itself (the most
recently observed week, i.e. `shift(0)`); `lag_L` = the value at `origin_week - (L-1)` weeks (i.e.
`shift(L-1)`), for `L` in `LAG_WEEKS` = [1, 2, 4, 8, 13, 52]. `lag_52` is NULLABLE by design --
most styles do not have 52 weeks of prior history -- rows are never dropped for a null `lag_52`;
LightGBM (the target downstream model) handles null/NaN natively.

EWMA: exponentially-weighted moving average of `intensity_shrunk`, computed causally
(`ewm_mean(..., ignore_nulls=True).over("style_key", order_by="week_start")` -- a strictly
recursive/expanding computation that only uses the current and earlier rows within a style) at two
half-lives: `EWMA_HALF_LIFE_SHORT_WEEKS` = 4 (recent-momentum signal, roughly "the last month"),
and `EWMA_HALF_LIFE_LONG_WEEKS` = 13 (medium-term level, matching the forecast horizon itself --
`nss.features.targets.HORIZON_WEEKS`). Uses polars' default `adjust=True` weighting convention
(the standard pandas/polars EWMA definition), and row-index (not calendar-time) half-life units,
which is exact here because the panel is densified to exactly one row per calendar week per style
(no gaps), so a row-index half-life of 4 is precisely a 4-*week* half-life.

SLOPES: `slope_Lw = (intensity_shrunk_now - intensity_shrunk_{L weeks ago}) / L`, a simple 2-point
finite difference (NOT a full OLS regression over all `L + 1` intervening points). Chosen over
OLS for simplicity (rule 103: simplest option that satisfies the constraints) and because it only
requires the two endpoint weeks to be non-null, rather than requiring a full non-null window --
consistent with this project's general preference for leaving partial history null rather than
imputing (see `style_panel.py`'s `price_index` / `intensity_shrunk` null-handling). Computed at
`L` = 4 and `L` = 13 weeks (`SLOPE_WINDOWS_WEEKS`), matching the two EWMA half-lives above.

N_ACTIVE_ARTICLES / PRICE_INDEX LEVEL + TREND: "level" is simply the value observed at
`origin_week` itself (already causal -- it is the origin row's own, already-observed value).
"Trend" reuses the identical 2-point finite-difference slope definition as above, over a single
trailing window of `TREND_WINDOW_WEEKS` = 13 weeks for both columns (chosen to match the long EWMA
half-life / forecast horizon, for one consistent "medium-term trend" definition across every
trended feature in this module, rather than introducing a third distinct window size).

WEEKS SINCE FIRST SEEN: `(origin_week - first_week_seen).days // 7`. Trivially causal -- a
function of the style's own already-known launch date and the origin week alone.

WEEK-OF-YEAR FOURIER TERMS: 2 harmonics of day-of-year (`pl.col("week_start").dt.ordinal_day()`),
`sin`/`cos` of `2 * pi * harmonic * doy / DAYS_PER_YEAR` for `harmonic` in `{1, 2}`. Day-of-year
(not ISO week-of-year) chosen as the more standard basis for Fourier seasonality terms, and
`DAYS_PER_YEAR = 365.25` (not 365) to average out leap-year drift over the multi-year panel.
Trivially causal -- a deterministic function of `origin_week` alone.

SHARE OF PARENT GROUP (`share_index_group`, `share_garment_group`): the style's own share of its
parent `index_group_name` / `garment_group_name` group's total `units`, computed with a TRAILING,
expanding, current-week-EXCLUDED window -- i.e. `share = sum(this style's units over weeks
STRICTLY BEFORE origin_week) / sum(the whole group's units over weeks STRICTLY BEFORE
origin_week)`, reusing the exact causal pattern (and rationale) of
`nss.features.style_panel.add_intensity_shrunk`'s `group_mean_trailing`: that function's docstring
explains in detail why a SAME-WEEK cross-sectional aggregate across other styles is NOT actually
knowable at prediction time (other styles' current-week sales are exactly as unobserved as this
style's own future, in a real rolling-origin forecast) and must instead be a trailing,
current-excluded statistic. The group total here is built the same way as that function's
`group_week` + as-of-join-forward-carry construction (generalized to a SUM instead of a MEAN, and
without the "active-only" filter, since a sum is well-defined over the full group-week grid
including zero-sale weeks). Both numerator and denominator use the identical strictly-before-origin
window, so the ratio is dimensionally consistent (trailing-share-of-trailing-total, not
current-week-value-over-trailing-total). Nullable only when the group has no trailing history at
all yet (mirrors `intensity_shrunk`'s null semantics for the same reason); a style with no own
trailing history yet (its own first observed week) contributes 0 to the numerator, not null --
"zero prior weeks of history" is a known, computable quantity (0 units), unlike "the group's prior
history does not exist yet" which is genuinely unknown.

CATEGORICAL ATTRIBUTES FOR COLD-START TRANSFER: the 5 style_key attribute columns
(`nss.features.style_panel.STYLE_KEY_COLS`) are included as native `pl.Categorical` columns (not
one-hot expanded) -- LightGBM consumes `category`/`Categorical` dtype columns directly without
requiring one-hot encoding, which avoids exploding the feature space for the higher-cardinality
columns (e.g. `product_type_name`). No target encoding is used (that would leak the target into
the features); this is a plain, target-free categorical representation whose sole purpose is
letting a model transfer knowledge to short-history / cold-start styles via their shared
attributes.

See `tests/test_model_features.py::test_build_features_is_causally_safe` for the mandatory test
that actively shuffles all future (`week_start > origin_week`) data and confirms every feature is
unchanged, alongside a negative control that shuffling SAME-STYLE pre-origin history DOES change
the lag/EWMA/slope features that depend on it (proving the test is actually sensitive).
"""

from __future__ import annotations

import math
from datetime import date

import polars as pl

from nss.features.style_panel import STYLE_KEY_COLS

LAG_WEEKS: list[int] = [1, 2, 4, 8, 13, 52]

EWMA_HALF_LIFE_SHORT_WEEKS = 4
EWMA_HALF_LIFE_LONG_WEEKS = 13

SLOPE_WINDOWS_WEEKS: list[int] = [4, 13]

TREND_WINDOW_WEEKS = 13

FOURIER_HARMONICS: list[int] = [1, 2]
DAYS_PER_YEAR = 365.25

_SHARE_GROUP_SPECS: list[tuple[list[str], str]] = [
    (["index_group_name"], "share_index_group"),
    (["garment_group_name"], "share_garment_group"),
]


def _add_lags(panel: pl.DataFrame) -> pl.DataFrame:
    """Add `lag_1..lag_52` of `intensity_shrunk`. See module docstring LAG CONVENTION."""
    return panel.with_columns(
        [
            pl.col("intensity_shrunk")
            .shift(lag - 1)
            .over("style_key", order_by="week_start")
            .alias(f"lag_{lag}")
            for lag in LAG_WEEKS
        ]
    )


def _add_ewma(panel: pl.DataFrame) -> pl.DataFrame:
    """Add the two causal EWMA columns. See module docstring EWMA section."""
    return panel.with_columns(
        ewma_halflife_4w=pl.col("intensity_shrunk")
        .ewm_mean(half_life=EWMA_HALF_LIFE_SHORT_WEEKS, ignore_nulls=True)
        .over("style_key", order_by="week_start"),
        ewma_halflife_13w=pl.col("intensity_shrunk")
        .ewm_mean(half_life=EWMA_HALF_LIFE_LONG_WEEKS, ignore_nulls=True)
        .over("style_key", order_by="week_start"),
    )


def _slope_expr(value_col: str, window: int) -> pl.Expr:
    """2-point finite-difference slope of `value_col` over `window` trailing weeks, causal.

    See module docstring SLOPES section for why a 2-point difference (not OLS) was chosen.
    """
    now = pl.col(value_col)
    then = pl.col(value_col).shift(window).over("style_key", order_by="week_start")
    return (now - then) / window


def _add_slopes(panel: pl.DataFrame) -> pl.DataFrame:
    """Add `slope_4w` / `slope_13w` of `intensity_shrunk`, and the trend columns for
    `n_active_articles` and `price_index` (see module docstring for both sections).
    """
    return panel.with_columns(
        [_slope_expr("intensity_shrunk", w).alias(f"slope_{w}w") for w in SLOPE_WINDOWS_WEEKS]
        + [
            _slope_expr("n_active_articles", TREND_WINDOW_WEEKS).alias(
                f"n_active_articles_trend_{TREND_WINDOW_WEEKS}w"
            ),
            _slope_expr("price_index", TREND_WINDOW_WEEKS).alias(
                f"price_index_trend_{TREND_WINDOW_WEEKS}w"
            ),
        ]
    )


def _add_levels(panel: pl.DataFrame) -> pl.DataFrame:
    """Add the trivially-causal `_level` aliases (the origin row's own observed value)."""
    return panel.with_columns(
        n_active_articles_level=pl.col("n_active_articles"),
        price_index_level=pl.col("price_index"),
    )


def _add_weeks_since_first_seen(panel: pl.DataFrame) -> pl.DataFrame:
    """Add `weeks_since_first_seen`. See module docstring WEEKS SINCE FIRST SEEN section."""
    return panel.with_columns(
        weeks_since_first_seen=(
            (pl.col("week_start") - pl.col("first_week_seen")).dt.total_days() // 7
        )
    )


def _add_fourier_terms(panel: pl.DataFrame) -> pl.DataFrame:
    """Add the 2-harmonic day-of-year Fourier terms. See module docstring FOURIER section."""
    doy = pl.col("week_start").dt.ordinal_day()
    exprs = []
    for h in FOURIER_HARMONICS:
        angle = 2 * math.pi * h * doy / DAYS_PER_YEAR
        exprs.append(angle.sin().alias(f"fourier_sin_{h}"))
        exprs.append(angle.cos().alias(f"fourier_cos_{h}"))
    return panel.with_columns(exprs)


def _trailing_group_sum_strictly_before(
    panel: pl.DataFrame, group_cols: list[str], value_col: str, out_col: str
) -> pl.DataFrame:
    """Attach a TRAILING (expanding, current-week-EXCLUDED) group sum of `value_col`.

    For each row, `out_col` = sum of `value_col` across ALL style_keys sharing the row's own
    `group_cols` group, over all weeks STRICTLY BEFORE that row's `week_start`. This is the same
    causal as-of-join pattern as `nss.features.style_panel.add_intensity_shrunk`'s
    `group_mean_trailing` (see that function's docstring for the full rationale), generalized to a
    SUM (no "active-only" filter needed -- a sum is well-defined over the full group-week grid,
    including zero-sale weeks contributing 0).

    Args:
        panel: The dense style-week panel (or any subset with `group_cols`, `week_start`,
            `value_col`).
        group_cols: Columns defining the group (e.g. `["index_group_name"]`).
        value_col: Column to sum within the group (e.g. `"units"`).
        out_col: Name of the output column to attach.

    Returns:
        `panel` with `out_col` added: the group's cumulative `value_col` total strictly before
        each row's `week_start`, forward-carried across weeks where the group itself had no rows
        (same forward-carry-via-as-of-join fix as `add_intensity_shrunk`), and null only for rows
        strictly before the group's very first ever row in the whole dataset.
    """
    group_week = (
        panel.group_by([*group_cols, "week_start"])
        .agg(_group_week_value=pl.col(value_col).sum())
        .sort([*group_cols, "week_start"])
    )
    group_week = group_week.with_columns(
        _cum_inclusive=pl.col("_group_week_value").cum_sum().over(group_cols, order_by="week_start")
    ).select([*group_cols, "week_start", "_cum_inclusive"])

    probe = (
        panel.select([*group_cols, "week_start"])
        .with_row_index("_row_id")
        .with_columns((pl.col("week_start") - pl.duration(days=1)).alias("_probe_week"))
        .sort([*group_cols, "_probe_week"])
    )
    trailing = (
        probe.join_asof(
            group_week.sort([*group_cols, "week_start"]),
            left_on="_probe_week",
            right_on="week_start",
            by=group_cols,
            strategy="backward",
        )
        .select(["_row_id", "_cum_inclusive"])
        .rename({"_cum_inclusive": out_col})
    )
    return (
        panel.with_row_index("_row_id")
        .join(trailing, on="_row_id", how="left", maintain_order="left")
        .drop("_row_id")
    )


def _add_share_of_parent_group(panel: pl.DataFrame) -> pl.DataFrame:
    """Add `share_index_group` / `share_garment_group`. See module docstring SHARE OF PARENT
    GROUP section for the full trailing-window rationale.
    """
    panel = panel.with_columns(
        _style_trailing_units=pl.col("units")
        .cum_sum()
        .shift(1)
        .over("style_key", order_by="week_start")
        .fill_null(0)
    )
    for group_cols, out_col in _SHARE_GROUP_SPECS:
        trailing_col = f"_{out_col}_group_total"
        panel = _trailing_group_sum_strictly_before(panel, group_cols, "units", trailing_col)
        panel = panel.with_columns(
            pl.when(pl.col(trailing_col) > 0)
            .then(pl.col("_style_trailing_units") / pl.col(trailing_col))
            .otherwise(None)
            .alias(out_col)
        )
        panel = panel.drop(trailing_col)
    return panel.drop("_style_trailing_units")


_FEATURE_COLS: list[str] = (
    [f"lag_{lag}" for lag in LAG_WEEKS]
    + ["ewma_halflife_4w", "ewma_halflife_13w"]
    + [f"slope_{w}w" for w in SLOPE_WINDOWS_WEEKS]
    + [
        "n_active_articles_level",
        f"n_active_articles_trend_{TREND_WINDOW_WEEKS}w",
        "price_index_level",
        f"price_index_trend_{TREND_WINDOW_WEEKS}w",
        "share_index_group",
        "share_garment_group",
        "weeks_since_first_seen",
    ]
    + [f"fourier_sin_{h}" for h in FOURIER_HARMONICS]
    + [f"fourier_cos_{h}" for h in FOURIER_HARMONICS]
)

_REQUIRED_INPUT_COLS: list[str] = [
    "style_key",
    *STYLE_KEY_COLS,
    "week_start",
    "first_week_seen",
    "units",
    "n_active_articles",
    "price_index",
    "intensity_shrunk",
]

_OUTPUT_COLS: list[str] = ["style_key", "origin_week", *STYLE_KEY_COLS, *_FEATURE_COLS]


def build_features(panel: pl.DataFrame, origin_weeks: list[date]) -> pl.DataFrame:
    """Build the full causal feature set for every `(style_key, origin_week)` pair.

    See the module docstring for the exact definition, causal-safety rationale, and every design
    judgment call (lag convention, EWMA half-lives, slope method, trend window, Fourier basis,
    share-of-group window, categorical encoding) behind each feature family.

    Args:
        panel: The dense style-week panel (`nss.features.style_panel`'s output, or any
            subset/superset with at least the columns in `_REQUIRED_INPUT_COLS`).
        origin_weeks: The origin weeks (Monday `week_start` values) to build features for.

    Returns:
        One row per `(style_key, origin_week)` pair, for every `origin_week` in `origin_weeks` and
        every `style_key` with a panel row at that exact `week_start` (i.e. the style already
        existed as of that origin -- see module docstring for why this is the eligibility rule).
        Columns: `style_key`, `origin_week`, the 5 `STYLE_KEY_COLS` attribute columns (cast to
        `pl.Categorical`), and every feature in `_FEATURE_COLS`.
    """
    missing = set(_REQUIRED_INPUT_COLS) - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing required columns: {sorted(missing)}")

    working = panel.select(_REQUIRED_INPUT_COLS)

    working = _add_lags(working)
    working = _add_ewma(working)
    working = _add_slopes(working)
    working = _add_levels(working)
    working = _add_weeks_since_first_seen(working)
    working = _add_fourier_terms(working)
    working = _add_share_of_parent_group(working)

    out = working.filter(pl.col("week_start").is_in(origin_weeks)).rename(
        {"week_start": "origin_week"}
    )
    out = out.with_columns([pl.col(c).cast(pl.Categorical) for c in STYLE_KEY_COLS])
    return out.select(_OUTPUT_COLS)
