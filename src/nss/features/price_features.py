"""Price-elasticity and full-price-demand features, computed strictly causally from the panel.

`model_features` already carries `price_index_level` (the origin row's own `price_index`) and
`price_index_trend_13w` (its 13-week 2-point slope). Those two are NOT duplicated here; this module
adds what the current feature set lacks -- how demand responds to price, and how well it holds at
full price. Every feature is a function of the style's own rows in the trailing
`WINDOW_WEEKS` (13) weeks ending at (and including) the origin week.

FEATURES (`PRICE_FEATURE_COLS`):

- `pr_elasticity_13w`: OLS slope of `ln(intensity_shrunk)` on `ln(mean_price)` over the window's
  weeks where both are positive (own-price elasticity of intensity). Null if fewer than
  `MIN_ELASTICITY_POINTS` weeks qualify or `ln(mean_price)` has a standard deviation below
  `MIN_LOG_PRICE_STD` (a regression on a flat price is not an elasticity);
- `pr_price_index_slope_4w`: `(price_index_now - price_index_{4 weeks ago}) / 4`, the same 2-point
  finite difference as `model_features` (the 13-week counterpart already exists there);
- `pr_full_price_intensity_13w`: mean `intensity_shrunk` over the window's weeks with
  `price_index >= FULL_PRICE_THRESHOLD` (0.95); null if there is no such week;
- `pr_discount_depth_13w`: mean of `max(0, 1 - price_index)` over the window's weeks with a
  `price_index` (0 in full-price weeks: an average depth, not depth-when-discounted);
- `pr_discount_freq_13w`: share of those weeks with `price_index < FULL_PRICE_THRESHOLD`;
- `pr_full_price_holding_ratio_13w`: full-price intensity / overall intensity, both over the weeks
  with a `price_index` (> 1: the style sells better at full price than on average; null if there is
  no full-price week or the overall mean is not positive).

`price_index` is null for a style's first 52 weeks (`style_panel.add_price_index`), so the price
features are null for young styles; nothing is imputed.

CAUSAL CONVENTION: the panel is dense (one row per style per calendar week inside a style's
lifetime), so a within-style row shift of `k` is exactly `k` weeks. Windows are sums of
`shift(0..12)` over `(style_key, order_by=week_start)`, which reads the row and earlier rows only,
computed densely over the panel and filtered to the origins afterwards, like `model_features`.
The tests in `tests/test_price_features.py` shuffle every post-origin value and
require the features to be unchanged, and require a deliberately leaky variant
(`_window_lead_weeks`, a test-only hook that slides the window into the future) to FAIL that same
test.
"""

from __future__ import annotations

from datetime import date

import polars as pl

WINDOW_WEEKS = 13
SLOPE_WEEKS = 4
FULL_PRICE_THRESHOLD = 0.95
MIN_ELASTICITY_POINTS = 6
MIN_LOG_PRICE_STD = 0.01  # a ~1% price spread; below this the OLS slope is numerical noise
MIN_PRICE_WEEKS = 4  # weeks with a price_index needed for the discount statistics

PRICE_FEATURE_COLS: list[str] = [
    "pr_elasticity_13w",
    "pr_price_index_slope_4w",
    "pr_full_price_intensity_13w",
    "pr_discount_depth_13w",
    "pr_discount_freq_13w",
    "pr_full_price_holding_ratio_13w",
]

_REQUIRED_COLS: list[str] = [
    "style_key",
    "week_start",
    "mean_price",
    "price_index",
    "intensity_shrunk",
]


def _window_sum(col: str, lead: int) -> pl.Expr:
    """Sum of `col` over the trailing window (nulls count as 0), per style, in week order.

    `lead` slides the window into the future (`shift` with a negative offset) and exists only for
    the deliberately leaky test variant.
    """
    shifted = [
        pl.col(col).shift(k - lead).over("style_key", order_by="week_start")
        for k in range(WINDOW_WEEKS)
    ]
    return pl.sum_horizontal(shifted)


def build_price_features(
    panel: pl.DataFrame, origin_weeks: list[date], *, _window_lead_weeks: int = 0
) -> pl.DataFrame:
    """Causal price features for every `(style_key, origin_week)` pair. See module docstring.

    Args:
        panel: The dense style-week panel (at least the columns in `_REQUIRED_COLS`).
        origin_weeks: The origin weeks to build features for.
        _window_lead_weeks: test-only hook making the features deliberately non-causal.

    Returns:
        One row per `(style_key, origin_week)` for every style with a panel row at that week,
        columns `style_key`, `origin_week` and every column in `PRICE_FEATURE_COLS`.
    """
    missing = set(_REQUIRED_COLS) - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing required columns: {sorted(missing)}")
    lead = _window_lead_weeks

    w = panel.select(_REQUIRED_COLS).sort(["style_key", "week_start"])
    intensity = pl.col("intensity_shrunk")
    price = pl.col("mean_price")
    pidx = pl.col("price_index")
    reg_ok = (intensity > 0) & (price > 0)
    has_pidx = pidx.is_not_null()
    full = has_pidx & (pidx >= FULL_PRICE_THRESHOLD)
    w = w.with_columns(
        pl.when(reg_ok).then(price.log()).otherwise(0.0).alias("_x"),
        pl.when(reg_ok).then(intensity.log()).otherwise(0.0).alias("_y"),
        reg_ok.fill_null(False).cast(pl.Float64).alias("_ok"),
        has_pidx.cast(pl.Float64).alias("_hp"),
        full.cast(pl.Float64).alias("_full"),
        pl.when(has_pidx).then((1.0 - pidx).clip(lower_bound=0.0)).otherwise(0.0).alias("_depth"),
        pl.when(has_pidx).then(intensity).otherwise(None).fill_null(0.0).alias("_int_hp"),
        pl.when(full).then(intensity).otherwise(None).fill_null(0.0).alias("_int_full"),
    )
    w = w.with_columns(
        (pl.col("_x") ** 2).alias("_xx"),
        (pl.col("_x") * pl.col("_y")).alias("_xy"),
    )
    sums = {
        name: _window_sum(name, lead)
        for name in (
            "_x",
            "_y",
            "_xx",
            "_xy",
            "_ok",
            "_hp",
            "_full",
            "_depth",
            "_int_hp",
            "_int_full",
        )
    }
    w = w.with_columns([expr.alias(f"{name}_w") for name, expr in sums.items()])

    n = pl.col("_ok_w")
    var_x = pl.col("_xx_w") / n - (pl.col("_x_w") / n) ** 2  # population variance of ln(price)
    cov_xy = pl.col("_xy_w") / n - (pl.col("_x_w") / n) * (pl.col("_y_w") / n)
    elasticity = (
        pl.when((n >= MIN_ELASTICITY_POINTS) & (var_x >= MIN_LOG_PRICE_STD**2))
        .then(cov_xy / var_x)
        .otherwise(None)
    )
    n_hp = pl.col("_hp_w")
    n_full = pl.col("_full_w")
    full_intensity = pl.when(n_full > 0).then(pl.col("_int_full_w") / n_full).otherwise(None)
    overall = pl.when(n_hp >= MIN_PRICE_WEEKS).then(pl.col("_int_hp_w") / n_hp).otherwise(None)
    w = w.with_columns(
        elasticity.alias("pr_elasticity_13w"),
        (
            (pidx - pidx.shift(SLOPE_WEEKS - lead)).over("style_key", order_by="week_start")
            / SLOPE_WEEKS
        ).alias("pr_price_index_slope_4w"),
        pl.when(n_hp >= MIN_PRICE_WEEKS)
        .then(full_intensity)
        .otherwise(None)
        .alias("pr_full_price_intensity_13w"),
        pl.when(n_hp >= MIN_PRICE_WEEKS)
        .then(pl.col("_depth_w") / n_hp)
        .otherwise(None)
        .alias("pr_discount_depth_13w"),
        pl.when(n_hp >= MIN_PRICE_WEEKS)
        .then(1.0 - n_full / n_hp)
        .otherwise(None)
        .alias("pr_discount_freq_13w"),
        pl.when((overall > 1e-9) & full_intensity.is_not_null())
        .then(full_intensity / overall)
        .otherwise(None)
        .alias("pr_full_price_holding_ratio_13w"),
    )
    out = w.filter(pl.col("week_start").is_in(origin_weeks)).rename({"week_start": "origin_week"})
    return out.select(["style_key", "origin_week", *PRICE_FEATURE_COLS])


def add_price_features(model_frame: pl.DataFrame, panel: pl.DataFrame) -> pl.DataFrame:
    """Join the price columns onto a `build_model_frame`-shaped frame (additive, order kept)."""
    origin_weeks = model_frame["origin_week"].unique().sort().to_list()
    feats = build_price_features(panel, origin_weeks)
    return model_frame.join(
        feats, on=["style_key", "origin_week"], how="left", maintain_order="left"
    )
