"""Cross-style neighbourhood features: is a style's cohort in the attribute lattice rising?

Every feature in `nss.features.model_features` derives from a style's OWN history. These features
encode whether the style's NEIGHBOURS -- other styles sharing part of its attribute key -- are
climbing, the trend-propagation hypothesis ("beige knitwear is rising across five product types,
which is signal for the sixth").

THREE NEIGHBOURHOODS (`NEIGHBOURHOODS`), each a set of SIBLING styles excluding the style itself:

- `ptgg`   same `product_type_name` + `garment_group_name`, all OTHER colours (siblings differ in
           `perceived_colour_master_name`; graphical appearance is free);
- `colgfx` same `perceived_colour_master_name` + `graphical_appearance_name`, all OTHER product
           types (siblings differ in `product_type_name`);
- `igg`    same `index_group_name` + `garment_group_name`, every other style (siblings differ in
           any other attribute; only the style itself is excluded).

FIVE FEATURES PER NEIGHBOURHOOD (`nb_<tag>_...`):

- `level_mean`: mean `intensity_shrunk` over siblings at the origin week (siblings with a null
  value are skipped);
- `slope_4w`, `slope_13w`: mean over siblings of each sibling's OWN 2-point finite-difference slope
  of `intensity_shrunk` (`(now - L weeks ago) / L`, the same definition as
  `model_features.slope_4w/13w`), so the cohort trend uses only siblings that have both endpoints;
- `ratio`: the style's own `intensity_shrunk` divided by `level_mean` (> 1: leading its cohort,
  < 1: lagging); null when `level_mean` <= `RATIO_MIN_DENOMINATOR` or either side is null;
- `breadth`: number of ACTIVE sibling styles, i.e. siblings with a panel row at the origin week and
  `n_active_articles > 0`.

CAUSAL CONVENTION (identical to `model_features`): the value at `origin_week` uses panel rows with
`week_start <= origin_week` only. Each sibling's level / slope at week t is a function of that
sibling's own rows at t, t-4, t-13 (`intensity_shrunk` is itself causal by construction, see
`style_panel.add_intensity_shrunk`); the cohort statistic then aggregates those already-causal
per-style values across styles WITHIN THE SAME WEEK (a same-week cross-section of values that are
each knowable as of that week -- no cross-week look-ahead is possible because the aggregation window
is a single `week_start`). Everything is computed densely over the whole panel with window
aggregations partitioned by (group, week_start) and only then filtered to the requested origins,
the same compute-then-filter pattern as `model_features.build_features`. Sibling statistics are
"group total minus excluded sub-group total" so the style itself (and, for `ptgg`/`colgfx`, all
same-colour / same-product-type styles) is excluded exactly. Sibling sets are the styles that have
a panel row at the origin week (the panel is dense within a style's lifetime).

`add_neighbourhood_features` joins the columns ADDITIVELY onto `lightgbm_model.build_model_frame`'s
output; nothing in the existing feature pipeline is changed. Causality is tested in
`tests/test_neighbourhood_features.py` (post-origin shuffling leaves every column unchanged).
"""

from __future__ import annotations

from datetime import date

import polars as pl

from nss.features.model_features import SLOPE_WINDOWS_WEEKS

# tag -> (group columns shared with siblings, extra column that DEFINES the excluded sub-group).
# Siblings = same group, minus the rows sharing `excluded_col` with the style (which includes the
# style itself). For "igg" the excluded column is `style_key`, i.e. only the style itself.
NEIGHBOURHOODS: dict[str, tuple[list[str], str]] = {
    "ptgg": (["product_type_name", "garment_group_name"], "perceived_colour_master_name"),
    "colgfx": (["perceived_colour_master_name", "graphical_appearance_name"], "product_type_name"),
    "igg": (["index_group_name", "garment_group_name"], "style_key"),
}

# A sibling mean at or below this makes the own/cohort ratio meaningless (division by ~0).
RATIO_MIN_DENOMINATOR = 1e-9

_REQUIRED_COLS: list[str] = [
    "style_key",
    "index_group_name",
    "product_type_name",
    "garment_group_name",
    "perceived_colour_master_name",
    "graphical_appearance_name",
    "week_start",
    "n_active_articles",
    "intensity_shrunk",
]

NEIGHBOURHOOD_FEATURE_COLS: list[str] = [
    f"nb_{tag}_{name}"
    for tag in NEIGHBOURHOODS
    for name in (
        "level_mean",
        *[f"slope_{w}w" for w in SLOPE_WINDOWS_WEEKS],
        "ratio",
        "breadth",
    )
]


def _sibling_sum_count(
    value: pl.Expr, broad_keys: list[str], excluded_keys: list[str]
) -> tuple[pl.Expr, pl.Expr]:
    """(sum, non-null count) of `value` over siblings = broad group minus the excluded sub-group."""
    total = value.sum().over(broad_keys) - value.sum().over(excluded_keys)
    count = value.is_not_null().sum().over(broad_keys) - value.is_not_null().sum().over(
        excluded_keys
    )
    return total, count


def build_neighbourhood_features(panel: pl.DataFrame, origin_weeks: list[date]) -> pl.DataFrame:
    """Causal cross-style neighbourhood features for every `(style_key, origin_week)` pair.

    See the module docstring for the definitions and the causal convention.

    Args:
        panel: The dense style-week panel (at least the columns in `_REQUIRED_COLS`).
        origin_weeks: The origin weeks (Monday `week_start` values) to build features for.

    Returns:
        One row per `(style_key, origin_week)` for every style with a panel row at that exact week
        (the same eligibility rule as `build_features`), columns `style_key`, `origin_week` and
        every column in `NEIGHBOURHOOD_FEATURE_COLS`.
    """
    missing = set(_REQUIRED_COLS) - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing required columns: {sorted(missing)}")

    w = panel.select(_REQUIRED_COLS).sort(["style_key", "week_start"])
    w = w.with_columns(
        [
            ((pl.col("intensity_shrunk") - pl.col("intensity_shrunk").shift(win)) / win)
            .over("style_key", order_by="week_start")
            .alias(f"_own_slope_{win}w")
            for win in SLOPE_WINDOWS_WEEKS
        ]
        + [(pl.col("n_active_articles") > 0).cast(pl.Int64).alias("_active")]
    )

    value_cols = {
        "level_mean": "intensity_shrunk",
        **{f"slope_{win}w": f"_own_slope_{win}w" for win in SLOPE_WINDOWS_WEEKS},
    }
    exprs: list[pl.Expr] = []
    for tag, (group_cols, excluded_col) in NEIGHBOURHOODS.items():
        broad = [*group_cols, "week_start"]
        excluded = [*group_cols, excluded_col, "week_start"]
        for name, col in value_cols.items():
            total, count = _sibling_sum_count(pl.col(col), broad, excluded)
            exprs.append(
                pl.when(count > 0).then(total / count).otherwise(None).alias(f"nb_{tag}_{name}")
            )
        active_total, _ = _sibling_sum_count(pl.col("_active"), broad, excluded)
        exprs.append(active_total.cast(pl.Int64).alias(f"nb_{tag}_breadth"))
    w = w.with_columns(exprs)
    w = w.with_columns(
        [
            pl.when(pl.col(f"nb_{tag}_level_mean") > RATIO_MIN_DENOMINATOR)
            .then(pl.col("intensity_shrunk") / pl.col(f"nb_{tag}_level_mean"))
            .otherwise(None)
            .alias(f"nb_{tag}_ratio")
            for tag in NEIGHBOURHOODS
        ]
    )
    out = w.filter(pl.col("week_start").is_in(origin_weeks)).rename({"week_start": "origin_week"})
    return out.select(["style_key", "origin_week", *NEIGHBOURHOOD_FEATURE_COLS])


def add_neighbourhood_features(model_frame: pl.DataFrame, panel: pl.DataFrame) -> pl.DataFrame:
    """Join the neighbourhood columns onto a `build_model_frame`-shaped frame (additive).

    Row order and every existing column of `model_frame` are unchanged (left join,
    `maintain_order="left"`, so LightGBM's training-row order keeps its determinism guarantee); the
    `NEIGHBOURHOOD_FEATURE_COLS` are appended at the end.
    """
    origin_weeks = model_frame["origin_week"].unique().sort().to_list()
    nb = build_neighbourhood_features(panel, origin_weeks)
    return model_frame.join(nb, on=["style_key", "origin_week"], how="left", maintain_order="left")
