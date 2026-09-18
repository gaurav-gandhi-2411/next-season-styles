"""Final production forecast: train on all available data, forecast AW2020, apply the selection
rule, and surface local (per-style) + seasonal-bonus rankings.

This module is deliberately separate from `nss.models.lightgbm_model` (the walk-forward-evaluated
model): everything here runs AFTER the honest walk-forward evaluation is already complete and
reported, and none of it feeds back into that evaluation's numbers. See the walk-forward module's
own docstring for the model itself (hyperparameters, the numpy-matrix ENVIRONMENT GOTCHA, SHAP
mechanics) -- this module reuses that module's `train_lightgbm` / `predict_lightgbm` /
`build_model_frame` / `feature_columns` / `_to_lgb_matrix` verbatim rather than duplicating them.

DISCREPANCY A (brief says "rank by predicted shrunk intensity"; there is no model that predicts
`intensity_shrunk` directly): the actually-trained model predicts `log1p(mean(units_per_active_
article over the forward 13-week window))` (see `nss.features.targets.compute_forward_target`);
`intensity_shrunk` is only ever a causal-safe INPUT FEATURE to that model (see
`nss.features.style_panel.add_intensity_shrunk`), never its target. RESOLUTION (this module):
"predicted intensity" operationally means `expm1(model's raw log1p prediction)` -- i.e. the
model's own predicted mean `units_per_active_article` over the forecast window, inverted off the
log1p training scale. This is NOT literally `intensity_shrunk`; ranking by it is the closest
faithful reading of the brief given what was actually built. See `predict_intensity`.

DISCREPANCY B (brief says "require forecast-window price_index >= 0.85"; the forecast window
2020-09-22 onward has not happened, and no price-forecasting model is in scope): RESOLUTION (this
module): guard 2 uses the style's most-recently-OBSERVED `price_index` as of the forecast origin
(2020-09-21, the panel's last real week) -- i.e. `price_index_level` from `build_features` at the
forecast origin itself, already a causal, already-observed value; not a forecast. See
`GUARD2_MIN_PRICE_INDEX` / `build_ranking_frame`.

DISCREPANCY C (bonus brief's parenthetical season definition, "Spring=Mar-Jun, Summer=Jun-Aug",
is internally inconsistent -- June appears in both Spring and Summer, and the brief in the same
breath asks for "standard meteorological, not astronomical" boundaries, which by definition are
four non-overlapping 3-month blocks). RESOLUTION (this module): `SEASON_MONTHS` uses the actual
standard meteorological definition (Winter=Dec/Jan/Feb, Spring=Mar/Apr/May, Summer=Jun/Jul/Aug,
Autumn=Sep/Oct/Nov) -- the reading consistent with "standard meteorological", not the brief's own
overlapping month list, which is treated as a typo rather than followed literally.

GUARD WINDOW CONVENTION (JUDGMENT CALL): all "last N observed weeks" guard windows (guard 1's
13-week trailing mean, guard 3's 52-week trailing viability count) are INCLUSIVE of the forecast
origin week itself. The origin week (2020-09-21) is the panel's last REAL, already-happened week --
not a future one -- so "the last 13/52 observed weeks as of the forecast origin" naturally includes
it, consistent with how `price_index_level` / `n_active_articles_level` (the origin row's own
already-observed value) are treated elsewhere in this codebase (see
`nss.features.model_features` LEVEL sections). Implemented via `pl.Expr.rolling_mean` /
`rolling_sum` with `window_size=N, min_samples=1` (current row is the window's last element by
polars convention) -- a style with fewer than N observed weeks total simply uses however many it
has, which is the same "leave partial windows as partial, never impute" convention already used
throughout this project (e.g. `nss.features.style_panel.add_price_index`'s trailing-median window).

GUARD 1 AGGREGATION (JUDGMENT CALL, per the task brief): mean (not sum) of `n_active_articles`
over the trailing 13-week window -- "n_active_articles >= 10" reads as a LEVEL check ("does this
style currently comprise a meaningfully sized assortment"), not a cumulative count, so summing
13 weeks of article-counts would answer a different question (roughly "13x the level check").

TRAINING DATA WIDTH (JUDGMENT CALL): the final model is trained on the WIDE origin set -- every
ISO week from the panel's earliest week through the last origin with a full forward target window
(`generate_origin_schedule(panel, step_weeks=1, burn_in_weeks=0)`, 93 origins, 2018-09-17 ..
2020-06-22, ~274k pooled (style_key, origin_week) rows), NOT the 20 origins used for the walk-
forward backtest (those are spaced 4 weeks apart purely for backtest cleanliness/cost, per
`nss.models.backtest`'s ORIGIN SCHEDULE docstring). `build_features` computes its causal columns
densely across the whole panel exactly once regardless of how many origins are requested (see that
module's IMPLEMENTATION STRATEGY docstring), so widening the origin set from 20 to 93 costs ~1.5s
of extra compute (measured) for ~4.6x more pooled training rows -- a easy trade for the FINAL
production model, whose only job is to generalize to one never-scored real forecast, not to
reproduce a comparable walk-forward metric. The wider set is NOT re-scored against any test
origin, so it carries no leakage risk for the already-reported walk-forward numbers.

SEASONAL BONUS GUARD CONVENTION (JUDGMENT CALL): the seasonal top-3 table (`build_seasonal_
ranking`) restricts guard 1 (commercial scale) to the season's own historical weeks (mean
`n_active_articles` over that style's historical rows falling in that season, across all years),
per the brief. Guards 2 (markdown) and 3 (viability) are NOT re-computed per season -- both reuse
the SAME current-as-of-forecast-origin values as the main (non-seasonal) ranking. Rationale: guard
2 is inherently a "right now" check (see DISCREPANCY B) with no seasonal analogue to restrict it
to, and guard 3 ("is this style still alive at all") is about current-day viability of the style as
a going concern, not about whether it happened to sell in a particular season historically -- a
style that is viable today but seasonally strong in an off-season it's not currently exhibiting
should still not be recommended if it is not currently a going concern.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import shap

from nss.features.model_features import build_features
from nss.features.style_panel import STYLE_KEY_COLS
from nss.models.backtest import generate_origin_schedule
from nss.models.lightgbm_model import (
    HYPERPARAM_GRID,
    LGBMConfig,
    _to_lgb_matrix,
    build_model_frame,
    feature_columns,
    predict_lightgbm,
    train_lightgbm,
)

# The panel's actual last observed Monday `week_start` -- verified at runtime against the real
# panel by `verify_forecast_origin` (never assumed silently). AW2020 = the 13 weeks after this.
FORECAST_ORIGIN = date(2020, 9, 21)

# The winning hyperparameters from the bounded search (see task/PLAN.md), equal to
# `HYPERPARAM_GRID`'s 5th entry in `nss.models.lightgbm_model` -- kept as an explicit literal here
# (not just an index into that list) so this module's intent is readable on its own, with a
# runtime assertion in `main()` tying it back to the grid to catch any future drift.
FINAL_MODEL_CONFIG: LGBMConfig = {
    "num_leaves": 63,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "min_child_samples": 50,
}

# See module docstring GUARD WINDOW CONVENTION / GUARD 1 AGGREGATION.
GUARD1_WINDOW_WEEKS = 13
GUARD1_MIN_MEAN_N_ACTIVE_ARTICLES = 10.0
GUARD2_MIN_PRICE_INDEX = 0.85
GUARD3_WINDOW_WEEKS = 52
GUARD3_MIN_WEEKS_ACTIVE = 26

TOP_N = 10
LOCAL_SHAP_TOP_N = 5

# See module docstring DISCREPANCY C.
SEASON_MONTHS: dict[str, tuple[int, ...]] = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
}
SEASON_TOP_N = 3

DEFAULT_PANEL_PATH = Path("data/processed/style_week_panel.parquet")
DEFAULT_TOP_STYLES_OUT_PATH = Path("reports/tables/top_styles.csv")
DEFAULT_MARKDOWN_EXCLUDED_OUT_PATH = Path("reports/tables/top_styles_markdown_excluded.csv")
DEFAULT_SEASONAL_OUT_PATH = Path("reports/tables/top_styles_by_season.csv")


def final_training_origin_weeks(panel: pl.DataFrame) -> list[date]:
    """The WIDE final-model training origin set. See module docstring TRAINING DATA WIDTH."""
    origins = generate_origin_schedule(panel, step_weeks=1, burn_in_weeks=0)
    return [o.origin_week for o in origins]


def train_final_model(
    panel: pl.DataFrame, config: LGBMConfig = FINAL_MODEL_CONFIG
) -> tuple[lgb.LGBMRegressor, pl.DataFrame, list[str]]:
    """Train the final production model on every wide-set training origin's pooled rows.

    Returns:
        `(model, model_frame, columns)` -- `model_frame` is `build_model_frame`'s full output
        (needed by no caller currently, returned for inspection/testing).
    """
    origin_weeks = final_training_origin_weeks(panel)
    model_frame = build_model_frame(panel, origin_weeks)
    columns = feature_columns(model_frame)
    model = train_lightgbm(model_frame, config, columns)
    return model, model_frame, columns


def verify_forecast_origin(panel: pl.DataFrame, forecast_origin: date = FORECAST_ORIGIN) -> None:
    """Raise if `forecast_origin` is not actually the panel's last observed week.

    Never assume the forecast origin is correct -- confirm against the real panel every time (per
    task instructions), the same defensive-gate pattern as
    `nss.features.style_panel.main`'s retention gate.
    """
    max_week: date = panel["week_start"].max()
    if max_week != forecast_origin:
        raise ValueError(
            f"FORECAST_ORIGIN ({forecast_origin}) does not match the panel's actual last "
            f"observed week ({max_week}) -- update FORECAST_ORIGIN or re-check the panel."
        )


def build_forecast_frame(
    panel: pl.DataFrame, forecast_origin: date = FORECAST_ORIGIN
) -> pl.DataFrame:
    """Every eligible style's causal feature row as of `forecast_origin`. See `build_features`."""
    return build_features(panel, [forecast_origin])


def predict_intensity(
    model: lgb.LGBMRegressor, forecast_frame: pl.DataFrame, columns: list[str]
) -> np.ndarray:
    """The model's predicted mean `units_per_active_article` over the forecast window.

    `expm1` of the model's raw log1p-scale prediction. See module docstring DISCREPANCY A for why
    this (not `intensity_shrunk`) is what "predicted intensity" means operationally here.
    """
    log_preds = predict_lightgbm(model, forecast_frame, columns)
    return np.expm1(log_preds)


def _add_guard_windows(panel: pl.DataFrame) -> pl.DataFrame:
    """Attach the two trailing, inclusive-of-current-row guard window columns. See module
    docstring GUARD WINDOW CONVENTION."""
    return panel.with_columns(
        guard1_n_active_articles_trailing_mean=pl.col("n_active_articles")
        .rolling_mean(window_size=GUARD1_WINDOW_WEEKS, min_samples=1)
        .over("style_key", order_by="week_start"),
        guard3_n_weeks_active_trailing=(pl.col("units") > 0)
        .cast(pl.Int32)
        .rolling_sum(window_size=GUARD3_WINDOW_WEEKS, min_samples=1)
        .over("style_key", order_by="week_start"),
    )


def build_guard_frame(panel: pl.DataFrame, forecast_origin: date = FORECAST_ORIGIN) -> pl.DataFrame:
    """Guard 1 + guard 3 underlying values, one row per style_key, as of `forecast_origin`."""
    windowed = _add_guard_windows(panel)
    return windowed.filter(pl.col("week_start") == forecast_origin).select(
        "style_key",
        "guard1_n_active_articles_trailing_mean",
        "guard3_n_weeks_active_trailing",
    )


def build_ranking_frame(
    panel: pl.DataFrame,
    model: lgb.LGBMRegressor,
    columns: list[str],
    forecast_origin: date = FORECAST_ORIGIN,
) -> pl.DataFrame:
    """The full eligible-style ranking: predicted intensity + all 3 guard values, sorted
    descending by predicted intensity, with an unguarded `rank_unguarded` column attached.

    Returns:
        One row per style eligible at `forecast_origin` (has a `build_features` row there),
        columns: `style_key`, the 5 `STYLE_KEY_COLS`, `predicted_intensity`,
        `guard1_n_active_articles_trailing_mean`, `guard1_pass`, `price_index_level` (guard 2's
        underlying value), `guard2_pass`, `guard3_n_weeks_active_trailing`, `guard3_pass`,
        `rank_unguarded` (1 = highest predicted intensity, no guards applied).
    """
    forecast_frame = build_forecast_frame(panel, forecast_origin)
    predicted = predict_intensity(model, forecast_frame, columns)
    guard_frame = build_guard_frame(panel, forecast_origin)

    ranking = (
        forecast_frame.select("style_key", *STYLE_KEY_COLS, "price_index_level")
        .with_columns(pl.Series("predicted_intensity", predicted))
        .join(guard_frame, on="style_key", how="left")
    )
    ranking = ranking.with_columns(
        (
            pl.col("guard1_n_active_articles_trailing_mean") >= GUARD1_MIN_MEAN_N_ACTIVE_ARTICLES
        ).alias("guard1_pass"),
        # null price_index (no 52w trailing price history yet) -> can't verify -> fail.
        pl.when(pl.col("price_index_level").is_not_null())
        .then(pl.col("price_index_level") >= GUARD2_MIN_PRICE_INDEX)
        .otherwise(False)
        .alias("guard2_pass"),
        (pl.col("guard3_n_weeks_active_trailing") >= GUARD3_MIN_WEEKS_ACTIVE).alias("guard3_pass"),
    )
    ranking = ranking.sort("predicted_intensity", descending=True).with_row_index(
        "rank_unguarded", offset=1
    )
    return ranking


def select_top_styles(ranking: pl.DataFrame, top_n: int = TOP_N) -> pl.DataFrame:
    """Ranks 1..`top_n` among styles PASSING all 3 guards, by predicted intensity descending."""
    passing = ranking.filter(
        pl.col("guard1_pass") & pl.col("guard2_pass") & pl.col("guard3_pass")
    ).sort("predicted_intensity", descending=True)
    return passing.head(top_n).with_row_index("rank", offset=1).drop("rank_unguarded")


def select_markdown_excluded(ranking: pl.DataFrame, top_n: int = TOP_N) -> pl.DataFrame:
    """Styles in the RAW (unguarded) top `top_n` by predicted intensity that fail guard 2
    SPECIFICALLY (pass guards 1 and 3, fail guard 2 -- see module docstring / task step 3.6)."""
    raw_top = ranking.filter(pl.col("rank_unguarded") <= top_n)
    excluded = raw_top.filter(
        pl.col("guard1_pass") & pl.col("guard3_pass") & ~pl.col("guard2_pass")
    )
    return excluded.rename({"rank_unguarded": "rank_if_included"})


def compute_local_shap_drivers(
    model: lgb.LGBMRegressor,
    style_keys_in_order: list[str],
    forecast_frame: pl.DataFrame,
    columns: list[str],
    top_n: int = LOCAL_SHAP_TOP_N,
) -> pl.DataFrame:
    """Per-style LOCAL SHAP top-`top_n` drivers (feature name + SHAP value), for exactly the
    styles in `style_keys_in_order`, preserving that order.

    Distinct from `nss.models.lightgbm_model.compute_global_shap_importance` (global, mean-
    absolute, whole-dataset) -- this is one row's own SHAP explanation per style.

    Returns:
        One row per `style_keys_in_order` entry (in that order), columns `style_key` plus
        `shap_driver_{i}_feature` / `shap_driver_{i}_value` for `i` in `1..top_n`.
    """
    order_df = pl.DataFrame({"style_key": style_keys_in_order}).with_row_index("_order")
    rows = forecast_frame.join(order_df, on="style_key", how="inner").sort("_order").drop("_order")
    X = _to_lgb_matrix(rows, columns)
    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(explainer.shap_values(X))

    out_rows: list[dict[str, object]] = []
    for i, style_key in enumerate(rows["style_key"].to_list()):
        row_shap = shap_values[i]
        top_idx = np.argsort(-np.abs(row_shap))[:top_n]
        record: dict[str, object] = {"style_key": style_key}
        for rank, idx in enumerate(top_idx, start=1):
            record[f"shap_driver_{rank}_feature"] = columns[idx]
            record[f"shap_driver_{rank}_value"] = float(row_shap[idx])
        out_rows.append(record)
    return pl.DataFrame(out_rows)


def build_seasonal_ranking(
    panel: pl.DataFrame,
    ranking: pl.DataFrame,
    top_n: int = SEASON_TOP_N,
) -> pl.DataFrame:
    """The BONUS seasonal top-3 table: historical season-restricted mean intensity, guarded.

    See module docstring SEASONAL BONUS GUARD CONVENTION for guard 2/3 reuse and DISCREPANCY C for
    the season-boundary resolution.

    Args:
        panel: The dense style-week panel.
        ranking: `build_ranking_frame`'s output -- supplies the forecast-eligible population and
            the current (non-seasonal) guard 2 / guard 3 values reused here.
        top_n: Styles per season. Defaults to `SEASON_TOP_N` (3).

    Returns:
        One row per `(season, rank)`, columns: `season`, `rank`, `style_key`, the 5
        `STYLE_KEY_COLS`, `season_mean_intensity`, `season_guard1_n_active_articles_mean`,
        `guard1_pass`, `price_index_level`, `guard2_pass`, `guard3_n_weeks_active_trailing`,
        `guard3_pass`.
    """
    guard_lookup = ranking.select(
        "style_key",
        *STYLE_KEY_COLS,
        "price_index_level",
        "guard2_pass",
        "guard3_n_weeks_active_trailing",
        "guard3_pass",
    )
    eligible_panel = panel.join(ranking.select("style_key"), on="style_key", how="inner")

    season_tables: list[pl.DataFrame] = []
    for season, months in SEASON_MONTHS.items():
        season_stats = (
            eligible_panel.filter(pl.col("week_start").dt.month().is_in(months))
            .group_by("style_key")
            .agg(
                season_mean_intensity=pl.col("units_per_active_article").mean(),
                season_guard1_n_active_articles_mean=pl.col("n_active_articles").mean(),
            )
        )
        season_stats = season_stats.join(guard_lookup, on="style_key", how="left")
        season_stats = season_stats.with_columns(
            (
                pl.col("season_guard1_n_active_articles_mean") >= GUARD1_MIN_MEAN_N_ACTIVE_ARTICLES
            ).alias("guard1_pass")
        )
        passing = season_stats.filter(
            pl.col("guard1_pass") & pl.col("guard2_pass") & pl.col("guard3_pass")
        )
        top = (
            passing.sort("season_mean_intensity", descending=True)
            .head(top_n)
            .with_row_index("rank", offset=1)
            .with_columns(pl.lit(season).alias("season"))
        )
        season_tables.append(top)

    result = pl.concat(season_tables)
    return result.select(
        "season",
        "rank",
        "style_key",
        *STYLE_KEY_COLS,
        "season_mean_intensity",
        "season_guard1_n_active_articles_mean",
        "guard1_pass",
        "price_index_level",
        "guard2_pass",
        "guard3_n_weeks_active_trailing",
        "guard3_pass",
    )


def main() -> None:
    """CLI entry point: train the final model, forecast AW2020, apply the selection rule, write
    `top_styles.csv`, `top_styles_markdown_excluded.csv`, and the seasonal bonus table."""
    assert (
        FINAL_MODEL_CONFIG in HYPERPARAM_GRID
    ), "FINAL_MODEL_CONFIG has drifted from HYPERPARAM_GRID -- re-check the winning config."

    panel = pl.read_parquet(DEFAULT_PANEL_PATH)
    verify_forecast_origin(panel)

    origin_weeks = final_training_origin_weeks(panel)
    print(f"Final training origins: {len(origin_weeks)} ({origin_weeks[0]} .. {origin_weeks[-1]})")
    model, model_frame, columns = train_final_model(panel)
    print(f"Trained on {model_frame.height} pooled (style_key, origin_week) rows")

    ranking = build_ranking_frame(panel, model, columns)
    print(f"Forecast-eligible styles at {FORECAST_ORIGIN}: {ranking.height}")

    top_styles = select_top_styles(ranking)
    markdown_excluded = select_markdown_excluded(ranking)

    shap_drivers = compute_local_shap_drivers(
        model, top_styles["style_key"].to_list(), build_forecast_frame(panel), columns
    )
    top_styles = top_styles.join(shap_drivers, on="style_key", how="left")

    top_styles_out = top_styles.select(
        "rank",
        "style_key",
        *STYLE_KEY_COLS,
        "predicted_intensity",
        "guard1_pass",
        "guard1_n_active_articles_trailing_mean",
        "guard2_pass",
        pl.col("price_index_level").alias("guard2_price_index"),
        "guard3_pass",
        "guard3_n_weeks_active_trailing",
        *[
            c
            for i in range(1, LOCAL_SHAP_TOP_N + 1)
            for c in (f"shap_driver_{i}_feature", f"shap_driver_{i}_value")
        ],
    )
    DEFAULT_TOP_STYLES_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    top_styles_out.write_csv(DEFAULT_TOP_STYLES_OUT_PATH)
    print(f"Wrote {DEFAULT_TOP_STYLES_OUT_PATH} ({top_styles_out.height} rows)")

    markdown_excluded_out = markdown_excluded.select(
        "rank_if_included",
        "style_key",
        *STYLE_KEY_COLS,
        "predicted_intensity",
        pl.col("price_index_level").alias("guard2_price_index"),
        "guard1_pass",
        "guard1_n_active_articles_trailing_mean",
        "guard3_pass",
        "guard3_n_weeks_active_trailing",
    )
    markdown_excluded_out.write_csv(DEFAULT_MARKDOWN_EXCLUDED_OUT_PATH)
    print(
        f"Wrote {DEFAULT_MARKDOWN_EXCLUDED_OUT_PATH} ({markdown_excluded_out.height} rows, "
        "markdown-driven exclusions from the raw top 10)"
    )

    seasonal = build_seasonal_ranking(panel, ranking)
    seasonal.write_csv(DEFAULT_SEASONAL_OUT_PATH)
    print(f"Wrote {DEFAULT_SEASONAL_OUT_PATH} ({seasonal.height} rows)")


if __name__ == "__main__":
    main()
