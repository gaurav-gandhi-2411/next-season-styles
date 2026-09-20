"""Diversity-constrained reselection (task A7): T1 "incumbent" / T2 "emerging" winners plus a
diversity-constrained seasonal bonus v2, all built on top of `nss.models.final_forecast`'s already-
trained final model, ranking frame, and guard/SHAP infrastructure (imported and reused verbatim,
never duplicated -- see that module's own docstring for the model/guard/season rationale, not
re-litigated here).

DIVERSITY COLLISION DEFINITION: two styles collide iff they share BOTH `product_type_name` AND
`perceived_colour_master_name` -- exactly the task brief's own pair, not the full 5-column
`style_key` (which would make collisions nearly impossible and defeat the constraint's purpose:
avoiding e.g. several near-identical black T-shirts dominating one list). See
`DIVERSITY_KEY_COLS` / `apply_diversity_constraint`.

DIVERSITY WALK CONVENTION: `apply_diversity_constraint` walks an already-sorted (best-first) ranking
top-down; a row whose `DIVERSITY_KEY_COLS` pair was already claimed by a higher-ranked row is
SKIPPED outright -- not included in the output at any rank, never backfilled or renumbered around,
per task instructions. If the eligible pool is exhausted before `top_n` diversity-compliant styles
are found, fewer than `top_n` rows are returned (never an error) -- `main()` logs this if it
happens.

T2 GROWTH RATIO (JUDGMENT CALL): `growth_ratio = predicted_intensity / trailing_13w_mean_intensity`,
where the trailing mean uses the SAME inclusive-of-origin-week windowing convention as
`final_forecast`'s GUARD WINDOW CONVENTION (13 weeks, `min_samples=1`, current row is the window's
last element) -- consistent with how every other "trailing N observed weeks" window in this
codebase treats the forecast origin. This is a NEW column computed here (not present anywhere in
`final_forecast`), on `units_per_active_article` (the raw historical intensity signal, matching
what `predicted_intensity` itself represents on the log1p scale -- see `final_forecast`'s
DISCREPANCY A), not on `intensity_shrunk` (a shrunk model INPUT feature, not a comparable historical
observation for a ratio denominator).

T2 ZERO-TRAILING-MEAN HANDLING (JUDGMENT CALL): styles with `trailing_13w_mean_intensity <= 0` (no
recent sales at all) are EXCLUDED from T2 entirely, rather than assigned an infinite/undefined
ratio or a floored denominator. "Growth from zero" is not a meaningful multiplicative ratio, and a
style with zero recent sales is exactly the kind of speculative, commercially-thin pick the
absolute-intensity floor below already exists to filter out -- excluding it at the ratio step is
consistent with that intent, not a separate ad hoc rule.

T2 ABSOLUTE-INTENSITY FLOOR (JUDGMENT CALL): a T2 candidate must ALSO have `predicted_intensity >=
t2_absolute_intensity_floor(ranking)`, defined as the MEDIAN `predicted_intensity` among ALL
guard-passing styles at the forecast origin (pre-diversity, pre-growth-ranking). Rationale: this
ties "emerging" to "commercially comparable to at least a typical guard-passing style" rather than
letting a tiny micro-style's large relative jump (e.g. 0.5 -> 2.0 units) dominate a list that is
supposed to be commercially meaningful. The median is a RELATIVE-TO-THE-DATA choice (moves with the
panel, needs no re-tuning if its scale shifts) rather than a fixed literal; `GUARD1_MIN_MEAN_N_
ACTIVE_ARTICLES` itself is an article-COUNT threshold, not on the same unit scale as
`predicted_intensity`, so it is not a valid floor here directly (considered and rejected).

FINAL THREE (`build_final_three`): T1 rank 1 + T2 ranks 1-2, skipping a T2 entry that duplicates the
T1 pick (moving to the next T2 entry on collision, per task step 4) -- note this is a SEPARATE
de-duplication step from the diversity constraint (a T1/T2 duplicate style can still collide on
`DIVERSITY_KEY_COLS` with itself trivially; the two rules compose but are not the same rule).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import lightgbm as lgb
import polars as pl

from nss.features.style_panel import STYLE_KEY_COLS
from nss.models import final_forecast
from nss.models.final_forecast import compute_local_shap_drivers

DIVERSITY_KEY_COLS: tuple[str, str] = ("product_type_name", "perceived_colour_master_name")

# Same inclusive-of-origin-week trailing window convention as final_forecast.GUARD1_WINDOW_WEEKS.
# See module docstring T2 GROWTH RATIO.
GROWTH_RATIO_WINDOW_WEEKS = final_forecast.GUARD1_WINDOW_WEEKS

DEFAULT_T1_OUT_PATH = Path("reports/tables/top_styles_incumbent.csv")
DEFAULT_T2_OUT_PATH = Path("reports/tables/top_styles_emerging.csv")
DEFAULT_FINAL_THREE_OUT_PATH = Path("reports/tables/top_styles_final_three.csv")
DEFAULT_SEASONAL_V2_OUT_PATH = Path("reports/tables/top_styles_by_season_v2.csv")


def apply_diversity_constraint(
    ranked: pl.DataFrame,
    top_n: int,
    key_cols: tuple[str, str] = DIVERSITY_KEY_COLS,
) -> pl.DataFrame:
    """Walk `ranked` (already sorted best-first) top-down, keeping at most one row per distinct
    `key_cols` pair, until `top_n` rows are kept or the input is exhausted. See module docstring
    DIVERSITY WALK CONVENTION.

    Args:
        ranked: Rows already sorted descending by the caller's ranking metric.
        top_n: Maximum rows to keep.
        key_cols: The 2 columns whose joint value defines a collision.

    Returns:
        Up to `top_n` rows in walk order (still best-first), with a fresh 1-based `rank` column.
        Fewer than `top_n` rows if the eligible pool runs out first.
    """
    seen: set[tuple[object, ...]] = set()
    keep_indices: list[int] = []
    for i, pair in enumerate(ranked.select(list(key_cols)).iter_rows()):
        if pair in seen:
            continue
        seen.add(pair)
        keep_indices.append(i)
        if len(keep_indices) == top_n:
            break
    kept = ranked[keep_indices] if keep_indices else ranked.clear()
    return kept.with_row_index("rank", offset=1)


def select_t1_incumbent(ranking: pl.DataFrame, top_n: int = final_forecast.TOP_N) -> pl.DataFrame:
    """T1 "incumbent winners": `final_forecast`'s own guard-passing, absolute-`predicted_intensity`
    ranking (identical basis to `top_styles.csv`), diversity-constrained."""
    passing = ranking.filter(
        pl.col("guard1_pass") & pl.col("guard2_pass") & pl.col("guard3_pass")
    ).sort("predicted_intensity", descending=True)
    return apply_diversity_constraint(passing, top_n).drop("rank_unguarded")


def _add_trailing_intensity_mean(panel: pl.DataFrame) -> pl.DataFrame:
    """Attach `trailing_13w_mean_intensity` (T2's growth-ratio denominator). See module docstring
    T2 GROWTH RATIO for the windowing convention."""
    return panel.with_columns(
        trailing_13w_mean_intensity=pl.col("units_per_active_article")
        .rolling_mean(window_size=GROWTH_RATIO_WINDOW_WEEKS, min_samples=1)
        .over("style_key", order_by="week_start")
    )


def build_trailing_intensity_frame(
    panel: pl.DataFrame, forecast_origin: date = final_forecast.FORECAST_ORIGIN
) -> pl.DataFrame:
    """One row per style_key: `trailing_13w_mean_intensity` as of `forecast_origin`."""
    windowed = _add_trailing_intensity_mean(panel)
    return windowed.filter(pl.col("week_start") == forecast_origin).select(
        "style_key", "trailing_13w_mean_intensity"
    )


def t2_absolute_intensity_floor(ranking: pl.DataFrame) -> float:
    """T2's absolute-intensity floor: the MEDIAN `predicted_intensity` among all guard-passing
    styles at the forecast origin. See module docstring T2 ABSOLUTE-INTENSITY FLOOR."""
    passing = ranking.filter(pl.col("guard1_pass") & pl.col("guard2_pass") & pl.col("guard3_pass"))
    median = passing["predicted_intensity"].median()
    if median is None:
        raise ValueError("No guard-passing styles -- cannot compute a T2 absolute-intensity floor.")
    return float(median)


def build_t2_candidate_frame(
    panel: pl.DataFrame,
    ranking: pl.DataFrame,
    forecast_origin: date = final_forecast.FORECAST_ORIGIN,
) -> pl.DataFrame:
    """T2's full guard-passing, floor-passing, growth-ratio-ranked candidate pool (pre-diversity,
    sorted descending by `growth_ratio`). See module docstring T2 ZERO-TRAILING-MEAN HANDLING and
    T2 ABSOLUTE-INTENSITY FLOOR for the two eligibility rules applied beyond the 3 base guards."""
    trailing = build_trailing_intensity_frame(panel, forecast_origin)
    floor = t2_absolute_intensity_floor(ranking)
    candidates = (
        ranking.join(trailing, on="style_key", how="left")
        .filter(
            pl.col("guard1_pass")
            & pl.col("guard2_pass")
            & pl.col("guard3_pass")
            & pl.col("trailing_13w_mean_intensity").is_not_null()
            & (pl.col("trailing_13w_mean_intensity") > 0.0)
            & (pl.col("predicted_intensity") >= floor)
        )
        .with_columns(
            (pl.col("predicted_intensity") / pl.col("trailing_13w_mean_intensity")).alias(
                "growth_ratio"
            )
        )
    )
    return candidates.sort("growth_ratio", descending=True)


def select_t2_emerging(
    panel: pl.DataFrame,
    ranking: pl.DataFrame,
    top_n: int = final_forecast.TOP_N,
    forecast_origin: date = final_forecast.FORECAST_ORIGIN,
) -> pl.DataFrame:
    """T2 "emerging winners": top `top_n` by `growth_ratio`, diversity-constrained. See
    `build_t2_candidate_frame` for eligibility."""
    candidates = build_t2_candidate_frame(panel, ranking, forecast_origin)
    return apply_diversity_constraint(candidates, top_n).drop("rank_unguarded")


_FINAL_THREE_COLS = [
    "style_key",
    *STYLE_KEY_COLS,
    "predicted_intensity",
    "guard1_pass",
    "guard1_n_active_articles_trailing_mean",
    "guard2_pass",
    "price_index_level",
    "guard3_pass",
    "guard3_n_weeks_active_trailing",
    "growth_ratio",
    "source_table",
]


def build_final_three(
    t1: pl.DataFrame,
    t2: pl.DataFrame,
    model: lgb.LGBMRegressor,
    forecast_frame: pl.DataFrame,
    columns: list[str],
) -> pl.DataFrame:
    """T1 rank 1 + T2 ranks 1-2 (skipping a T2 entry that duplicates the T1 pick, per module
    docstring FINAL THREE), with top-5 local SHAP drivers and a `source_table` column attached.

    Args:
        t1: `select_t1_incumbent`'s output (already diversity-constrained, sorted by rank).
        t2: `select_t2_emerging`'s output (already diversity-constrained, sorted by rank).
        model: The trained final model (for SHAP).
        forecast_frame: `final_forecast.build_forecast_frame`'s output (for SHAP).
        columns: The model's feature columns (for SHAP).

    Returns:
        Up to 3 rows: T1's rank-1 style, then up to 2 T2 styles (fewer if T2 itself has fewer than
        3 non-duplicate entries), each with `source_table` in `{"T1_incumbent", "T2_emerging"}`,
        `growth_ratio` (null for the T1-sourced row), and `shap_driver_{1..5}_{feature,value}`.
    """
    t1_pick = t1.head(1).with_columns(
        pl.lit("T1_incumbent").alias("source_table"),
        pl.lit(None, dtype=pl.Float64).alias("growth_ratio"),
    )
    t1_style_key = t1_pick["style_key"][0] if t1_pick.height else None

    t2_pick_keys: list[str] = []
    for style_key in t2["style_key"].to_list():
        if style_key == t1_style_key:
            continue
        t2_pick_keys.append(style_key)
        if len(t2_pick_keys) == 2:
            break
    t2_picks = (
        t2.filter(pl.col("style_key").is_in(t2_pick_keys))
        .join(pl.DataFrame({"style_key": t2_pick_keys}).with_row_index("_order"), on="style_key")
        .sort("_order")
        .drop("_order")
        .with_columns(pl.lit("T2_emerging").alias("source_table"))
    )

    combined = pl.concat(
        [t1_pick.select(_FINAL_THREE_COLS), t2_picks.select(_FINAL_THREE_COLS)],
        how="vertical",
    )
    shap_drivers = compute_local_shap_drivers(
        model, combined["style_key"].to_list(), forecast_frame, columns
    )
    return combined.join(shap_drivers, on="style_key", how="left")


def build_seasonal_ranking_diverse(
    panel: pl.DataFrame,
    ranking: pl.DataFrame,
    top_n: int = final_forecast.SEASON_TOP_N,
) -> pl.DataFrame:
    """`final_forecast.build_seasonal_ranking`'s seasonal bonus table, ADDITIONALLY diversity-
    constrained within each season -- same season-restricted guard 1 / reused guard 2+3 convention
    as that function (see its own docstring's SEASONAL BONUS GUARD CONVENTION, reused here
    verbatim, not re-litigated)."""
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
    for season, months in final_forecast.SEASON_MONTHS.items():
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
                pl.col("season_guard1_n_active_articles_mean")
                >= final_forecast.GUARD1_MIN_MEAN_N_ACTIVE_ARTICLES
            ).alias("guard1_pass")
        )
        passing = season_stats.filter(
            pl.col("guard1_pass") & pl.col("guard2_pass") & pl.col("guard3_pass")
        ).sort("season_mean_intensity", descending=True)
        top = apply_diversity_constraint(passing, top_n).with_columns(
            pl.lit(season).alias("season")
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


def _shap_columns() -> list[str]:
    return [
        c
        for i in range(1, final_forecast.LOCAL_SHAP_TOP_N + 1)
        for c in (f"shap_driver_{i}_feature", f"shap_driver_{i}_value")
    ]


def _t1_output_frame(t1_with_shap: pl.DataFrame) -> pl.DataFrame:
    return t1_with_shap.select(
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
        *_shap_columns(),
    )


def _t2_output_frame(t2_with_shap: pl.DataFrame) -> pl.DataFrame:
    return t2_with_shap.select(
        "rank",
        "style_key",
        *STYLE_KEY_COLS,
        "predicted_intensity",
        "trailing_13w_mean_intensity",
        "growth_ratio",
        "guard1_pass",
        "guard1_n_active_articles_trailing_mean",
        "guard2_pass",
        pl.col("price_index_level").alias("guard2_price_index"),
        "guard3_pass",
        "guard3_n_weeks_active_trailing",
        *_shap_columns(),
    )


def main() -> None:
    """CLI entry point: reuse `final_forecast`'s trained model + ranking frame, then write
    `top_styles_incumbent.csv`, `top_styles_emerging.csv`, `top_styles_final_three.csv`, and
    `top_styles_by_season_v2.csv`."""
    panel = pl.read_parquet(final_forecast.DEFAULT_PANEL_PATH)
    final_forecast.verify_forecast_origin(panel)

    model, _model_frame, columns = final_forecast.train_final_model(panel)
    ranking = final_forecast.build_ranking_frame(panel, model, columns)
    forecast_frame = final_forecast.build_forecast_frame(panel)
    print(f"Forecast-eligible styles at {final_forecast.FORECAST_ORIGIN}: {ranking.height}")

    target_n = final_forecast.TOP_N
    t1 = select_t1_incumbent(ranking)
    if t1.height < target_n:
        print(f"T1: diversity-constrained pool exhausted at {t1.height} styles (< {target_n})")
    t2 = select_t2_emerging(panel, ranking)
    if t2.height < target_n:
        print(f"T2: diversity-constrained pool exhausted at {t2.height} styles (< {target_n})")

    t1_shap = compute_local_shap_drivers(model, t1["style_key"].to_list(), forecast_frame, columns)
    t1_out = _t1_output_frame(t1.join(t1_shap, on="style_key", how="left"))
    DEFAULT_T1_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    t1_out.write_csv(DEFAULT_T1_OUT_PATH)
    print(f"Wrote {DEFAULT_T1_OUT_PATH} ({t1_out.height} rows)")

    t2_shap = compute_local_shap_drivers(model, t2["style_key"].to_list(), forecast_frame, columns)
    t2_out = _t2_output_frame(t2.join(t2_shap, on="style_key", how="left"))
    t2_out.write_csv(DEFAULT_T2_OUT_PATH)
    print(f"Wrote {DEFAULT_T2_OUT_PATH} ({t2_out.height} rows)")

    final_three = build_final_three(t1, t2, model, forecast_frame, columns)
    final_three.write_csv(DEFAULT_FINAL_THREE_OUT_PATH)
    print(f"Wrote {DEFAULT_FINAL_THREE_OUT_PATH} ({final_three.height} rows)")

    seasonal_v2 = build_seasonal_ranking_diverse(panel, ranking)
    seasonal_v2.write_csv(DEFAULT_SEASONAL_V2_OUT_PATH)
    print(f"Wrote {DEFAULT_SEASONAL_V2_OUT_PATH} ({seasonal_v2.height} rows)")


if __name__ == "__main__":
    main()
