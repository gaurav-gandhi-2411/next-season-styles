"""Data for the DEMO's forecast explorer (task P5): the top styles as one JSON file.

Everything is derived from the frozen forecast (`final_forecast`: `FINAL_MODEL_CONFIG`, wide
training
origins, seed 42, deterministic; the same code path `forecast_all_styles` uses), so the explorer
shows the shipped numbers, not a re-estimate:

- the top `TOP_N` guard-passing styles by predicted intensity, plus the emerging leaderboard's top
  10 and the final concepts' styles (flagged) so a buyer can find them;
- per style: attributes, predicted intensity, rank among guard-passing styles, trailing-13-week
  intensity and growth ratio, the three guards (value and pass), peak season by historical mean
  intensity (season months as in `final_forecast.SEASON_MONTHS`), the last 78 weeks of weekly
  intensity (the sparkline) and the LOCAL SHAP top-5 drivers.

Output: `reports/tables/explorer_styles.json` (a list of objects; nothing else reads it but the
demo builder).

Usage:
    uv run python -m nss.viz.explorer_data
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from nss.features.style_panel import STYLE_KEY_COLS
from nss.generate import final_registry
from nss.models import diversity_forecast as dm
from nss.models import final_forecast as ff

OUT = Path("reports/tables/explorer_styles.json")
TOP_N = 200
TRAJECTORY_WEEKS = 78
T2_TABLE = Path("reports/tables/top_styles_t2_emerging.csv")


def peak_seasons(panel: pl.DataFrame) -> dict[str, tuple[str, float]]:
    """Per style: (season with the highest mean intensity, that mean over the overall mean)."""
    month_to_season = {m: s for s, months in ff.SEASON_MONTHS.items() for m in months}
    tagged = panel.with_columns(
        pl.col("week_start")
        .dt.month()
        .map_elements(lambda m: month_to_season[m], return_dtype=pl.String)
        .alias("season")
    )
    by = tagged.group_by("style_key", "season").agg(
        pl.col("units_per_active_article").mean().alias("m")
    )
    overall = tagged.group_by("style_key").agg(pl.col("units_per_active_article").mean().alias("o"))
    best = by.sort("m", descending=True).group_by("style_key", maintain_order=True).first()
    joined = best.join(overall, on="style_key")
    return {
        r["style_key"]: (r["season"], float(r["m"] / r["o"]) if r["o"] else 1.0)
        for r in joined.iter_rows(named=True)
    }


def main() -> None:
    """Train the frozen model, assemble the explorer rows and write the JSON."""
    panel = pl.read_parquet(ff.DEFAULT_PANEL_PATH)
    ff.verify_forecast_origin(panel)
    model, _frame, columns = ff.train_final_model(panel)
    ranking = ff.build_ranking_frame(panel, model, columns)
    trailing = dm.build_trailing_intensity_frame(panel, ff.FORECAST_ORIGIN)
    ranking = (
        ranking.join(trailing, on="style_key", how="left")
        .with_columns(
            (pl.col("predicted_intensity") / pl.col("trailing_13w_mean_intensity")).alias("growth")
        )
        .with_columns(
            (pl.col("guard1_pass") & pl.col("guard2_pass") & pl.col("guard3_pass")).alias("guards")
        )
    )
    passing = ranking.filter(pl.col("guards")).sort("predicted_intensity", descending=True)
    passing = passing.with_row_index("rank", offset=1)
    keep = set(passing.head(TOP_N)["style_key"].to_list())
    keep |= set(pl.read_csv(T2_TABLE).head(10)["style_key"].to_list())
    keep |= set(final_registry.STYLE_ORDER)
    chosen = passing.filter(pl.col("style_key").is_in(list(keep))).sort("rank")
    order = chosen["style_key"].to_list()
    shap = ff.compute_local_shap_drivers(model, order, ff.build_forecast_frame(panel), columns)
    shap_by = {r["style_key"]: r for r in shap.iter_rows(named=True)}
    seasons = peak_seasons(panel)
    recent = (
        panel.filter(pl.col("style_key").is_in(order))
        .sort("week_start")
        .group_by("style_key", maintain_order=True)
        .tail(TRAJECTORY_WEEKS)
    )
    traj = {
        k: [round(float(v), 2) for v in g["units_per_active_article"].to_list()]
        for (k,), g in recent.group_by("style_key", maintain_order=True)
    }
    rows = []
    for r in chosen.iter_rows(named=True):
        key = r["style_key"]
        sh = shap_by[key]
        rows.append(
            {
                "key": key,
                **{c: r[c] for c in STYLE_KEY_COLS},
                "rank": r["rank"],
                "pred": round(float(r["predicted_intensity"]), 3),
                "trail": round(float(r["trailing_13w_mean_intensity"] or 0.0), 3),
                "growth": round(float(r["growth"]), 3) if r["growth"] is not None else None,
                "guard": {
                    "n_active": round(float(r["guard1_n_active_articles_trailing_mean"]), 1),
                    "price_index": round(float(r["price_index_level"]), 3),
                    "weeks_active": int(r["guard3_n_weeks_active_trailing"]),
                },
                "season": seasons[key][0],
                "season_index": round(seasons[key][1], 2),
                "shap": [
                    [sh[f"shap_driver_{i}_feature"], round(sh[f"shap_driver_{i}_value"], 4)]
                    for i in range(1, 6)
                ],
                "traj": traj.get(key, []),
                "concept": key in final_registry.STYLE_ORDER,
            }
        )
    OUT.write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT}: {len(rows)} styles, {OUT.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
