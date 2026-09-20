"""Re-tune the hyperparameters under the EMBARGOED protocol, selecting on rolling origins only.

WHY: `lightgbm_model.select_hyperparameters` picked `FINAL_MODEL_CONFIG` (grid entry 5: 63 leaves,
lr 0.05, 200 trees, min_child_samples 50) by training on pool origins 0..6 and validating on pool
origin 7 -- origin 6's 13-week label window overlaps origin 7's, the same label leak the embargo
check found in the walk-forward. The choice was never repeated after the embargo.

PROTOCOL (fixed before running, no changes after seeing results):

- SAME grid (`lightgbm_model.HYPERPARAM_GRID`, 6 configs), SAME criterion (validation RMSE on the
  log1p target scale), current shipped feature set (`build_model_frame`; features are held fixed so
  this answers "was the config right", independent of the customer/price features).
- ROLLING-ORIGIN validation over the initial-pool origins ONLY: validation origin index `v` in
  `ROLLING_VALIDATION_INDICES = (4, 5, 6, 7)` (2019-04-08 .. 2019-07-01), each trained on the
  embargoed set `embargoed_train_origin_weeks(origins, v)` (origins `<= v - 16 weeks`, i.e. indices
  `0..v-4`: 1, 2, 3, 4 training origins). The 12 walk-forward TEST origins (indices 8..19, the
  holdout every reported metric comes from) are never trained on, scored on, or otherwise read here.
- Selection = lowest MEAN validation RMSE over the four validation origins (equal weight per origin;
  ties resolved by grid order). Secondary columns (Hit@3-in-top20, NDCG@10 per origin) are reported
  but do not select.

DOCUMENTED LIMITATION: validation origins 5..7 have label windows (13 weeks after the origin) that
run to 2019-08-05 .. 2019-09-30, i.e. into the calendar period of the first test origin's forecast
window (test origin 2019-07-29 forecasts 2019-07-30 .. 2019-10-28). No test-origin feature,
prediction or metric enters selection, but a fully closed selection would use only validation
origins whose label window ends by 2019-07-29, which is exactly one origin (index 4, 2019-04-08,
trained on a single origin). That strict single-origin selection is reported as a sensitivity
column (`strict_single_origin_rmse`); the four-origin criterion is the one that decides, because one
origin trained on one origin is not a usable selection signal.

Outputs (new files): `reports/tables/embargoed_retune_grid.csv` (config x validation origin),
`reports/tables/embargoed_retune_selected.csv` (one row per config: mean RMSE, rank, strict-origin
RMSE, flags for selected and current).

Usage:
    python -m nss.models.embargoed_retune
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from nss.models.backtest import Origin, generate_origin_schedule
from nss.models.backtest_embargo_check import embargoed_train_origin_weeks
from nss.models.final_forecast import FINAL_MODEL_CONFIG
from nss.models.lightgbm_model import (
    HYPERPARAM_GRID,
    LGBMConfig,
    build_model_frame,
    feature_columns,
    predict_lightgbm,
    train_lightgbm,
)
from nss.models.metrics import score_predictions
from nss.models.experiment_common import load_panel

ROLLING_VALIDATION_INDICES: tuple[int, ...] = (4, 5, 6, 7)
# the only validation origin whose label window closes before the first test origin
STRICT_VALIDATION_INDEX = 4
OUT_DIR = Path("reports/tables")
GRID_OUT = OUT_DIR / "embargoed_retune_grid.csv"
SELECTED_OUT = OUT_DIR / "embargoed_retune_selected.csv"


def config_label(config: LGBMConfig) -> str:
    """A stable short label such as `nl63_lr0.05_n200_mcs50`."""
    return (
        f"nl{int(config['num_leaves'])}_lr{config['learning_rate']}"
        f"_n{int(config['n_estimators'])}_mcs{int(config['min_child_samples'])}"
    )


def select_config(grid_results: pl.DataFrame) -> pl.DataFrame:
    """Per-config mean validation RMSE, rank (1 = best; ties by grid order) and strict-origin RMSE.

    Args:
        grid_results: One row per (config, validation origin) with columns `config`, `grid_index`,
            `val_index`, `val_rmse`.

    Returns:
        One row per config sorted by rank, columns `config`, `grid_index`, `mean_val_rmse`, `rank`,
        `strict_single_origin_rmse`.
    """
    strict = grid_results.filter(pl.col("val_index") == STRICT_VALIDATION_INDEX).select(
        "config", pl.col("val_rmse").alias("strict_single_origin_rmse")
    )
    summary = (
        grid_results.group_by("config", "grid_index")
        .agg(pl.col("val_rmse").mean().alias("mean_val_rmse"))
        .join(strict, on="config", how="left")
        .sort("mean_val_rmse", "grid_index")
        .with_row_index("rank", offset=1)
    )
    return summary.select(
        "config", "grid_index", "mean_val_rmse", "rank", "strict_single_origin_rmse"
    )


def rolling_origin_grid(
    frame: pl.DataFrame, origins: list[Origin], configs: list[LGBMConfig] = HYPERPARAM_GRID
) -> pl.DataFrame:
    """Train every config for every rolling validation origin under the embargo (long table)."""
    columns = feature_columns(frame)
    rows: list[dict[str, object]] = []
    for val_index in ROLLING_VALIDATION_INDICES:
        val_origin = origins[val_index]
        train_weeks = embargoed_train_origin_weeks(origins, val_index)
        train_frame = frame.filter(pl.col("origin_week").is_in(train_weeks))
        val_frame = frame.filter(pl.col("origin_week") == val_origin.origin_week)
        y_val = val_frame["y_true"].to_numpy().astype(np.float64)
        for grid_index, config in enumerate(configs):
            model = train_lightgbm(train_frame, config, columns)
            preds = predict_lightgbm(model, val_frame, columns)
            metrics = score_predictions(
                y_val, preds, val_frame["n_active_articles_level"].to_numpy()
            )
            rows.append(
                {
                    "config": config_label(config),
                    "grid_index": grid_index,
                    "val_index": val_index,
                    "val_origin": val_origin.origin_week,
                    "n_train_origins": len(train_weeks),
                    "n_train_rows": train_frame.height,
                    "val_rmse": float(np.sqrt(np.mean((y_val - preds) ** 2))),
                    "hit_at_3_in_top20": metrics["hit_at_3_in_top20"],
                    "ndcg_at_10": metrics["ndcg_at_10"],
                }
            )
    return pl.DataFrame(rows)


def main() -> None:
    """Run the rolling-origin grid, write the two tables, print selected vs current."""
    panel = load_panel()
    origins = generate_origin_schedule(panel)
    frame = build_model_frame(panel, [o.origin_week for o in origins])
    grid = rolling_origin_grid(frame, origins)
    selected = select_config(grid).with_columns(
        (pl.col("rank") == 1).alias("selected"),
        (pl.col("config") == config_label(FINAL_MODEL_CONFIG)).alias("current_shipped"),
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid.write_csv(GRID_OUT)
    selected.write_csv(SELECTED_OUT)
    with pl.Config(tbl_rows=40, tbl_width_chars=200, tbl_cols=-1, float_precision=5):
        print(
            grid.select("config", "val_index", "n_train_origins", "val_rmse", "hit_at_3_in_top20")
        )
        print(selected)
    best = selected.filter(pl.col("selected")).row(0, named=True)
    cur = selected.filter(pl.col("current_shipped")).row(0, named=True)
    same = best["config"] == cur["config"]
    print(f"selected: {best['config']}  current: {cur['config']}  same={same}")
    rel = 100 * (best["mean_val_rmse"] / cur["mean_val_rmse"] - 1)
    print(
        f"mean rolling RMSE selected {best['mean_val_rmse']:.5f} "
        f"vs current {cur['mean_val_rmse']:.5f} ({rel:+.2f}%)"
    )


if __name__ == "__main__":
    main()
