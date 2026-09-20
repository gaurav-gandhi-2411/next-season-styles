"""Label-leakage check: re-run the shipped walk-forward LightGBM backtest WITH a 13-week embargo.

THE SUSPECTED FLAW: `nss.models.lightgbm_model._expanding_train_origin_weeks` trains the model for
walk-forward test origin `t` on EVERY earlier origin (`origins[:test_index]`, 4-week step). A
training origin `o`'s label is the mean over `o+1w .. o+HORIZON_WEEKS w` (see
`nss.features.targets.compute_forward_target`), so for `o` in `{t-4w, t-8w, t-12w}` that window runs
past `t` -- the training LABELS contain realised outcomes of weeks inside the test origin's own
forecast window (`t+1w .. t+13w`). At real forecast time (as of `t`) those labels do not exist yet.

THIS MODULE: re-runs the same walk-forward (same features, same LightGBM config, same scoring)
except the training set for test origin `t` is restricted to origins with
`o + HORIZON_WEEKS weeks <= t`, i.e. their target window ends on or before `t`. With the 4-week
schedule that means `o <= t - 16w` (`t-13w` is not on the grid). An origin with no eligible training
origin (or no training rows) is reported as SKIPPED, never imputed. Comparison is paired per origin
over the origins evaluable in BOTH arms, using the repo's own `block_bootstrap_ci` on the per-origin
(shipped - embargoed) differences -- the same utility `backtest_v2.paired_diff_table` uses.

SCOPE NOTE: the hyperparameter grid search (`select_hyperparameters`, validated on pool origin 7
after training on pool origins 0..6) has the same overlap (origin 6's window overlaps origin 7's).
This check re-uses the shipped selected config unchanged for BOTH arms so the ONLY thing that varies
is the training-origin set; it does not re-run selection under an embargo.

Run: `python -m nss.models.backtest_embargo_check` (writes only
`reports/tables/backtest_embargo_check.csv`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from nss.features.targets import HORIZON_WEEKS
from nss.models.backtest import (
    WMAPE_WEIGHT_COL,
    Origin,
    block_bootstrap_ci,
    generate_origin_schedule,
)
from nss.models.lightgbm_model import (
    INITIAL_POOL_SIZE,
    METHOD_NAME,
    LGBMConfig,
    build_model_frame,
    feature_columns,
    predict_lightgbm,
    run_lightgbm_walk_forward,
    train_lightgbm,
)
from nss.models.metrics import METRIC_KEYS, score_predictions

RANDOM_SEED = 42  # every stochastic component (LightGBM, bootstrap) is pinned to this upstream.
DEFAULT_PANEL_PATH = Path("data/processed/style_week_panel.parquet")
DEFAULT_OUT_PATH = Path("reports/tables/backtest_embargo_check.csv")
# The committed headline (reports/tables/backtest_summary_v2.csv, lightgbm/pooled) -- used only to
# confirm this script reproduces the shipped number, never as an input to any computation.
COMMITTED_HEADLINE_METRIC = "hit_at_3_in_top20"
COMMITTED_HEADLINE_VALUE = 0.7222222222222222


def embargoed_train_origin_weeks(
    origins: Sequence[Origin], test_index: int, horizon_weeks: int = HORIZON_WEEKS
) -> set[date]:
    """Training origin weeks for test origin `origins[test_index]` under a label embargo.

    An earlier origin `o` is eligible only if its forward-target window has fully closed by the
    test origin: `o.origin_week + horizon_weeks weeks <= test origin week`. Compare against the
    shipped `nss.models.lightgbm_model._expanding_train_origin_weeks`, which has no such filter.

    Args:
        origins: The chronological origin schedule.
        test_index: Index of the test origin in `origins`.
        horizon_weeks: Forward-target window length. Defaults to `HORIZON_WEEKS` (13).

    Returns:
        The set of eligible training origin weeks (possibly empty).
    """
    test_week = origins[test_index].origin_week
    cutoff = test_week - timedelta(weeks=horizon_weeks)
    return {o.origin_week for o in origins[:test_index] if o.origin_week <= cutoff}


def run_embargoed_walk_forward(
    panel: pl.DataFrame,
    origins: list[Origin],
    config: LGBMConfig,
    pool_size: int = INITIAL_POOL_SIZE,
) -> tuple[pl.DataFrame, list[date]]:
    """Walk-forward LightGBM identical to `run_lightgbm_walk_forward` except for the embargo.

    Reuses `build_model_frame`, `train_lightgbm`, `predict_lightgbm` and `score_predictions`
    unchanged; only the training-origin set differs (`embargoed_train_origin_weeks`).

    Args:
        panel: The dense style-week panel.
        origins: The full origin schedule.
        config: The LightGBM config (the shipped one, unchanged).
        pool_size: Initial pool size; test origins are `origins[pool_size:]`.

    Returns:
        `(per_origin, skipped_weeks)` -- `per_origin` has the same schema as
        `run_lightgbm_walk_forward`'s output; `skipped_weeks` lists test origins with no usable
        embargoed training data (not scored, not imputed).
    """
    model_frame = build_model_frame(panel, [o.origin_week for o in origins])
    columns = feature_columns(model_frame)

    rows: list[dict[str, object]] = []
    skipped: list[date] = []
    for test_index in range(pool_size, len(origins)):
        test_origin = origins[test_index]
        train_weeks = embargoed_train_origin_weeks(origins, test_index)
        train_frame = model_frame.filter(pl.col("origin_week").is_in(train_weeks))
        test_frame = model_frame.filter(pl.col("origin_week") == test_origin.origin_week)
        if not train_weeks or train_frame.height == 0 or test_frame.height == 0:
            skipped.append(test_origin.origin_week)
            continue

        model = train_lightgbm(train_frame, config, columns)
        preds = predict_lightgbm(model, test_frame, columns)
        metrics = score_predictions(
            test_frame["y_true"].to_numpy(),
            preds,
            test_frame[WMAPE_WEIGHT_COL].to_numpy(),
        )
        n_eval = metrics.pop("n_eval")
        rows.append(
            {
                "origin_week": test_origin.origin_week,
                "method": METHOD_NAME,
                "has_52w_lag": test_origin.has_52w_lag,
                "is_covid": test_origin.is_covid,
                "n_eval_set": test_frame.height,
                "n_eval": int(n_eval),
                "n_train_origins": len(train_weeks),
                "n_train_rows": train_frame.height,
                **metrics,
            }
        )
    return pl.DataFrame(rows), skipped


def compare_arms(shipped: pl.DataFrame, embargoed: pl.DataFrame) -> pl.DataFrame:
    """Paired shipped-vs-embargoed comparison on origins present in BOTH per-origin tables.

    Args:
        shipped: Per-origin table from the no-embargo run.
        embargoed: Per-origin table from the embargoed run.

    Returns:
        Long-format table: `split` (pooled/covid/non_covid), `metric`, `shipped`, `embargoed`
        (per-split means over the paired origins), `diff` (mean of shipped - embargoed),
        `ci_lo`/`ci_hi` (block-bootstrap 95% CI of the per-origin difference series), `n_origins`.
    """
    joined = shipped.join(embargoed, on="origin_week", how="inner", suffix="_emb").sort(
        "origin_week"
    )
    splits = (
        ("pooled", joined),
        ("covid", joined.filter(pl.col("is_covid"))),
        ("non_covid", joined.filter(~pl.col("is_covid"))),
    )
    rows: list[dict[str, object]] = []
    for split_name, sdf in splits:
        for metric in METRIC_KEYS:
            a = sdf[metric].to_numpy().astype(float)
            b = sdf[f"{metric}_emb"].to_numpy().astype(float)
            diff = (a - b).tolist()
            mean_diff, lo, hi = block_bootstrap_ci(diff)
            rows.append(
                {
                    "split": split_name,
                    "metric": metric,
                    "shipped": float(np.nanmean(a)) if a.size else math.nan,
                    "embargoed": float(np.nanmean(b)) if b.size else math.nan,
                    "diff": mean_diff,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "n_origins": sdf.height,
                }
            )
    return pl.DataFrame(rows)


def main() -> None:
    """Run shipped + embargoed walk-forward, print per-origin detail, write the comparison CSV."""
    panel = pl.read_parquet(DEFAULT_PANEL_PATH)
    origins = generate_origin_schedule(panel)

    shipped, config, _grid = run_lightgbm_walk_forward(panel, origins)
    print(f"Shipped config (grid-selected on pool, reused for both arms): {config}")
    embargoed, skipped = run_embargoed_walk_forward(panel, origins, config)
    print(f"Embargoed: {embargoed.height} origins scored, skipped={skipped}")

    shipped_mean = float(shipped[COMMITTED_HEADLINE_METRIC].mean())
    ok = math.isclose(shipped_mean, COMMITTED_HEADLINE_VALUE, abs_tol=1e-9)
    print(
        f"Reproduced shipped {COMMITTED_HEADLINE_METRIC} (12 origins) = {shipped_mean:.10f}; "
        f"committed = {COMMITTED_HEADLINE_VALUE:.10f}; match={ok}"
    )

    with pl.Config(
        tbl_rows=40,
        tbl_cols=20,
        tbl_width_chars=220,
        tbl_formatting="ASCII_FULL",
        float_precision=4,
    ):
        print("Per-origin (embargoed):")
        print(
            embargoed.select(
                "origin_week", "is_covid", "n_train_origins", "n_train_rows", *METRIC_KEYS
            )
        )
        print("Per-origin (shipped):")
        print(shipped.select("origin_week", "is_covid", *METRIC_KEYS))

    table = compare_arms(shipped, embargoed)
    with pl.Config(
        tbl_rows=40,
        tbl_cols=20,
        tbl_width_chars=220,
        tbl_formatting="ASCII_FULL",
        float_precision=4,
    ):
        print(table)

    DEFAULT_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(DEFAULT_OUT_PATH)
    print(f"Wrote {DEFAULT_OUT_PATH} ({table.height} rows)")


if __name__ == "__main__":
    main()
