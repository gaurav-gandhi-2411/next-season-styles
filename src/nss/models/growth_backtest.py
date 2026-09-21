"""Backtest of the GROWTH ranking (the emerging list's `growth_ratio`), embargoed, with baselines.

Rules are pre-registered in `reports/v3/PREREGISTRATION.md` (2a), committed before this ran. See
that section for the populations, the decision categories and the restriction that only styles with
an observed 13-week outcome can be scored.

Method. At each test origin `t` the model trained under the embargoed protocol (training origins at
least 16 weeks before `t`, the locked `FINAL_MODEL_CONFIG`) predicts every style; the deployed
selection quantity is reproduced:

    growth_ratio(method) = expm1(method's log1p prediction) / trailing_13w_mean_intensity
    realised growth      = expm1(y_true) / trailing_13w_mean_intensity

then each method's ranking of the SAME population is scored against realised growth with
`nss.models.metrics.score_predictions`, the random floor is the realised growth permuted across the
population, and comparisons are paired per origin with the project's block bootstrap.

`predictions_for_origins` accepts any set of test origins and a 4-week grid: an origin between grid
points uses the model of the greatest grid origin at or before it. For 4-week-spaced origins that is
the shipped protocol exactly; it is also what makes the 1-week-step run in 2b "the same model
measured with more origins" (a weekly origin's training set under the 16-week embargo is
identical to that of its grid block).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np
import polars as pl

from nss.models import final_forecast
from nss.models.backtest import (
    BASELINE_METHODS,
    BOOTSTRAP_BLOCK_SIZE,
    Origin,
    block_bootstrap_ci,
    build_predictions_frame,
    summarize_backtest,
)
from nss.models.backtest_embargo_check import embargoed_train_origin_weeks
from nss.models.backtest_v2 import paired_diff_table
from nss.models.diversity_forecast import build_trailing_intensity_frame
from nss.models.lightgbm_model import (
    METHOD_NAME,
    build_model_frame,
    feature_columns,
    predict_lightgbm,
    train_lightgbm,
)
from nss.models.metrics import METRIC_KEYS, score_predictions
from nss.models.random_floor import (
    RANDOM_FLOOR_METHOD,
    aggregate_random_floor_over_seeds,
    score_random_floor_per_origin,
)

MODEL = METHOD_NAME
METHODS: tuple[str, ...] = (MODEL, *BASELINE_METHODS)
PRIMARY_METRIC = "hit_at_3_in_top20"
TOP_K_LIFT = 3


def predictions_for_origins(
    panel: pl.DataFrame,
    grid_origins: Sequence[Origin],
    test_weeks: Sequence[date],
    initial_pool_size: int,
) -> pl.DataFrame:
    """One row per (style, test origin) with y_true, every method's log1p prediction and guards.

    Args:
        panel: The dense style-week panel.
        grid_origins: The full 4-week-grid origin schedule (`generate_origin_schedule`).
        test_weeks: Test origins to score. Each is served by the model of the greatest grid TEST
            origin (index >= `initial_pool_size`) at or before it.
        initial_pool_size: Number of leading grid origins that are training-only.
    """
    grid_weeks = [o.origin_week for o in grid_origins]
    all_weeks = sorted({*grid_weeks, *test_weeks})
    frame = build_model_frame(panel, all_weeks)
    columns = feature_columns(frame)
    baselines = build_predictions_frame(panel, list(test_weeks))

    models: dict[int, object] = {}
    parts = []
    for t in sorted(test_weeks):
        k = max(i for i, w in enumerate(grid_weeks) if w <= t)
        if k < initial_pool_size:
            raise ValueError(f"test origin {t} precedes the first walk-forward test origin")
        if k not in models:
            train_weeks = embargoed_train_origin_weeks(list(grid_origins), k)
            train = frame.filter(pl.col("origin_week").is_in(train_weeks))
            models[k] = train_lightgbm(train, final_forecast.FINAL_MODEL_CONFIG, columns)
        test = frame.filter(pl.col("origin_week") == t)
        preds = predict_lightgbm(models[k], test, columns)
        parts.append(
            test.select(
                "style_key",
                "origin_week",
                "y_true",
                "price_index_level",
                "n_active_articles_level",
            ).with_columns(pl.Series(f"y_pred_{MODEL}", preds))
        )
    out = pl.concat(parts).join(
        baselines.select("style_key", "origin_week", *[f"y_pred_{m}" for m in BASELINE_METHODS]),
        on=["style_key", "origin_week"],
        how="left",
    )
    return attach_guards_and_trailing(panel, out)


def attach_guards_and_trailing(panel: pl.DataFrame, frame: pl.DataFrame) -> pl.DataFrame:
    """Add guard 1-3 pass flags and the deployed growth denominator at each origin."""
    parts = []
    for t in frame["origin_week"].unique().sort().to_list():
        guard = final_forecast.build_guard_frame(panel, t)
        trailing = build_trailing_intensity_frame(panel, t)
        sub = frame.filter(pl.col("origin_week") == t).join(guard, on="style_key", how="left")
        sub = sub.join(trailing, on="style_key", how="left")
        parts.append(sub)
    out = pl.concat(parts)
    return out.with_columns(
        (
            pl.col("guard1_n_active_articles_trailing_mean")
            >= final_forecast.GUARD1_MIN_MEAN_N_ACTIVE_ARTICLES
        ).alias("guard1_pass"),
        pl.when(pl.col("price_index_level").is_not_null())
        .then(pl.col("price_index_level") >= final_forecast.GUARD2_MIN_PRICE_INDEX)
        .otherwise(False)
        .alias("guard2_pass"),
        (pl.col("guard3_n_weeks_active_trailing") >= final_forecast.GUARD3_MIN_WEEKS_ACTIVE).alias(
            "guard3_pass"
        ),
    )


def population(frame: pl.DataFrame, which: str) -> pl.DataFrame:
    """The scored population: "P1" guard-passing, "P2" deployed (adds the model's own floor)."""
    guarded = frame.filter(pl.col("guard1_pass") & pl.col("guard2_pass") & pl.col("guard3_pass"))
    positive = guarded.filter(pl.col("trailing_13w_mean_intensity") > 0.0)
    if which == "P1":
        return positive
    if which != "P2":
        raise ValueError(which)
    parts = []
    for t in guarded["origin_week"].unique().sort().to_list():
        g = guarded.filter(pl.col("origin_week") == t)
        floor = float(np.median(np.expm1(g[f"y_pred_{MODEL}"].to_numpy())))
        p = positive.filter(pl.col("origin_week") == t)
        parts.append(p.filter(np.expm1(pl.col(f"y_pred_{MODEL}")) >= floor))
    return pl.concat(parts)


def growth_columns(pop: pl.DataFrame) -> pl.DataFrame:
    """Realised growth and every method's predicted growth (`expm1(pred) / trailing mean`)."""
    denom = pl.col("trailing_13w_mean_intensity")
    return pop.with_columns(
        (pl.col("y_true").exp() - 1.0).truediv(denom).alias("g_true"),
        *[((pl.col(f"y_pred_{m}").exp() - 1.0) / denom).alias(f"g_pred_{m}") for m in METHODS],
    )


def score_methods(
    pop: pl.DataFrame, origins: Sequence[Origin]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(per-origin metric rows for every method and the random floor, top-3 lift rows)."""
    g = growth_columns(pop)
    rows: list[dict] = []
    lift: list[dict] = []
    for o in origins:
        sub = g.filter(pl.col("origin_week") == o.origin_week)
        if sub.height == 0:
            continue
        for m in METHODS:
            s = sub.filter(pl.col(f"g_pred_{m}").is_not_null())
            if s.height == 0:
                continue
            metrics = score_predictions(
                s["g_true"].to_numpy(),
                s[f"g_pred_{m}"].to_numpy(),
                s["n_active_articles_level"].to_numpy(),
            )
            n_eval = metrics.pop("n_eval")
            rows.append(
                {
                    "origin_week": o.origin_week,
                    "method": m,
                    "has_52w_lag": o.has_52w_lag,
                    "is_covid": o.is_covid,
                    "n_eval_set": sub.height,
                    "n_eval": int(n_eval),
                    **metrics,
                }
            )
            top = s.sort(f"g_pred_{m}", descending=True, maintain_order=True).head(TOP_K_LIFT)
            lift.append(
                {
                    "origin_week": o.origin_week,
                    "method": m,
                    "top3_mean_realised_growth": float(top["g_true"].mean()),
                    "population_mean_realised_growth": float(s["g_true"].mean()),
                    "lift": float(top["g_true"].mean() / s["g_true"].mean()),
                }
            )
    predictions = g.select(
        "style_key",
        "origin_week",
        pl.col("g_true").alias("y_true"),
        pl.col("n_active_articles_level").alias("weight"),
    )
    floor = aggregate_random_floor_over_seeds(
        score_random_floor_per_origin(
            predictions, [o for o in origins if o.origin_week in set(g["origin_week"])]
        )
    )
    per_origin = pl.concat([pl.DataFrame(rows), floor.select(pl.DataFrame(rows).columns)])
    return per_origin, pl.DataFrame(lift)


def beats(paired: pl.DataFrame, comparator: str) -> dict[str, object]:
    """Whether the model beats `comparator` under the pre-registered rule (A or B), pooled."""
    p = paired.filter((pl.col("method_b") == comparator) & (pl.col("split") == "pooled"))

    def row(metric: str) -> dict:
        return p.filter(pl.col("metric") == metric).row(0, named=True)

    top20, ndcg, rho = row(PRIMARY_METRIC), row("ndcg_at_10"), row("spearman_rho")
    rule_a = bool(top20["diff_ci_low"] > 0)
    rule_b = bool(ndcg["diff_ci_low"] > 0 and rho["diff_ci_low"] > 0 and top20["mean_diff"] >= 0)
    return {
        "comparator": comparator,
        "rule_a": rule_a,
        "rule_b": rule_b,
        "beats": rule_a or rule_b,
        "hit_diff": top20["mean_diff"],
        "hit_ci_lo": top20["diff_ci_low"],
        "hit_ci_hi": top20["diff_ci_high"],
        "ndcg_ci_lo": ndcg["diff_ci_low"],
        "spearman_ci_lo": rho["diff_ci_low"],
        "n_origins": top20["n_origins"],
    }


def categorise(paired: pl.DataFrame) -> tuple[str, pl.DataFrame]:
    """VALIDATED / PARTIAL / NOT_VALIDATED per the pre-registered categories."""
    table = pl.DataFrame([beats(paired, c) for c in (*BASELINE_METHODS, RANDOM_FLOOR_METHOD)])
    beats_floor = bool(table.filter(pl.col("comparator") == RANDOM_FLOOR_METHOD)["beats"][0])
    beats_all = bool(table.filter(pl.col("comparator") != RANDOM_FLOOR_METHOD)["beats"].all())
    if beats_floor and beats_all:
        return "VALIDATED", table
    return ("PARTIAL" if beats_floor else "NOT_VALIDATED"), table


def run(
    frame: pl.DataFrame, origins: Sequence[Origin], which: str
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Score one population: (per-origin, summary, paired vs every comparator, lift)."""
    per_origin, lift = score_methods(population(frame, which), origins)
    summary = summarize_backtest(per_origin)
    paired = paired_diff_table(
        per_origin,
        treatment_method=MODEL,
        baseline_methods=(*BASELINE_METHODS, RANDOM_FLOOR_METHOD),
        block_size=BOOTSTRAP_BLOCK_SIZE,
    )
    return per_origin, summary, paired, lift


def lift_ci(lift: pl.DataFrame) -> pl.DataFrame:
    """Per-method mean top-3 lift with a block-bootstrap CI over origins."""
    rows = []
    for m in lift["method"].unique(maintain_order=True).to_list():
        vals = lift.filter(pl.col("method") == m).sort("origin_week")["lift"].to_list()
        mean, lo, hi = block_bootstrap_ci(vals)
        rows.append(
            {"method": m, "n_origins": len(vals), "mean_lift": mean, "ci_lo": lo, "ci_hi": hi}
        )
    return pl.DataFrame(rows)


def week_range(start: date, end: date, step_weeks: int = 1) -> list[date]:
    """Weekly Mondays from `start` to `end` inclusive."""
    out, d = [], start
    while d <= end:
        out.append(d)
        d += timedelta(weeks=step_weeks)
    return out


OUT_DIR = "reports/tables"


def main() -> None:
    """2a: growth backtest on the 12 shared embargoed origins, populations P2 (primary) and P1."""
    from nss.models.backtest import generate_origin_schedule
    from nss.models.backtest_v2 import identify_lightgbm_origins
    from nss.models.lightgbm_model import INITIAL_POOL_SIZE

    panel = pl.read_parquet("data/processed/style_week_panel.parquet")
    grid = generate_origin_schedule(panel)
    origins = identify_lightgbm_origins(panel)
    test_weeks = [o.origin_week for o in origins]
    frame = predictions_for_origins(panel, grid, test_weeks, INITIAL_POOL_SIZE)
    frame.write_parquet("data/generated/v3_growth_predictions.parquet")
    for which in ("P2", "P1"):
        per_origin, summary, paired, lift = run(frame, origins, which)
        per_origin.write_csv(f"{OUT_DIR}/v3_growth_per_origin_{which}.csv")
        summary.write_csv(f"{OUT_DIR}/v3_growth_summary_{which}.csv")
        paired.write_csv(f"{OUT_DIR}/v3_growth_paired_{which}.csv")
        lift.write_csv(f"{OUT_DIR}/v3_growth_lift_{which}.csv")
        lift_ci(lift).write_csv(f"{OUT_DIR}/v3_growth_lift_ci_{which}.csv")
        category, table = categorise(paired)
        table.write_csv(f"{OUT_DIR}/v3_growth_decision_{which}.csv")
        sizes = population(frame, which).group_by("origin_week").len().sort("origin_week")
        sizes.write_csv(f"{OUT_DIR}/v3_growth_population_{which}.csv")
        with pl.Config(
            tbl_rows=60, tbl_width_chars=220, tbl_formatting="ASCII_FULL", float_precision=3
        ):
            print(f"===== population {which}: median size {sizes['len'].median()} =====")
            keep = ["hit_at_3_in_top20", "ndcg_at_10", "spearman_rho", "wmape"]
            print(
                summary.filter(pl.col("split") == "pooled").select(
                    "method", *[f"{m}_mean" for m in keep]
                )
            )
            print(table)
            print(lift_ci(lift))
        print(f"CATEGORY ({which}): {category}")


if __name__ == "__main__":
    main()


__all__ = [
    "METRIC_KEYS",
    "categorise",
    "growth_columns",
    "lift_ci",
    "population",
    "predictions_for_origins",
    "run",
]
