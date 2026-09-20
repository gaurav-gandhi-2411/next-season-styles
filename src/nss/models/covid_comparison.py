"""COVID two-model comparison: does training on COVID-period data help, hurt, or not matter?

WHY: the shipped forecaster (`nss.models.final_forecast.FINAL_MODEL_CONFIG`, LightGBM L2, seed 42)
is trained on every weekly origin through 2020-06-22, so a large share of its training rows have a
13-week target window inside the March-June 2020 COVID shock. Reviewers reasonably ask whether
those rows help the model, or whether they teach it a regime that will not recur. The existing
backtest only stratifies EVALUATION origins by `is_covid`; it never varies TRAINING data. This
module compares two models directly:

- Model A: the shipped configuration/features, trained on every eligible training row.
- Model B: identical config and features, but its training set DROPS every row whose origin week `o`
  has a target window overlapping `COVID_WINDOW_START..COVID_WINDOW_END` (2020-03-01..2020-06-30).

TARGET-WINDOW DEFINITION (verified against `nss.features.targets.compute_forward_target` and
`nss.models.backtest.generate_origin_schedule`): the target for origin `o` is built from the weeks
`o + 1w .. o + 13w` inclusive. A row is COVID-affected iff
`[o + 1w, o + 13w]` intersects `[2020-03-01, 2020-06-30]`, i.e. `o + 1w <= 2020-06-30` and
`o + 13w >= 2020-03-01` -- exactly the harness's own `is_covid` origin flag
(`horizon_overlaps_window` below reproduces it). The same predicate defines which TEST origins are
"non-COVID". Only the target window is used to drop rows: features of later, non-dropped origins
still contain COVID-period history (trailing windows, lag_52); dropping those would need a
different, feature-aware exclusion rule and is out of scope (see CAVEATS).

THREE DESIGNS (all reuse the harness's model frame / trainer / `score_predictions`; none reimplement
metrics). Why three: the literal request ("train on all data, evaluate on non-COVID origins") has
a structural problem, so the honest answer needs more than one view.

1. `purged` (PRIMARY): leave-one-origin-out with purging. For each test origin `t`, both models
   train on every wide-set weekly origin `o` (past AND future -- the shipped model also trains on
   all data) with `|o - t| >= 13` weeks, i.e. whose target window does not overlap `t`'s target
   window; B additionally drops COVID-affected rows. Test origins are scored out-of-sample, so
   this is a genuine "does adding COVID rows improve generalisation" test. It is NOT a causal
   forecast simulation (future rows train the model), and a residual leak remains for `o > t + 13w`
   (their trailing features include `t`'s realised window); the leak is identical for A and B, so
   it affects levels, not the paired difference much. Reported on the 5 non-COVID origins among
   the 12 shared walk-forward origins (`purged_shared_noncovid`), and on all 13 non-COVID origins of
   the 20-origin schedule (`purged_all_noncovid`, more origins, same fits).
2. `walk_forward` (harness-faithful, causal): the harness's expanding window (`origins[:t]` of the
   20-origin schedule, config fixed to FINAL_MODEL_CONFIG); B drops COVID-affected training rows.
   By construction A and B are IDENTICAL on every non-COVID test origin (no earlier origin's
   target window can overlap COVID when the test horizon itself precedes it -- asserted at runtime),
   so the non-COVID stratum is a vacuous 0-difference; the informative rows are the `covid` and
   `pooled` strata over the 12 shared origins. That comparison answers "did COVID rows in the
   training history help forecast later COVID/post-COVID origins".
3. `frozen_in_sample` (the literal request, for completeness): A and B each trained ONCE on the wide
   set, both scored on the 5 shared non-COVID test origins. Those origins' rows are inside both
   training sets, so this is IN-SAMPLE fit, not forecast skill -- reported and labelled as such,
   never as headline evidence.

PAIRED STATISTICS: per-origin difference `B - A` per metric, block-bootstrapped with
`nss.models.backtest.block_bootstrap_ci` (block 4, 2000 resamples, seed 42 -- the same defaults as
`backtest_v2.paired_diff_table`). With n = 5 origins the bootstrap has only two block start
positions, so its CI is nearly degenerate; treat "CI excludes zero" at n < 8 as directional at best
(`directional_only`, same MIN_ORIGINS_FOR_NON_DIRECTIONAL threshold as backtest_v2). Verdicts are
direction-aware: `diff = B - A`; for higher-is-better metrics a CI above 0 means dropping COVID
data was better, i.e. COVID data HURT; below 0 means COVID data HELPED; WMAPE (lower is better) is
the mirror image. A CI containing 0 (or identical models) is `no_difference`.
Verdicts whose CI excludes zero are suffixed `_directional_only` (n < 8 origins) or
`_in_sample_only` (design 3), so a bare `covid_data_hurt` / `covid_data_helped` only appears with
n >= 8 out-of-sample origins.

CAVEATS: (a) B has fewer training rows than A, so any A-B gap mixes "COVID regime" with "less
data"; no volume-matched control is run. (b) Feature windows of retained post-COVID rows still see
COVID history. (c) There are no post-COVID scored origins (the panel ends 2020-09-21; the last
origin with a full 13-week target is 2020-06-22), so this cannot test the deployed use case
(forecasting AW2020 from a model trained through mid-2020) directly. (d) The harness's expanding
window uses origins strictly before `t` at a 4-week step, whose target windows partly extend past
`t` -- a pre-existing property of the harness, inherited unchanged in `walk_forward`.

Scratch outputs (per-origin metrics parquet) go under `%TEMP%/nss_covid/`, never into the committed
tables. Deterministic: LightGBM determinism params are inherited from `train_lightgbm`.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterable, Sequence
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import polars as pl

from nss.features.targets import HORIZON_WEEKS
from nss.models.backtest import (
    BOOTSTRAP_BLOCK_SIZE,
    BOOTSTRAP_N_RESAMPLES,
    BOOTSTRAP_SEED,
    COVID_WINDOW_END,
    COVID_WINDOW_START,
    WMAPE_WEIGHT_COL,
    Origin,
    block_bootstrap_ci,
    generate_origin_schedule,
)
from nss.models.backtest_v2 import MIN_ORIGINS_FOR_NON_DIRECTIONAL, identify_lightgbm_origins
from nss.models.final_forecast import FINAL_MODEL_CONFIG, final_training_origin_weeks
from nss.models.lightgbm_model import (
    INITIAL_POOL_SIZE,
    LGBMConfig,
    _expanding_train_origin_weeks,
    build_model_frame,
    feature_columns,
    predict_lightgbm,
    run_lightgbm_walk_forward,
    train_lightgbm,
)
from nss.models.metrics import METRIC_KEYS, score_predictions

MODEL_A = "A_all_data"
MODEL_B = "B_excl_covid_targets"

EPS = 1e-12

# Suffixes qualifying a verdict whose CI excludes zero but which is not strong evidence.
DIRECTIONAL_SUFFIX = "_directional_only"  # n_origins < MIN_ORIGINS_FOR_NON_DIRECTIONAL
IN_SAMPLE_SUFFIX = "_in_sample_only"  # scored on rows the model trained on: fit, not skill

# WMAPE is the only lower-is-better metric in `METRIC_KEYS`.
LOWER_IS_BETTER: frozenset[str] = frozenset({"wmape"})

DEFAULT_PANEL_PATH = Path("data/processed/style_week_panel.parquet")
DEFAULT_OUT_PATH = Path("reports/tables/covid_two_model_comparison.csv")
SCRATCH_DIR = Path(tempfile.gettempdir()) / "nss_covid"


def horizon_overlaps_window(
    origin_week: date,
    window_start: date = COVID_WINDOW_START,
    window_end: date = COVID_WINDOW_END,
    horizon_weeks: int = HORIZON_WEEKS,
) -> bool:
    """True iff `[origin+1w, origin+horizon_weeks w]` intersects `[window_start, window_end]`.

    Same predicate as the `is_covid` flag in `nss.models.backtest.generate_origin_schedule` and the
    target window of `nss.features.targets.compute_forward_target` (both bounds inclusive).
    """
    horizon_start = origin_week + timedelta(weeks=1)
    horizon_end = origin_week + timedelta(weeks=horizon_weeks)
    return horizon_start <= window_end and horizon_end >= window_start


def drop_covid_origins(origin_weeks: Iterable[date]) -> list[date]:
    """`origin_weeks` minus every origin whose target window overlaps the COVID window."""
    return [o for o in origin_weeks if not horizon_overlaps_window(o)]


def purge_overlapping_origins(
    train_weeks: Iterable[date], test_week: date, horizon_weeks: int = HORIZON_WEEKS
) -> list[date]:
    """Drop training origins whose target window overlaps the test origin's target window.

    Windows `[o+1w, o+Hw]` and `[t+1w, t+Hw]` overlap iff `|o - t| < H` weeks (in whole weeks:
    `|o - t| <= H - 1`), so origins strictly within that band (including `t` itself) are removed.
    """
    band = timedelta(weeks=horizon_weeks)
    return [o for o in train_weeks if abs(o - test_week) >= band]


def _score_origin(
    frame: pl.DataFrame, model: lgb.LGBMRegressor, columns: list[str], test_week: date
) -> dict[str, float]:
    """Score `model` on `frame`'s rows at `test_week` via the harness's `score_predictions`."""
    test_frame = frame.filter(pl.col("origin_week") == test_week)
    preds = predict_lightgbm(model, test_frame, columns)
    metrics = score_predictions(
        test_frame["y_true"].to_numpy(), preds, test_frame[WMAPE_WEIGHT_COL].to_numpy()
    )
    metrics["n_eval"] = float(metrics["n_eval"])
    return metrics


def _row(design: str, origin: Origin, model: str, metrics: dict[str, float]) -> dict[str, object]:
    """One per-origin result row."""
    return {
        "design": design,
        "origin_week": origin.origin_week,
        "is_covid": origin.is_covid,
        "model": model,
        **metrics,
    }


def run_purged(
    panel: pl.DataFrame, test_origins: Sequence[Origin], config: LGBMConfig = FINAL_MODEL_CONFIG
) -> pl.DataFrame:
    """Design 1: purged leave-one-origin-out, A vs B, per test origin. See module docstring."""
    wide_weeks = final_training_origin_weeks(panel)
    frame = build_model_frame(panel, wide_weeks)
    columns = feature_columns(frame)
    wide_set = set(wide_weeks)
    rows: list[dict[str, object]] = []
    for origin in test_origins:
        if origin.origin_week not in wide_set:
            raise ValueError(f"test origin {origin.origin_week} is not in the wide weekly set")
        weeks_a = purge_overlapping_origins(wide_weeks, origin.origin_week)
        weeks_b = drop_covid_origins(weeks_a)
        for name, weeks in ((MODEL_A, weeks_a), (MODEL_B, weeks_b)):
            train = frame.filter(pl.col("origin_week").is_in(weeks))
            model = train_lightgbm(train, config, columns)
            metrics = _score_origin(frame, model, columns, origin.origin_week)
            metrics["n_train_rows"] = float(train.height)
            rows.append(_row("purged", origin, name, metrics))
    return pl.DataFrame(rows)


def run_walk_forward(panel: pl.DataFrame, config: LGBMConfig = FINAL_MODEL_CONFIG) -> pl.DataFrame:
    """Design 2: harness expanding-window walk-forward; A via the harness itself, B own loop."""
    origins = generate_origin_schedule(panel)
    per_origin_a, _cfg, _grid = run_lightgbm_walk_forward(panel, origins, config=config)
    frame = build_model_frame(panel, [o.origin_week for o in origins])
    columns = feature_columns(frame)

    rows: list[dict[str, object]] = []
    for row in per_origin_a.iter_rows(named=True):
        metrics = {k: float(row[k]) for k in ("n_eval", *METRIC_KEYS)}
        origin = next(o for o in origins if o.origin_week == row["origin_week"])
        rows.append(_row("walk_forward", origin, MODEL_A, metrics))

    for test_index in range(INITIAL_POOL_SIZE, len(origins)):
        origin = origins[test_index]
        weeks_a = _expanding_train_origin_weeks(origins, test_index)
        weeks_b = set(drop_covid_origins(weeks_a))
        if not origin.is_covid and weeks_a != weeks_b:
            raise AssertionError(f"non-COVID test origin {origin.origin_week} has COVID train rows")
        train = frame.filter(pl.col("origin_week").is_in(weeks_b))
        model = train_lightgbm(train, config, columns)
        metrics = _score_origin(frame, model, columns, origin.origin_week)
        metrics["n_train_rows"] = float(train.height)
        rows.append(_row("walk_forward", origin, MODEL_B, metrics))
    return pl.DataFrame(rows, infer_schema_length=None)


def run_frozen_in_sample(
    panel: pl.DataFrame, test_origins: Sequence[Origin], config: LGBMConfig = FINAL_MODEL_CONFIG
) -> pl.DataFrame:
    """Design 3: A and B each trained once on the wide set, scored IN-SAMPLE on `test_origins`."""
    wide_weeks = final_training_origin_weeks(panel)
    frame = build_model_frame(panel, wide_weeks)
    columns = feature_columns(frame)
    rows: list[dict[str, object]] = []
    for name, weeks in ((MODEL_A, wide_weeks), (MODEL_B, drop_covid_origins(wide_weeks))):
        train = frame.filter(pl.col("origin_week").is_in(weeks))
        model = train_lightgbm(train, config, columns)
        for origin in test_origins:
            metrics = _score_origin(frame, model, columns, origin.origin_week)
            metrics["n_train_rows"] = float(train.height)
            rows.append(_row("frozen_in_sample", origin, name, metrics))
    return pl.DataFrame(rows)


def verdict_for(metric: str, diff: float, ci_lo: float, ci_hi: float) -> str:
    """Direction-aware verdict on the paired difference `diff = B - A` (B excludes COVID rows).

    Returns `covid_data_helped` / `covid_data_hurt` when the CI excludes zero in the corresponding
    direction, else `no_difference` (also for a NaN CI -- "can't tell" is never a finding).
    """
    if any(v != v for v in (diff, ci_lo, ci_hi)):
        return "no_difference"
    lower = metric in LOWER_IS_BETTER
    # EPS: paired diffs of identical predictions can come out as ~1e-17 float noise; that must not
    # read as "CI excludes zero".
    b_better = (ci_hi < -EPS) if lower else (ci_lo > EPS)
    b_worse = (ci_lo > EPS) if lower else (ci_hi < -EPS)
    if b_better:
        return "covid_data_hurt"  # dropping COVID rows improved the metric
    if b_worse:
        return "covid_data_helped"
    return "no_difference"


def qualify_verdict(verdict: str, design: str, n_origins: int) -> str:
    """Tag a non-null verdict as in-sample-only or directional-only so it cannot be over-read.

    `no_difference` is never qualified. In-sample designs are tagged regardless of n; otherwise
    fewer than `MIN_ORIGINS_FOR_NON_DIRECTIONAL` origins (block bootstrap with only a couple of
    distinct blocks -- CI is near-degenerate) is tagged directional-only.
    """
    if verdict == "no_difference":
        return verdict
    if design == "frozen_in_sample":
        return verdict + IN_SAMPLE_SUFFIX
    if n_origins < MIN_ORIGINS_FOR_NON_DIRECTIONAL:
        return verdict + DIRECTIONAL_SUFFIX
    return verdict


def paired_rows(
    per_origin: pl.DataFrame, design: str, split: str, origin_weeks: Sequence[date]
) -> list[dict[str, object]]:
    """Long-format A-vs-B rows (one per metric) for `origin_weeks` of one design. See docstring."""
    sub = per_origin.filter(
        (pl.col("design") == design) & pl.col("origin_week").is_in(list(origin_weeks))
    )
    a = sub.filter(pl.col("model") == MODEL_A).sort("origin_week")
    b = sub.filter(pl.col("model") == MODEL_B).sort("origin_week")
    if a["origin_week"].to_list() != b["origin_week"].to_list():
        raise ValueError("model A and B do not share the same origins")
    out: list[dict[str, object]] = []
    for metric in METRIC_KEYS:
        a_vals, b_vals = a[metric].to_list(), b[metric].to_list()
        diffs = [bv - av for av, bv in zip(a_vals, b_vals, strict=True)]
        kwargs = {
            "block_size": BOOTSTRAP_BLOCK_SIZE,
            "n_resamples": BOOTSTRAP_N_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
        }
        a_mean, a_lo, a_hi = block_bootstrap_ci(a_vals, **kwargs)
        b_mean, b_lo, b_hi = block_bootstrap_ci(b_vals, **kwargs)
        diff, lo, hi = block_bootstrap_ci(diffs, **kwargs)
        out.append(
            {
                "design": design,
                "split": split,
                "metric": metric,
                "model_a": a_mean,
                "model_a_ci_lo": a_lo,
                "model_a_ci_hi": a_hi,
                "model_b": b_mean,
                "model_b_ci_lo": b_lo,
                "model_b_ci_hi": b_hi,
                "diff": diff,
                "ci_lo": lo,
                "ci_hi": hi,
                "n_origins": a.height,
                "directional_only": a.height < MIN_ORIGINS_FOR_NON_DIRECTIONAL,
                "verdict": qualify_verdict(verdict_for(metric, diff, lo, hi), design, a.height),
            }
        )
    return out


def build_comparison(panel: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run all three designs; return `(per_origin, comparison_long)`."""
    schedule = generate_origin_schedule(panel)
    shared = identify_lightgbm_origins(panel)  # the 12 origins every method shares
    shared_noncovid = [o for o in shared if not o.is_covid]
    all_noncovid = [o for o in schedule if not o.is_covid]
    print(f"shared origins: {len(shared)}; non-COVID among them: {len(shared_noncovid)}")
    print(f"  {[o.origin_week.isoformat() for o in shared_noncovid]}")
    print(f"schedule origins: {len(schedule)}; non-COVID: {len(all_noncovid)}")
    print(f"  {[o.origin_week.isoformat() for o in all_noncovid]}")

    purged = run_purged(panel, all_noncovid)
    walk = run_walk_forward(panel)
    frozen = run_frozen_in_sample(panel, shared_noncovid)
    per_origin = pl.concat([purged, walk, frozen], how="diagonal")

    shared_weeks = [o.origin_week for o in shared]
    noncovid_weeks = [o.origin_week for o in shared_noncovid]
    covid_weeks = [o.origin_week for o in shared if o.is_covid]
    rows: list[dict[str, object]] = []
    rows += paired_rows(per_origin, "purged", "shared_noncovid", noncovid_weeks)
    rows += paired_rows(per_origin, "purged", "all_noncovid", [o.origin_week for o in all_noncovid])
    rows += paired_rows(per_origin, "walk_forward", "shared_noncovid", noncovid_weeks)
    rows += paired_rows(per_origin, "walk_forward", "shared_covid", covid_weeks)
    rows += paired_rows(per_origin, "walk_forward", "shared_pooled", shared_weeks)
    rows += paired_rows(per_origin, "frozen_in_sample", "shared_noncovid", noncovid_weeks)
    return per_origin, pl.DataFrame(rows)


def main() -> None:
    """CLI entry point: run the comparison, write scratch per-origin parquet and the CSV."""
    panel = pl.read_parquet(DEFAULT_PANEL_PATH)
    per_origin, comparison = build_comparison(panel)

    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    scratch_path = SCRATCH_DIR / "covid_comparison_per_origin.parquet"
    per_origin.write_parquet(scratch_path)
    print(f"Wrote scratch {scratch_path} ({per_origin.height} rows)")

    DEFAULT_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.write_csv(DEFAULT_OUT_PATH)
    print(f"Wrote {DEFAULT_OUT_PATH} ({comparison.height} rows)")
    with pl.Config(
        tbl_rows=200,
        tbl_cols=20,
        fmt_str_lengths=40,
        tbl_width_chars=250,
        tbl_formatting="ASCII_FULL",
    ):
        print(
            comparison.select(
                "design",
                "split",
                "metric",
                "model_a",
                "model_b",
                "diff",
                "ci_lo",
                "ci_hi",
                "n_origins",
                "verdict",
            )
        )


if __name__ == "__main__":
    main()
