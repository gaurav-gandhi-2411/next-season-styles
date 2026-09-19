"""Task G2: LightGBM `lambdarank` (LambdaMART) model -- a ranking-objective alternative to the
existing L2-regression LightGBM model (`nss.models.lightgbm_model`), trained/scored through the
SAME walk-forward harness and feature/target pipeline.

MOTIVATION (see PLAN.md / Task G1, `reports/tables/g1_diagnostics_summary.csv`): G1 found that a
causally-clean 16-week-lag persistence oracle actually UNDERPERFORMS the current L2 model on all 3
headline metrics -- so this task is NOT "closing a gap to an achievable oracle" (that framing was
the task's original, now-falsified, motivating hypothesis). What DID hold up from G1: prediction
compression (predictions ~18% narrower by std than actuals) consistent with L2-on-log1p
regression-to-the-mean, and near-tie noise at the #3/#4 cutoff (mean gap 0.61% of the #3 value) that
lowers the Precision@3 ceiling for ANY model, not specifically LightGBM. A ranking objective doesn't
share L2's mean-shrinkage defect and optimizes directly for the head-of-ranking decision (which
styles are in the top few) rather than pointwise accuracy across the whole distribution -- this
module tests whether that actually helps, honestly reporting whatever the result is.

REUSE, UNCHANGED: `nss.features.model_features.build_features`, `nss.features.targets.
compute_forward_target`, `nss.models.backtest`'s origin schedule / eval-set construction /
block-bootstrap machinery, `nss.models.lightgbm_model.build_model_frame` / `feature_columns` /
`_to_lgb_matrix` / `_categorical_indices` / `_expanding_train_origin_weeks` /
`LGBM_DETERMINISM_PARAMS` / `RANDOM_SEED`, and `nss.models.metrics.score_predictions`. Nothing about
the causal feature/target pipeline or the walk-forward split logic is touched by this module -- only
the model's own training objective (L2 regression -> `lambdarank`) and its label representation
(continuous `y_true` -> per-origin relevance GRADES, see RELEVANCE GRADES below) differ.

RELEVANCE GRADES (`compute_relevance_grades` / `add_relevance_grades`): `lambdarank` needs a
discrete, per-QUERY-GROUP (per-origin) relevance label, not the continuous `y_true` the L2 model
regresses against directly. Each origin's styles are graded by their OWN rank (by realised `y_true`,
descending, ties broken by row order -- the same stable-tie convention `nss.models.metrics` already
uses for top-K argsort) against `RELEVANCE_GRADE_BANDS`: true top-3 -> grade 4, top-10 -> grade 3,
top-50 -> grade 2, top-200 -> grade 1, everything else -> grade 0. This concentrates the ranking
loss's attention on the SAME head-of-ranking region the project's headline metrics
(Precision@3, Hit@3-in-top20/top10) score, which an L2 loss spread evenly across the whole
distribution does not do.

GROUPING FOR `lgb.LGBMRanker`: LightGBM's ranking objective needs each origin's rows CONTIGUOUS in
the training matrix, with a `group` array of per-origin row counts in that same contiguous order.
`build_model_frame`'s output is NOT origin-contiguous
(`nss.features.model_features.build_features` sorts by `[style_key, week_start]`, so rows are
contiguous per STYLE, not per origin) -- `_sort_for_ranking` re-sorts by `["origin_week",
"style_key"]` before every train/predict call. This sort key is a full row identity (one row per
`(style_key, origin_week)` pair, per `build_model_frame`'s own contract), so it is fully
deterministic on its own with no ties to break -- no `maintain_order` flag
is needed for THIS specific sort (unlike the D3a join fix elsewhere in this project, which fixed a
genuinely tie-prone/unstable ordering).

DETERMINISM: reuses `nss.models.lightgbm_model.LGBM_DETERMINISM_PARAMS` verbatim
(`deterministic=True`, `force_row_wise=True`, `num_threads=1`, and the three RNG sub-seeds, all
pinned to `RANDOM_SEED`) on the `lgb.LGBMRanker` constructor -- the same fix that made the L2 model
bit-identical within- and cross-process (see that module's DETERMINISM (A5) / (A5 FOLLOW-UP) (D3a)
docstring sections for the full mechanism) applies identically here: `LGBMRanker` is still a
gradient-boosted tree ensemble built via the same C++ histogram/split code, so the same floating-
point-summation-order and RNG-stream sources of nondeterminism apply, and the same fixes close them.
Verified via `tests/test_lambdarank_model.py`'s
`test_train_lambdarank_is_bit_identical_across_repeated_runs` (within-process) and
`tests/test_lambdarank_determinism_cross_process.py` (cross-process, via a
`subprocess.run` worker -- the ONLY way to exercise the `pl.Enum`/cross-process class of bug at all,
per the existing L2 test's own docstring).

TRUNCATION LEVEL SELECTION (JUDGMENT CALL, bounded): `lambdarank_truncation_level` controls how
many top-ranked items per query LightGBM's lambda-gradient computation actually considers -- a
value too small starves the loss of signal below that rank, too large wastes gradient budget on
irrelevant-tail pairs. Tried a SMALL, bounded set (`LAMBDARANK_TRUNCATION_CANDIDATES` = 10, 20,
30 -- bracketing the project's own Hit@3-in-top20/Hit@3-in-top10 headline cutoffs), validated via
the SAME
single-held-out-pool-origin split the L2 model's own `select_hyperparameters` uses
(`nss.models.lightgbm_model.INITIAL_POOL_SIZE`, pool origins 0..6 train / pool origin 7 validate --
never touching any walk-forward test origin). Selection criterion is validation NDCG@10 (not
RMSE -- RMSE is meaningless for a model whose raw output is an unscaled ranking score, not a
calibrated regression prediction; NDCG@10 is itself a ranking-quality metric already computed by
`nss.models.metrics.score_predictions`, so no new metric code is needed). All other
hyperparameters (`num_leaves`, `learning_rate`, `n_estimators`, `min_child_samples`) are held
FIXED at the L2 model's own winning config (`nss.models.final_forecast.FINAL_MODEL_CONFIG`) rather
than re-run through a
fresh grid search -- this isolates the comparison to "same tree-building budget, different
objective", which is the actual question this task asks, and keeps the search bounded per the task's
own "not a full hyperparameter search" instruction.

`eval_at=(3, 10, 20)` is passed to every `LGBMRanker.fit` call together with a SELF eval_set (the
same training data/group, not a held-out split) purely so LightGBM's own internal NDCG@{3,10,20}
book-keeping (`model.evals_result_`) is populated for inspection -- it is NEVER used for early
stopping (`n_estimators` stays fixed, per the L2 model's own fixed-round-count convention) and never
feeds the truncation-level SELECTION criterion above (that uses the project's own held-out
`score_predictions`/NDCG@10, a genuinely separate held-out origin).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from nss.models.backtest import (
    WMAPE_WEIGHT_COL,
    Origin,
    block_bootstrap_ci,
    generate_origin_schedule,
)
from nss.models.final_forecast import FINAL_MODEL_CONFIG
from nss.models.lightgbm_model import (
    INITIAL_POOL_SIZE,
    LGBM_DETERMINISM_PARAMS,
    RANDOM_SEED,
    LGBMConfig,
    _categorical_indices,
    _expanding_train_origin_weeks,
    _to_lgb_matrix,
    build_model_frame,
    feature_columns,
)
from nss.models.metrics import METRIC_KEYS, score_predictions

METHOD_NAME = "lambdarank"

LGBM_RANKER_OBJECTIVE = "lambdarank"

# See module docstring RELEVANCE GRADES. Ordered (rank_cutoff, grade); applied loosest-cutoff-first
# so a tighter (higher-grade) band always overwrites a looser one for ranks it also covers.
RELEVANCE_GRADE_BANDS: tuple[tuple[int, int], ...] = ((200, 1), (50, 2), (10, 3), (3, 4))
DEFAULT_RELEVANCE_GRADE = 0

# See module docstring TRUNCATION LEVEL SELECTION.
LAMBDARANK_TRUNCATION_CANDIDATES: tuple[int, ...] = (10, 20, 30)
EVAL_AT: tuple[int, ...] = (3, 10, 20)

# Held fixed at the L2 model's own winning config -- see module docstring TRUNCATION LEVEL
# SELECTION for why only lambdarank_truncation_level is tuned in this task.
LAMBDARANK_CONFIG: LGBMConfig = dict(FINAL_MODEL_CONFIG)

DEFAULT_PANEL_PATH = Path("data/processed/style_week_panel.parquet")
DEFAULT_PER_ORIGIN_OUT_PATH = Path("reports/tables/backtest_per_origin_lambdarank.csv")
DEFAULT_SUMMARY_OUT_PATH = Path("reports/tables/backtest_summary_lambdarank.csv")


@dataclass(frozen=True)
class TruncationSearchResult:
    """One `lambdarank_truncation_level` search outcome. See module docstring TRUNCATION LEVEL
    SELECTION. `all_results` keeps every candidate's validation NDCG@10, not just the winner."""

    best_truncation_level: int
    all_results: list[dict[str, float]]


def compute_relevance_grades(y_true: np.ndarray) -> np.ndarray:
    """Per-origin relevance grades from realised `y_true`, for ONE origin's eval set.

    See module docstring RELEVANCE GRADES. Ranks `y_true` descending (ties broken by original row
    order via `kind="stable"`, the same tie-break convention `nss.models.metrics` uses for its own
    top-K argsorts), then maps rank -> grade via `RELEVANCE_GRADE_BANDS`.

    Args:
        y_true: Realised target values for one origin's eval set.

    Returns:
        Integer grades, same length/order as `y_true`.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    n = y_true_arr.shape[0]
    order = np.argsort(-y_true_arr, kind="stable")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(1, n + 1)

    grades = np.full(n, DEFAULT_RELEVANCE_GRADE, dtype=np.int64)
    for cutoff, grade in RELEVANCE_GRADE_BANDS:
        grades[ranks <= cutoff] = grade
    return grades


def add_relevance_grades(model_frame: pl.DataFrame) -> pl.DataFrame:
    """Attach a `relevance_grade` column to `model_frame`, computed per `origin_week` group.

    Polars equivalent of `compute_relevance_grades`, vectorized across every origin at once via
    `.rank(method="ordinal", descending=True).over("origin_week")` -- verified (see module tests)
    to break ties identically to `compute_relevance_grades`'s `np.argsort(kind="stable")` (first
    row in original order gets the lower rank number).

    MUST be called AFTER the caller has already captured `feature_columns(model_frame)` -- the new
    `relevance_grade` column is a direct function of `y_true` (the answer being predicted) and
    must NEVER be treated as a model feature; capturing the column list first, then calling this
    function, is how every walk-forward/search function in this module avoids that leak.

    Args:
        model_frame: `nss.models.lightgbm_model.build_model_frame`'s output (must have `y_true`,
            `origin_week`).

    Returns:
        `model_frame` with `relevance_grade` (int) appended.
    """
    rank_expr = pl.col("y_true").rank(method="ordinal", descending=True).over("origin_week")
    grade_expr: pl.Expr = pl.lit(DEFAULT_RELEVANCE_GRADE)
    for cutoff, grade in RELEVANCE_GRADE_BANDS:
        grade_expr = pl.when(rank_expr <= cutoff).then(pl.lit(grade)).otherwise(grade_expr)
    return model_frame.with_columns(grade_expr.alias("relevance_grade"))


def _sort_for_ranking(frame: pl.DataFrame) -> pl.DataFrame:
    """Sort a `relevance_grade`-bearing frame so each `origin_week`'s rows are contiguous, in a
    fully deterministic order. See module docstring GROUPING FOR `lgb.LGBMRanker`."""
    return frame.sort(["origin_week", "style_key"])


def _group_sizes(sorted_frame: pl.DataFrame) -> list[int]:
    """Per-`origin_week` row counts, in the SAME contiguous order as `sorted_frame`'s own rows
    (`maintain_order=True` preserves first-occurrence order, which after `_sort_for_ranking` is
    ascending `origin_week` order) -- exactly the `group` array `lgb.LGBMRanker.fit` requires."""
    counts = sorted_frame.group_by("origin_week", maintain_order=True).agg(pl.len())
    return counts["len"].to_list()


def train_lambdarank(
    train_frame: pl.DataFrame,
    config: LGBMConfig,
    truncation_level: int,
    columns: list[str],
) -> lgb.LGBMRanker:
    """Fit one `lgb.LGBMRanker` on `train_frame`'s pooled, `relevance_grade`-bearing rows.

    Args:
        train_frame: A `build_model_frame`-shaped frame with `relevance_grade` already attached
            (see `add_relevance_grades`); must have `y_true` plus every `columns` entry.
        config: Fixed LightGBM tree hyperparameters (`LAMBDARANK_CONFIG` by default -- see module
            docstring TRUNCATION LEVEL SELECTION).
        truncation_level: `lambdarank_truncation_level` value to use.
        columns: `feature_columns(...)` captured BEFORE `add_relevance_grades` was called on the
            parent frame (see that function's docstring for why the ordering matters).

    Returns:
        The fitted `lgb.LGBMRanker`.
    """
    sorted_frame = _sort_for_ranking(train_frame)
    X = _to_lgb_matrix(sorted_frame, columns)
    y = sorted_frame["relevance_grade"].to_numpy().astype(np.int32)
    group = _group_sizes(sorted_frame)
    cat_indices = _categorical_indices(sorted_frame, columns)

    model = lgb.LGBMRanker(
        objective=LGBM_RANKER_OBJECTIVE,
        random_state=RANDOM_SEED,
        verbosity=-1,
        num_leaves=int(config["num_leaves"]),
        learning_rate=float(config["learning_rate"]),
        n_estimators=int(config["n_estimators"]),
        min_child_samples=int(config["min_child_samples"]),
        lambdarank_truncation_level=int(truncation_level),
        **LGBM_DETERMINISM_PARAMS,
    )
    # eval_at/eval_set: self-eval bookkeeping only, never early stopping -- see module docstring.
    model.fit(
        X,
        y,
        group=group,
        categorical_feature=cat_indices,
        feature_name=columns,
        eval_X=X,
        eval_y=y,
        eval_group=[group],
        eval_at=list(EVAL_AT),
    )
    return model


def predict_lambdarank(
    model: lgb.LGBMRanker, frame: pl.DataFrame, columns: list[str]
) -> np.ndarray:
    """Predict `frame`'s rows (any row order -- `LGBMRanker.predict` needs no `group`) with a fitted
    `model`. `columns` must match the model's training columns (see `train_lambdarank`)."""
    X = _to_lgb_matrix(frame, columns)
    return model.predict(X)


def select_truncation_level(
    pool_frame: pl.DataFrame,
    pool_origins: list[Origin],
    columns: list[str],
    config: LGBMConfig = LAMBDARANK_CONFIG,
    candidates: Sequence[int] = LAMBDARANK_TRUNCATION_CANDIDATES,
) -> TruncationSearchResult:
    """Small bounded search over `lambdarank_truncation_level`, validated on a SINGLE held-out
    origin within the initial pool. See module docstring TRUNCATION LEVEL SELECTION.

    Args:
        pool_frame: `relevance_grade`-bearing `build_model_frame` output, filtered/available for at
            least `pool_origins`'s origin weeks.
        pool_origins: The initial training pool's `Origin`s, in chronological order. Must have at
            least 2 entries.
        columns: `feature_columns(pool_frame)`, captured before `relevance_grade` was attached.
        config: Fixed tree hyperparameters. Defaults to `LAMBDARANK_CONFIG`.
        candidates: Truncation levels to try. Defaults to `LAMBDARANK_TRUNCATION_CANDIDATES`.

    Returns:
        `TruncationSearchResult` with the best (highest validation NDCG@10) truncation level and
        every candidate's result.
    """
    if len(pool_origins) < 2:
        raise ValueError("select_truncation_level needs at least 2 pool origins (train + val)")

    train_weeks = {o.origin_week for o in pool_origins[:-1]}
    val_week = pool_origins[-1].origin_week
    train_frame = pool_frame.filter(pl.col("origin_week").is_in(train_weeks))
    val_frame = pool_frame.filter(pl.col("origin_week") == val_week)
    y_val_true = val_frame["y_true"].to_numpy()

    results: list[dict[str, float]] = []
    best_truncation: int | None = None
    best_ndcg = -float("inf")
    for truncation_level in candidates:
        model = train_lambdarank(train_frame, config, truncation_level, columns)
        preds = predict_lambdarank(model, val_frame, columns)
        ndcg = score_predictions(y_val_true, preds)["ndcg_at_10"]
        results.append(
            {"lambdarank_truncation_level": float(truncation_level), "val_ndcg_at_10": ndcg}
        )
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_truncation = truncation_level

    assert best_truncation is not None  # candidates is non-empty by construction
    return TruncationSearchResult(best_truncation_level=best_truncation, all_results=results)


def run_lambdarank_walk_forward(
    panel: pl.DataFrame,
    origins: list[Origin] | None = None,
    pool_size: int = INITIAL_POOL_SIZE,
    config: LGBMConfig = LAMBDARANK_CONFIG,
    truncation_level: int | None = None,
) -> tuple[pl.DataFrame, int, list[dict[str, float]]]:
    """Run the full expanding-window walk-forward `lambdarank` evaluation.

    Mirrors `nss.models.lightgbm_model.run_lightgbm_walk_forward`'s expanding-window loop exactly
    (same pool/test-origin split, same `_expanding_train_origin_weeks`), swapping in
    `train_lambdarank`/`predict_lambdarank` and a `relevance_grade`-bearing frame.

    Args:
        panel: The dense style-week panel.
        origins: Origins to evaluate over. Defaults to `generate_origin_schedule(panel)`.
        pool_size: Origins reserved as the training pool. Defaults to `INITIAL_POOL_SIZE` (8).
        config: Fixed tree hyperparameters. Defaults to `LAMBDARANK_CONFIG`.
        truncation_level: `lambdarank_truncation_level` to use for every expanding-window fit. If
            `None`, `select_truncation_level` is run once (on the pool only) to pick it.

    Returns:
        `(per_origin, truncation_level_used, search_results)` -- `per_origin` has the same columns
        as `nss.models.lightgbm_model.run_lightgbm_walk_forward`'s output (`method="lambdarank"`).
        `search_results` is `[]` if `truncation_level` was passed in directly.
    """
    if origins is None:
        origins = generate_origin_schedule(panel)
    if pool_size >= len(origins):
        raise ValueError(f"pool_size ({pool_size}) must leave at least 1 walk-forward test origin")

    origin_weeks = [o.origin_week for o in origins]
    model_frame = build_model_frame(panel, origin_weeks)
    columns = feature_columns(model_frame)  # captured BEFORE relevance_grade is attached
    model_frame = add_relevance_grades(model_frame)

    search_results: list[dict[str, float]] = []
    if truncation_level is None:
        pool_origins = origins[:pool_size]
        search = select_truncation_level(model_frame, pool_origins, columns, config)
        truncation_level = search.best_truncation_level
        search_results = search.all_results

    rows: list[dict[str, object]] = []
    for test_index in range(pool_size, len(origins)):
        test_origin = origins[test_index]
        train_weeks = _expanding_train_origin_weeks(origins, test_index)
        train_frame = model_frame.filter(pl.col("origin_week").is_in(train_weeks))
        test_frame = model_frame.filter(pl.col("origin_week") == test_origin.origin_week)

        model = train_lambdarank(train_frame, config, truncation_level, columns)
        preds = predict_lambdarank(model, test_frame, columns)
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
                **metrics,
            }
        )
    return pl.DataFrame(rows), truncation_level, search_results


def summarize_lambdarank(per_origin: pl.DataFrame) -> pl.DataFrame:
    """Pooled / COVID-only / non-COVID-only block-bootstrap aggregates for the `lambdarank` method.

    Identical aggregation logic to `nss.models.lightgbm_model.summarize_lightgbm` (reuses
    `block_bootstrap_ci`, same block size/resamples/seed), restricted to a single method so the
    output schema matches `backtest_summary.csv`/`backtest_summary_lightgbm.csv` exactly.
    """
    splits = (
        ("pooled", per_origin),
        ("covid", per_origin.filter(pl.col("is_covid"))),
        ("non_covid", per_origin.filter(~pl.col("is_covid"))),
    )
    rows: list[dict[str, object]] = []
    for split_name, split_df in splits:
        row: dict[str, object] = {
            "method": METHOD_NAME,
            "split": split_name,
            "n_origins": split_df.height,
        }
        for metric in METRIC_KEYS:
            mean, ci_low, ci_high = block_bootstrap_ci(split_df[metric].to_list())
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = ci_low
            row[f"{metric}_ci_high"] = ci_high
        rows.append(row)
    return pl.DataFrame(rows)


def main() -> None:
    """CLI entry point: run the truncation-level search + walk-forward eval; write output
    artifacts."""
    panel = pl.read_parquet(DEFAULT_PANEL_PATH)
    origins = generate_origin_schedule(panel)
    print(
        f"Origin schedule: {len(origins)} origins, {origins[0].origin_week} .. "
        f"{origins[-1].origin_week}"
    )

    per_origin, truncation_level, search_results = run_lambdarank_walk_forward(panel, origins)
    print(f"Truncation-level search ({len(search_results)} candidates tried):")
    for r in search_results:
        print(f"  {r}")
    print(f"Winning lambdarank_truncation_level: {truncation_level}")

    summary = summarize_lambdarank(per_origin)

    DEFAULT_PER_ORIGIN_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    per_origin.write_csv(DEFAULT_PER_ORIGIN_OUT_PATH)
    summary.write_csv(DEFAULT_SUMMARY_OUT_PATH)
    print(f"Wrote {DEFAULT_PER_ORIGIN_OUT_PATH} ({per_origin.height} rows)")
    print(f"Wrote {DEFAULT_SUMMARY_OUT_PATH} ({summary.height} rows)")


if __name__ == "__main__":
    main()
