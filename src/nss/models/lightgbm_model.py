"""LightGBM model: expanding-window walk-forward evaluation through the existing backtest harness,
a small bounded hyperparameter search, and global SHAP feature importance.

TRAINING POOL SPLIT (JUDGMENT CALL): the first `INITIAL_POOL_SIZE` (8) of the 20 rolling origins
(`nss.models.backtest.generate_origin_schedule`'s real-panel schedule: `2018-12-17 .. 2020-06-01`)
are reserved as an initial training pool (origins index 0..7, `2018-12-17 .. 2019-07-01`) and are
NEVER scored as walk-forward test origins. Walk-forward test origins are index 8..19 (12 origins,
`2019-07-29 .. 2020-06-01`), which splits 5 non-COVID + 7 COVID -- comparable in shape to the full
20-origin schedule's own 13/7 split, so the walk-forward test set isn't accidentally COVID-only or
COVID-free. 8 is chosen (over the suggested 8-10 range) because it is the smallest pool
that still leaves a genuinely separate origin for hyperparameter validation (pool origins 0..6 for
training, pool origin 7 held out -- see HYPERPARAMETER SELECTION below) while giving the FIRST
walk-forward test origin (index 8) an expanding training window built from 7 full origins' worth of
pooled `(style_key, origin_week)` rows -- comfortably more than one 13-week horizon
(`nss.features.targets.HORIZON_WEEKS`) of style-level history.

HYPERPARAMETER SELECTION: a SMALL (6-config) grid over `num_leaves`, `learning_rate`,
`n_estimators`, `min_child_samples` (LightGBM's sklearn-API name for `min_data_in_leaf`), validated
via a SINGLE held-out origin WITHIN the initial pool only (train on pool origins 0..6, validate on
pool origin 7) -- never touching the walk-forward test origins (8..19), to avoid contaminating the
honest walk-forward evaluation. Validation RMSE (not single-origin Precision@3) is the selection
criterion: Precision@3 on one held-out origin only takes 4 possible values (0, 1/3, 2/3, 1) -- far
too coarse to reliably separate 6 close configs -- whereas RMSE uses the full continuous
distribution of that origin's ~3,000-style eval set. The winning config is locked in and reused,
UNCHANGED, for every expanding-window fit in the walk-forward loop -- no re-tuning after seeing
walk-forward results (see module docstring of the calling script / PLAN.md for why: re-tuning
against the test set would be exactly the test-set leakage this split is designed to prevent).

EXPANDING WINDOW: for walk-forward test origin index `t` (t in 8..19), the model is trained on the
POOLED feature-target rows from every origin STRICTLY BEFORE `t` (origins 0..t-1) -- the training
set grows by ~3,000 rows each step as `t` walks forward. Rows with a null target are dropped (can't
train or score on them, per `nss.features.targets.compute_forward_target`'s own null-handling
rule); null FEATURE values are left as NaN/missing and handled natively by LightGBM (never
imputed), per `nss.features.model_features`'s own documented null convention.

ENVIRONMENT GOTCHA (why this module never hands LightGBM a pandas DataFrame): on this project's
actual dev environment (lightgbm==4.7.0, numpy==2.5.3, pandas==2.3.3, Windows), passing ANY pandas
DataFrame into `lightgbm.Dataset`/`LGBMRegressor.fit` -- even a plain all-float, no-null, no-
categorical one -- crashes the interpreter with a native `OSError: access violation` inside
`LGBM_DatasetSetField`, reproduced with a minimal isolated repro before writing this module. Plain
`numpy.float64` 2-D arrays do NOT crash (verified the same way), so every LightGBM-facing helper
here (`_to_lgb_matrix`, `train_lightgbm`, `predict_lightgbm`) builds a numpy matrix directly from
polars, and categorical columns are passed as integer category-code columns (`Series.to_physical()`
-- see CATEGORICAL ENCODING below) with LightGBM's own `categorical_feature=<column indices>`
support, rather than relying on pandas `category` dtype. `shap.TreeExplainer`/`shap.summary_plot`
are unaffected (they read the already-fitted booster's tree structure, not LightGBM's own Dataset
C API) and are called with the same numpy matrix + `feature_names=...`, so this module has no
pandas dependency at all.

CATEGORICAL ENCODING: the 5 `STYLE_KEY_COLS` attribute columns come out of `build_features` already
cast to a fixed, sorted `pl.Enum` (NOT `pl.Categorical` -- see DETERMINISM (CROSS-PROCESS) below for
why: `pl.Categorical`'s dictionary is built via an internal, non-deterministic-across-processes
unique-value collection, CONFIRMED via a minimal repro to assign different integer codes to the
same category string across separate `uv run` invocations on the identical input, which flips
which side of a LightGBM categorical split a row falls on -- a real, independent source of
cross-process nondeterminism, though MEASURED to be a small contributor to this project's actual
prediction jitter relative to the row-order bug below). `Series.to_physical()` maps each value to
its integer category code (consistent across any filtered subset of the SAME parent
`pl.DataFrame`/`Series`, because polars filtering does not rebuild the category dictionary --
verified directly, and true of `pl.Enum` the same way it was true of `pl.Categorical`) and leaves
nulls as null, which `.cast(pl.Float64)` turns into `NaN` -- exactly the "missing category" signal
LightGBM expects. This is why `build_model_frame` is called ONCE across ALL origins and every
train/test split downstream is a `.filter()` of that single frame: splitting AFTER building keeps
every subset's category codes aligned to the same dictionary; rebuilding features separately per
split would risk two frames assigning different codes to the same category string (this risk is
now purely hypothetical for a single process run -- `pl.Enum`'s mapping is a pure function of the
sorted vocabulary -- but the ONE-CALL contract is kept as the simplest way to guarantee every split
sees literally the same dtype object).

SHAP MODEL CHOICE (JUDGMENT CALL): global SHAP importance is computed from a model trained on the
FULL pooled walk-forward train+test set (every one of the 20 origins' rows, i.e. all of
`build_model_frame`'s output) using the winning hyperparameter config -- not the walk-forward loop's
own last expanding-window fit (which stops one origin short, at origins 0..18). The full-data model
strictly dominates it on training data volume and is the closer proxy for what the later
final-forecast step (trained on all data through 2020-09-22) will actually look like -- this is a
SHAP-analysis-only model, never scored against a test origin, so it carries
no leakage risk for the walk-forward evaluation numbers reported alongside it.

DETERMINISM (WITHIN-PROCESS): `random_state=RANDOM_SEED` alone was observed NOT to make repeated
training runs bit-identical (a few percent variation in predicted values for the same style across
runs) -- LightGBM's sklearn `random_state` only seeds one of several internal RNG streams, and
multi-threaded histogram building is order-dependent regardless of seeding. `train_lightgbm`
therefore also passes `LGBM_DETERMINISM_PARAMS` (`deterministic=True`, `force_row_wise=True`,
`num_threads=1`, and explicit `bagging_seed`/`feature_fraction_seed`/`data_random_seed`, all pinned
to `RANDOM_SEED`) to every `lgb.LGBMRegressor` this module constructs. Verified bit-identical
(`np.array_equal`) via
`tests/test_lightgbm_model.py::test_train_lightgbm_is_bit_identical_across_repeated_runs`, at the
cost of `num_threads=1` giving up multi-threaded training speed -- acceptable here given this
project's dataset size (see PLAN.md for the measured wall-clock impact).

DETERMINISM (CROSS-PROCESS) -- this WITHIN-PROCESS fix alone did not close the loop: a
separate, CROSS-PROCESS source of jitter (reported as "~1%"; MEASURED via a controlled ablation on
the real production panel + `nss.models.final_forecast.FINAL_MODEL_CONFIG` to be as large as ~52%
relative / ~6.2 absolute on some styles -- the original "~1%" description undersold the real
magnitude) was found and root-caused via a 3-way ablation (full pristine code vs. two single-fix
variants, each run twice as separate `uv run` processes on the identical real panel):

  1. PRIMARY CAUSE, CONFIRMED (closes essentially all of the measured jitter on its own -- max
     abs diff drops from 6.14 to 5.3e-15, a ~1.15e15x reduction, see below): the
     `features.join(targets, ...)` call in `build_model_frame` (below) had no `maintain_order`
     argument. Polars does not guarantee a join's output row order is stable across runs (an
     implementation detail of its hash-join, observed to vary process-to-process on identical
     input). Since `deterministic=True`/`force_row_wise=True`/`num_threads=1` only guarantee
     "same data IN THE SAME ORDER -> same result" (LightGBM's histogram-building gradient/hessian
     accumulation is a floating-point summation, which is not associative -- a different row order
     is a different summation order, hence different bin statistics, hence potentially different
     split decisions, compounding across all `n_estimators` boosting rounds), an unordered join
     feeding LightGBM's TRAINING set was enough to make every downstream tree -- and therefore
     every prediction -- diverge between separate process runs, even with every LightGBM RNG seed
     already pinned. FIXED by adding `maintain_order="left"` to that join (see `build_model_frame`
     below): an isolated ablation with ONLY this one-line fix applied (leaving `pl.Categorical`
     unfixed) reduced the cross-process diff from max_abs=6.14 (45.1% relative) to
     max_abs=5.3e-15 -- floating-point-noise level, not zero, which is exactly the residual the
     second, independent cause below explains.
  2. SECONDARY CAUSE, CONFIRMED, MEASURED SMALL ON THIS MODEL: `pl.Categorical`'s dictionary
     construction (see CATEGORICAL ENCODING above) is independently non-deterministic across
     processes -- confirmed via a minimal repro (`pl.DataFrame(...).with_columns([pl.col(c).cast(
     pl.Categorical) for c in 5_cols])` on as few as 50 rows already differs across separate `uv
     run` invocations on identical input, once more than one categorical column is cast in the
     same `with_columns` call). On THIS project's specific model/data, an ablation with ONLY the
     `maintain_order="left"` fix applied (`pl.Categorical` left buggy) already leaves at most
     5.3e-15 absolute difference -- i.e. this cause's OWN measured contribution here is
     floating-point noise, not a visible percentage. It is fixed anyway (`pl.Enum`, an explicit
     sorted vocabulary -- see `nss.features.model_features` DETERMINISM (CROSS-PROCESS) section)
     because it is a real, independently-reproducible nondeterminism in a value LightGBM consumes
     directly via `categorical_feature=...`, and its magnitude on a DIFFERENT model/dataset (one
     where the categorical features drive more decisive splits) is not something this repo can
     assume will always stay negligible.

Combining both fixes (the one already-shipped in `build_model_frame`, plus `pl.Enum`) was verified
BIT-IDENTICAL (`np.array_equal`, max abs diff = 0.0) across two genuinely separate `uv run`
process invocations of the real `nss.models.final_forecast` pipeline on the real production panel
(1,980 forecast-eligible styles, all identical) -- see PLAN.md for the
full ablation table and `tests/test_determinism_cross_process.py` for the regression test.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")  # noqa: E402 -- must precede pyplot import; headless (CI/CLI) rendering only.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import polars as pl
import shap

from nss.features.model_features import build_features
from nss.features.targets import HORIZON_WEEKS, compute_forward_target
from nss.models.backtest import (
    WMAPE_WEIGHT_COL,
    Origin,
    block_bootstrap_ci,
    generate_origin_schedule,
)
from nss.models.metrics import METRIC_KEYS, score_predictions

METHOD_NAME = "lightgbm"

# See module docstring TRAINING POOL SPLIT.
INITIAL_POOL_SIZE = 8

RANDOM_SEED = 42
LGBM_OBJECTIVE = "regression"

# DETERMINISM (WITHIN-PROCESS): non-bit-identical predictions were observed across repeated
# training runs despite `random_state=RANDOM_SEED` -- LightGBM's `random_state`/`seed` sklearn
# param does NOT by itself fix every internal RNG stream (bagging, feature sampling, and the
# Dataset-construction "data" RNG each have their own seed knobs), and multi-threaded histogram
# building is order-dependent (floating-point summation is not associative), which reintroduces
# nondeterminism even with every seed fixed. `deterministic=True` + `force_row_wise=True` remove
# that within-run, multi-threaded nondeterminism from histogram building (LightGBM's own docs
# recommend both together for bit-exact repeatability); `num_threads=1` closes the remaining gap
# by removing thread-scheduling nondeterminism entirely, at a training-speed cost.
# All four extra seeds are pinned to `RANDOM_SEED` for a single source of truth, not because they
# need to differ from it.
LGBM_DETERMINISM_PARAMS: dict[str, bool | int] = {
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 1,
    "bagging_seed": RANDOM_SEED,
    "feature_fraction_seed": RANDOM_SEED,
    "data_random_seed": RANDOM_SEED,
}

LGBMConfig = dict[str, int | float]

# See module docstring HYPERPARAMETER SELECTION. 6 configs, varying model complexity (num_leaves),
# the learning_rate / n_estimators pairing (fewer, larger steps vs more, smaller ones), and
# min_child_samples (leaf-size regularization) -- the "usual LightGBM regression knobs",
# nothing more exotic.
HYPERPARAM_GRID: list[LGBMConfig] = [
    {"num_leaves": 15, "learning_rate": 0.05, "n_estimators": 200, "min_child_samples": 20},
    {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 200, "min_child_samples": 20},
    {"num_leaves": 31, "learning_rate": 0.10, "n_estimators": 100, "min_child_samples": 20},
    {"num_leaves": 15, "learning_rate": 0.10, "n_estimators": 100, "min_child_samples": 50},
    {"num_leaves": 63, "learning_rate": 0.05, "n_estimators": 200, "min_child_samples": 50},
    {"num_leaves": 31, "learning_rate": 0.03, "n_estimators": 300, "min_child_samples": 30},
]

SHAP_TOP_N = 15

DEFAULT_PANEL_PATH = Path("data/processed/style_week_panel.parquet")
DEFAULT_PER_ORIGIN_OUT_PATH = Path("reports/tables/backtest_per_origin_lightgbm.csv")
DEFAULT_SUMMARY_OUT_PATH = Path("reports/tables/backtest_summary_lightgbm.csv")
DEFAULT_SHAP_TABLE_OUT_PATH = Path("reports/tables/shap_global_importance.csv")
DEFAULT_SHAP_FIG_OUT_PATH = Path("reports/figures/shap_global_importance.png")


@dataclass(frozen=True)
class HyperparamSearchResult:
    """One hyperparameter grid search outcome: `best_config` plus every config's validation RMSE.

    `all_results` is kept (not discarded) so the search is fully reportable -- every config tried,
    and its validation RMSE, not just the winner.
    """

    best_config: LGBMConfig
    all_results: list[dict[str, float]]


def build_model_frame(panel: pl.DataFrame, origin_weeks: list[date]) -> pl.DataFrame:
    """Build the full feature + target frame for a set of origins, in ONE `build_features` call.

    Reuses `build_features` and `compute_forward_target` exactly as
    `nss.models.backtest.build_predictions_frame` does, but keeps every feature column (not just
    the WMAPE weight) since this frame is the actual LightGBM training/prediction input. See module
    docstring CATEGORICAL ENCODING for why this must be called ONCE across every origin a caller
    will ever split train/test from.

    Args:
        panel: The dense style-week panel.
        origin_weeks: Origin weeks to build the frame for.

    Returns:
        One row per `(style_key, origin_week)` with a non-null target, columns: `style_key`,
        `origin_week`, every feature column from `build_features` (including the categorical
        `STYLE_KEY_COLS` attributes and `WMAPE_WEIGHT_COL`), and `y_true` (never null).
    """
    features = build_features(panel, origin_weeks)
    targets = pl.concat(
        [compute_forward_target(panel, ow, horizon_weeks=HORIZON_WEEKS) for ow in origin_weeks]
    )
    # maintain_order="left" -- THIS IS THE PRIMARY FIX for the cross-process determinism bug
    # (see module docstring DETERMINISM (CROSS-PROCESS) for the full ablation). Without it,
    # this join's output row order is not guaranteed stable across separate process runs (a polars
    # hash-join implementation detail); since LightGBM's histogram-building gradient/hessian
    # accumulation is floating-point summation over the TRAINING rows in the order given (not
    # associative), a shuffled training-row order alone was enough to make trained models --
    # and therefore predictions -- diverge by tens of percent between separate `uv run` processes,
    # even with `deterministic=True`/`force_row_wise=True`/`num_threads=1`/every RNG seed already
    # pinned (those only guarantee "same order -> same result", not "any order -> same result").
    merged = features.join(
        targets.select("style_key", "origin_week", pl.col("target").alias("y_true")),
        on=["style_key", "origin_week"],
        how="inner",
        maintain_order="left",
    )
    return merged.filter(pl.col("y_true").is_not_null())


def feature_columns(model_frame: pl.DataFrame) -> list[str]:
    """The LightGBM feature column list: every `model_frame` column except identifiers/target."""
    excluded = {"style_key", "origin_week", "y_true"}
    return [c for c in model_frame.columns if c not in excluded]


def _categorical_indices(model_frame: pl.DataFrame, columns: list[str]) -> list[int]:
    """Positions (within `columns`) of the categorical (`pl.Enum` or `pl.Categorical`)-typed
    feature columns. `build_features` produces `pl.Enum` columns (see that module's docstring
    DETERMINISM section); `pl.Categorical` is matched too for robustness against any caller that
    hands this function a frame built another way."""
    categorical_base_types = (pl.Categorical, pl.Enum)
    return [
        i
        for i, c in enumerate(columns)
        if model_frame.schema[c].base_type() in categorical_base_types
    ]


def _to_lgb_matrix(frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
    """Build a plain `numpy.float64` matrix for LightGBM. See module docstring ENVIRONMENT GOTCHA.

    Categorical columns are converted to their integer category codes (`Series.to_physical()`,
    null-safe -- see module docstring CATEGORICAL ENCODING); numeric columns are cast to float64
    directly. Both null categoricals and null numerics become `NaN`, which LightGBM treats as
    "missing" natively.
    """
    cols = [frame[c].to_physical().cast(pl.Float64).to_numpy() for c in columns]
    return np.column_stack(cols) if cols else np.empty((frame.height, 0))


def train_lightgbm(
    train_frame: pl.DataFrame, config: LGBMConfig, columns: list[str]
) -> lgb.LGBMRegressor:
    """Fit one LightGBM regressor on `train_frame`'s pooled `(style_key, origin_week)` rows.

    Args:
        train_frame: A `build_model_frame`-shaped frame (must have `y_true` plus every `columns`
            entry).
        config: One entry from `HYPERPARAM_GRID` (or the winning config).
        columns: `feature_columns(...)` for the parent frame `train_frame` was filtered from (see
            module docstring CATEGORICAL ENCODING for why column identity/order must be consistent
            across every split of the same parent frame).

    Returns:
        The fitted `lgb.LGBMRegressor`.
    """
    X = _to_lgb_matrix(train_frame, columns)
    y = train_frame["y_true"].to_numpy().astype(np.float64)
    cat_indices = _categorical_indices(train_frame, columns)

    model = lgb.LGBMRegressor(
        objective=LGBM_OBJECTIVE,
        random_state=RANDOM_SEED,
        verbosity=-1,
        num_leaves=int(config["num_leaves"]),
        learning_rate=float(config["learning_rate"]),
        n_estimators=int(config["n_estimators"]),
        min_child_samples=int(config["min_child_samples"]),
        **LGBM_DETERMINISM_PARAMS,
    )
    model.fit(X, y, categorical_feature=cat_indices, feature_name=columns)
    return model


def predict_lightgbm(
    model: lgb.LGBMRegressor, frame: pl.DataFrame, columns: list[str]
) -> np.ndarray:
    """Predict `frame`'s rows with a fitted `model`. `columns` must match the model's training
    columns (see `train_lightgbm`)."""
    X = _to_lgb_matrix(frame, columns)
    return model.predict(X)


def select_hyperparameters(
    pool_frame: pl.DataFrame, pool_origins: list[Origin], columns: list[str]
) -> HyperparamSearchResult:
    """Small bounded grid search, validated on a SINGLE held-out origin within the initial pool.

    See module docstring HYPERPARAMETER SELECTION. Trains each `HYPERPARAM_GRID` config on
    `pool_origins[:-1]` and validates (RMSE, on the log1p target scale) on `pool_origins[-1]` --
    never touching any walk-forward test origin.

    Args:
        pool_frame: `build_model_frame`'s output, filtered/available for at least
            `pool_origins`'s origin weeks (extra origins are ignored -- this function filters
            internally).
        pool_origins: The initial training pool's `Origin`s, in chronological order (the schedule's
            first `INITIAL_POOL_SIZE` origins). Must have at least 2 entries.
        columns: `feature_columns(pool_frame)`.

    Returns:
        `HyperparamSearchResult` with the best (lowest validation RMSE) config and every config's
        result.
    """
    if len(pool_origins) < 2:
        raise ValueError("select_hyperparameters needs at least 2 pool origins (train + val)")

    train_weeks = {o.origin_week for o in pool_origins[:-1]}
    val_week = pool_origins[-1].origin_week
    train_frame = pool_frame.filter(pl.col("origin_week").is_in(train_weeks))
    val_frame = pool_frame.filter(pl.col("origin_week") == val_week)
    y_val = val_frame["y_true"].to_numpy().astype(np.float64)

    results: list[dict[str, float]] = []
    best_config: LGBMConfig | None = None
    best_rmse = float("inf")
    for config in HYPERPARAM_GRID:
        model = train_lightgbm(train_frame, config, columns)
        preds = predict_lightgbm(model, val_frame, columns)
        rmse = float(np.sqrt(np.mean((y_val - preds) ** 2)))
        results.append({**config, "val_rmse": rmse})
        if rmse < best_rmse:
            best_rmse = rmse
            best_config = config

    assert best_config is not None  # HYPERPARAM_GRID is non-empty by construction
    return HyperparamSearchResult(best_config=best_config, all_results=results)


def _expanding_train_origin_weeks(origins: list[Origin], test_index: int) -> set[date]:
    """The expanding-window training set of origin weeks for walk-forward test origin
    `origins[test_index]`: every origin STRICTLY BEFORE it (`origins[:test_index]`).

    Pulled out as its own function specifically so the causal-safety property (never includes the
    test origin or any later one) is directly unit-testable in isolation from the rest of the
    training/scoring pipeline -- see `tests/test_lightgbm_model.py`.
    """
    return {o.origin_week for o in origins[:test_index]}


def run_lightgbm_walk_forward(
    panel: pl.DataFrame,
    origins: list[Origin] | None = None,
    pool_size: int = INITIAL_POOL_SIZE,
    config: LGBMConfig | None = None,
) -> tuple[pl.DataFrame, LGBMConfig, list[dict[str, float]]]:
    """Run the full expanding-window walk-forward LightGBM evaluation.

    See module docstring TRAINING POOL SPLIT and EXPANDING WINDOW. For each origin in
    `origins[pool_size:]`, trains on every origin strictly before it (pooled, expanding) and scores
    via `nss.models.metrics.score_predictions` -- same per-origin row schema as
    `nss.models.backtest.run_backtest`, so the output is directly comparable / concatenable.

    Args:
        panel: The dense style-week panel.
        origins: Origins to evaluate over. Defaults to `generate_origin_schedule(panel)`.
        pool_size: Number of initial origins reserved as the training pool (never scored). Defaults
            to `INITIAL_POOL_SIZE` (8).
        config: The LightGBM hyperparameter config to use for every expanding-window fit. If
            `None`, `select_hyperparameters` is run once (on the pool only) to pick it -- see
            HYPERPARAMETER SELECTION.

    Returns:
        `(per_origin, config_used, hyperparam_search_results)` -- `per_origin` has the same
        columns as `nss.models.backtest.run_backtest`'s output (`origin_week`, `method`,
        `has_52w_lag`, `is_covid`, `n_eval_set`, `n_eval`, plus every `METRIC_KEYS` metric).
        `hyperparam_search_results` is `[]` if `config` was passed in directly (no search run).
    """
    if origins is None:
        origins = generate_origin_schedule(panel)
    if pool_size >= len(origins):
        raise ValueError(f"pool_size ({pool_size}) must leave at least 1 walk-forward test origin")

    origin_weeks = [o.origin_week for o in origins]
    model_frame = build_model_frame(panel, origin_weeks)
    columns = feature_columns(model_frame)

    grid_results: list[dict[str, float]] = []
    if config is None:
        pool_origins = origins[:pool_size]
        search = select_hyperparameters(model_frame, pool_origins, columns)
        config = search.best_config
        grid_results = search.all_results

    rows: list[dict[str, object]] = []
    for test_index in range(pool_size, len(origins)):
        test_origin = origins[test_index]
        train_weeks = _expanding_train_origin_weeks(origins, test_index)
        train_frame = model_frame.filter(pl.col("origin_week").is_in(train_weeks))
        test_frame = model_frame.filter(pl.col("origin_week") == test_origin.origin_week)

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
                **metrics,
            }
        )
    return pl.DataFrame(rows), config, grid_results


def summarize_lightgbm(per_origin: pl.DataFrame) -> pl.DataFrame:
    """Pooled / COVID-only / non-COVID-only block-bootstrap aggregates for the LightGBM method.

    Identical aggregation logic to `nss.models.backtest.summarize_backtest` (reuses
    `block_bootstrap_ci`, same block size / resamples / seed), restricted to a single method so the
    output schema matches `backtest_summary.csv` exactly and the two are directly comparable.
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


def train_full_model_for_shap(
    panel: pl.DataFrame, origins: list[Origin], config: LGBMConfig
) -> tuple[lgb.LGBMRegressor, pl.DataFrame, list[str]]:
    """Train the SHAP-analysis-only model on the FULL pooled walk-forward train+test set.

    See module docstring SHAP MODEL CHOICE for why this (not the walk-forward loop's last
    expanding-window fit) is used.

    Returns:
        `(model, model_frame, columns)` -- `model_frame` is every origin's rows (needed by the
        caller to run SHAP over the same data the model was trained on).
    """
    origin_weeks = [o.origin_week for o in origins]
    model_frame = build_model_frame(panel, origin_weeks)
    columns = feature_columns(model_frame)
    model = train_lightgbm(model_frame, config, columns)
    return model, model_frame, columns


def compute_global_shap_importance(
    model: lgb.LGBMRegressor, frame: pl.DataFrame, columns: list[str]
) -> pl.DataFrame:
    """Global SHAP feature importance: mean absolute SHAP value per feature, across `frame`.

    Returns:
        `feature`, `mean_abs_shap`, sorted descending by `mean_abs_shap`.
    """
    X = _to_lgb_matrix(frame, columns)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs = np.abs(shap_values).mean(axis=0)
    return pl.DataFrame({"feature": columns, "mean_abs_shap": mean_abs}).sort(
        "mean_abs_shap", descending=True
    )


def save_shap_summary_plot(
    model: lgb.LGBMRegressor, frame: pl.DataFrame, columns: list[str], out_path: Path
) -> None:
    """Save a standard SHAP beeswarm summary plot to `out_path` (parents created as needed)."""
    X = _to_lgb_matrix(frame, columns)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=columns, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    """CLI entry point: run the hyperparameter search, walk-forward eval, and global SHAP; write
    all output artifacts."""
    panel = pl.read_parquet(DEFAULT_PANEL_PATH)
    origins = generate_origin_schedule(panel)
    print(
        f"Origin schedule: {len(origins)} origins, {origins[0].origin_week} .. "
        f"{origins[-1].origin_week}"
    )
    print(
        f"  initial pool: origins[0:{INITIAL_POOL_SIZE}] "
        f"({origins[0].origin_week} .. {origins[INITIAL_POOL_SIZE - 1].origin_week})"
    )
    print(
        f"  walk-forward test: origins[{INITIAL_POOL_SIZE}:{len(origins)}] "
        f"({origins[INITIAL_POOL_SIZE].origin_week} .. {origins[-1].origin_week})"
    )

    per_origin, config, grid_results = run_lightgbm_walk_forward(panel, origins)
    print(f"Hyperparameter search ({len(grid_results)} configs tried):")
    for r in grid_results:
        print(f"  {r}")
    print(f"Winning config: {config}")

    summary = summarize_lightgbm(per_origin)

    DEFAULT_PER_ORIGIN_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    per_origin.write_csv(DEFAULT_PER_ORIGIN_OUT_PATH)
    summary.write_csv(DEFAULT_SUMMARY_OUT_PATH)
    print(f"Wrote {DEFAULT_PER_ORIGIN_OUT_PATH} ({per_origin.height} rows)")
    print(f"Wrote {DEFAULT_SUMMARY_OUT_PATH} ({summary.height} rows)")

    shap_model, shap_frame, shap_columns = train_full_model_for_shap(panel, origins, config)
    importance = compute_global_shap_importance(shap_model, shap_frame, shap_columns)
    DEFAULT_SHAP_TABLE_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    importance.write_csv(DEFAULT_SHAP_TABLE_OUT_PATH)
    print(f"Wrote {DEFAULT_SHAP_TABLE_OUT_PATH} ({importance.height} rows)")
    print(f"Top {SHAP_TOP_N} SHAP features:")
    for row in importance.head(SHAP_TOP_N).iter_rows(named=True):
        print(f"  {row['feature']}: {row['mean_abs_shap']:.5f}")

    save_shap_summary_plot(shap_model, shap_frame, shap_columns, DEFAULT_SHAP_FIG_OUT_PATH)
    print(f"Wrote {DEFAULT_SHAP_FIG_OUT_PATH}")


if __name__ == "__main__":
    main()
