"""Random-permutation floor: the "no signal at all" baseline every real method must beat.

WHY A RANDOM FLOOR (see PLAN / backtest_v2): the earlier claim "LightGBM wins decisively on
NDCG@10" was never checked against a random-guessing floor. A metric that scores well in absolute
terms can still be indistinguishable from chance once you know its floor -- NDCG@10 in particular
is NOT 0 for a random ranking (a random top-10 out of ~3,000 mostly-similar-magnitude styles can
still pick up meaningful relevance by chance), so "0.90 NDCG@10" only means something once compared
against what a coin flip would score on the SAME eval sets.

CONSTRUCTION: `RANDOM_FLOOR_METHOD` treats each origin's TRUE target vector as its own prediction,
randomly PERMUTED across styles (`permute_predictions`) -- i.e. "assign the right realized values to
the wrong styles" -- rather than drawing unrelated random numbers or random rank indices. This keeps
the random floor on the exact same scale as `y_true` for every origin, which matters for WMAPE (an
absolute-magnitude metric, not rank-based like the other 6): comparing WMAPE against an arbitrary-
scale random score would be meaningless, but comparing it against a random RELABELING of the true
values themselves is a well-defined, standard permutation-test floor that every metric in
`nss.models.metrics.METRIC_KEYS` (rank-based or magnitude-based) can be honestly scored against.

SEEDS + AGGREGATION (JUDGMENT CALL, see backtest_v2 report for the full writeup):
`RANDOM_FLOOR_SEEDS` (20 seeds, 0..19) are scored independently per origin via
`score_random_floor_per_origin`, producing one row per (origin, seed).
`aggregate_random_floor_over_seeds` then collapses the 20 seeds into ONE value per origin (the mean
across seeds) BEFORE the per-origin block-bootstrap -- i.e. the random floor's reported 95% CI is
computed by the *exact same* `nss.models.backtest.block_bootstrap_ci` machinery (block size 4, 2000
resamples, seed 42) as the 4 baselines and LightGBM, applied to this per-origin-mean-over-seeds
series. This was chosen (over e.g. a two-level origin+seed bootstrap) for direct methodological
comparability: every method's reported CI in `backtest_summary_v2.csv` is constructed by the
identical procedure, so the "does this method's CI overlap random_floor's CI" check compares like
with like. With 20 seeds per origin, seed-to-seed noise in the per-origin mean is already small
relative to origin-to-origin variation (the dominant source of uncertainty this project's
block-bootstrap is designed to capture -- see `nss.models.backtest` module docstring); a more
elaborate two-level resample was judged not worth the added complexity given that.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from nss.models.backtest import Origin
from nss.models.metrics import METRIC_KEYS, score_predictions

RANDOM_FLOOR_METHOD = "random_floor"

# 20 independent seeds -- see module docstring SEEDS + AGGREGATION.
RANDOM_FLOOR_SEEDS: tuple[int, ...] = tuple(range(20))


def permute_predictions(y_true: Sequence[float], seed: int) -> np.ndarray:
    """A uniform-random permutation of `y_true`, used as that seed's "prediction".

    See module docstring CONSTRUCTION for why permuting the true values (rather than drawing
    unrelated random numbers) is the right floor construction.

    Args:
        y_true: Realized target values, one per style, for a single origin.
        seed: RNG seed -- same seed always gives the same permutation for the same-length input
            (reproducibility), different seeds give (with overwhelming probability) different
            permutations (genuine randomness across seeds).

    Returns:
        A `numpy.float64` array, a permutation of `y_true`.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    rng = np.random.default_rng(seed)
    return rng.permutation(y_true_arr)


def score_random_floor_per_origin(
    predictions: pl.DataFrame,
    origins: Sequence[Origin],
    seeds: Sequence[int] = RANDOM_FLOOR_SEEDS,
) -> pl.DataFrame:
    """Score the random-permutation floor at every origin, for every seed.

    Args:
        predictions: A `nss.models.backtest.build_predictions_frame`-shaped frame (must have
            `style_key`, `origin_week`, `y_true`, `weight`), restricted to exactly the origins being
            evaluated -- the SAME eval-set population (`y_true`/`weight`) real methods are scored
            against at each origin, so the floor is apples-to-apples with the real methods.
        origins: The origins to score, each tagged `has_52w_lag`/`is_covid` (propagated through to
            the output, same schema as `nss.models.backtest.run_backtest`'s per-origin table).
        seeds: RNG seeds. Defaults to `RANDOM_FLOOR_SEEDS` (20 seeds, 0..19).

    Returns:
        One row per `(origin_week, seed)`, columns: `origin_week`, `seed`, `has_52w_lag`,
        `is_covid`, `n_eval_set`, `n_eval`, plus one column per metric in
        `nss.models.metrics.METRIC_KEYS`.
    """
    rows: list[dict[str, object]] = []
    for origin in origins:
        origin_df = predictions.filter(pl.col("origin_week") == origin.origin_week)
        y_true = origin_df["y_true"].to_numpy()
        weight = origin_df["weight"].to_numpy()
        n_eval_set = origin_df.height
        for seed in seeds:
            y_pred = permute_predictions(y_true, seed)
            metrics = score_predictions(y_true, y_pred, weight)
            n_eval = metrics.pop("n_eval")
            rows.append(
                {
                    "origin_week": origin.origin_week,
                    "seed": seed,
                    "has_52w_lag": origin.has_52w_lag,
                    "is_covid": origin.is_covid,
                    "n_eval_set": n_eval_set,
                    "n_eval": int(n_eval),
                    **metrics,
                }
            )
    return pl.DataFrame(rows)


def aggregate_random_floor_over_seeds(per_origin_per_seed: pl.DataFrame) -> pl.DataFrame:
    """Collapse `score_random_floor_per_origin`'s per-(origin, seed) rows into one row per origin.

    Each metric is averaged across all seeds for that origin -- see module docstring SEEDS +
    AGGREGATION for why this precedes (rather than replaces) the per-origin block-bootstrap.

    Args:
        per_origin_per_seed: `score_random_floor_per_origin`'s output.

    Returns:
        One row per `origin_week`, same column schema as
        `nss.models.backtest.run_backtest`'s per-origin table (`origin_week`, `method` (always
        `RANDOM_FLOOR_METHOD`), `has_52w_lag`, `is_covid`, `n_eval_set`, `n_eval`, plus every
        `nss.models.metrics.METRIC_KEYS` metric, each the across-seed mean), directly concatenable
        with baseline/LightGBM per-origin tables for a shared `summarize_backtest` call.
    """
    agg_exprs = [pl.col(metric).mean().alias(metric) for metric in METRIC_KEYS]
    aggregated = (
        per_origin_per_seed.group_by("origin_week", maintain_order=True)
        .agg(
            pl.col("has_52w_lag").first(),
            pl.col("is_covid").first(),
            pl.col("n_eval_set").first(),
            pl.col("n_eval").first(),
            *agg_exprs,
        )
        .sort("origin_week")
    )
    return aggregated.with_columns(pl.lit(RANDOM_FLOOR_METHOD).alias("method")).select(
        "origin_week", "method", "has_52w_lag", "is_covid", "n_eval_set", "n_eval", *METRIC_KEYS
    )
