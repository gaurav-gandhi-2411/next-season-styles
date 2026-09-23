"""Ranking and regression metrics for scoring rolling-origin backtest predictions.

Every function here scores ONE origin's eval set against ONE method's predictions: a 1-D array of
realized targets (`y_true`), a 1-D array of that method's predictions (`y_pred`), and (for WMAPE
only) a 1-D array of per-row weights. `score_predictions` is the single reusable entry point -- it
takes those three arrays and returns all 7 metrics in one dict. A later model that plugs in a
LightGBM prediction column (or a random-permutation floor -- see `nss.models.random_floor`) calls
`score_predictions` exactly the same way the 4 baselines in `nss.models.backtest` do; nothing in
this module is baseline-specific.

HEADLINE METRIC (v2, see backtest_v2 report): Hit@3-in-top20 -- the fraction of the model's
predicted top-3 style_keys that land ANYWHERE within the TRUE top-20 (by realized target) -- is the
project's headline metric as of the v2 rebuild. Hit@3-in-top10 (true top-10) is its stricter
secondary companion. Precision@3/Precision@10 remain computed and reported for every origin and
method, but are now SECONDARY, not headline -- the v2 backtest report demoted them once the
random-floor check made clear Precision@3 in particular sits close to the random-guessing floor on
this project's ~3,000-style eval sets (see `nss.models.random_floor` module docstring and
`reports/tables/backtest_summary_v2.csv`'s `*_no_demonstrated_signal` columns), whereas Hit@k-in-
top-N's larger true-set denominator gives it a meaningfully higher, more separable floor.
`Hit@k-in-top-N = |predicted top-k (by predicted value) intersect actual top-N (by realized
target)| / k_eff` -- see `hit_at_k_in_top_n` for the exact `k_eff`/`n_eff` clipping rule.

`Precision@K = |predicted top-K styles (by predicted value) intersect actual top-K styles (by
realized target)| / K`.

NDCG@10: relevance is the realized target value itself (a continuous, non-negative log1p-intensity
quantity -- every target/prediction in this project is non-negative, so it is a valid NDCG
relevance score), styles are ranked by PREDICTED value, and DCG is restricted to the top-10
predicted styles (not the top-10 actual styles -- NDCG's whole point is to penalize a predicted
ranking that puts low-relevance styles where high-relevance ones belong).

Spearman rho: rank correlation between predicted and actual values across the FULL eval set for an
origin (unlike the top-K metrics above, this uses every style, not just the top of the ranking).

WMAPE: `sum(weight * |actual - pred|) / sum(weight * |actual|)`, weighted -- see
`nss.models.backtest.WMAPE_WEIGHT_COL` for the weighting-column choice and its rationale.

K-CLIPPING: if an eval set has fewer than K styles, `ndcg_at_k`, `precision_at_k`, and
`hit_at_k_in_top_n` all use `k_eff = min(k, n)` (and, for `hit_at_k_in_top_n`, `n_eff = min(n, n)`
independently) rather than raising or padding -- a top-K metric over fewer than K items is still
well-defined once K is clipped to the available population, and this project's eval sets are
expected to be far larger than 20 in practice (see backtest report), so clipping is a defensive
boundary case handled correctly by construction, not the common path.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import spearmanr

NDCG_DEFAULT_K = 10
PRECISION_DEFAULT_KS: tuple[int, ...] = (3, 10)

# Headline metric params (v2): predicted top-3 checked for membership in the true top-20 (primary)
# and true top-10 (secondary, stricter). See module docstring HEADLINE METRIC.
HIT_TOP_K = 3
HIT_TOP_N_PRIMARY = 20
HIT_TOP_N_SECONDARY = 10

# The 7 mandatory metrics, in report order (Hit@3-in-top20/top10 first -- see module docstring
# HEADLINE METRIC; Precision@3/10 now secondary but still always computed). Kept as a module-level
# constant so aggregation code (`nss.models.backtest`, `nss.models.random_floor`) can iterate over
# "every metric" without hardcoding the list a second time.
METRIC_KEYS: tuple[str, ...] = (
    f"hit_at_{HIT_TOP_K}_in_top{HIT_TOP_N_PRIMARY}",
    f"hit_at_{HIT_TOP_K}_in_top{HIT_TOP_N_SECONDARY}",
    "precision_at_3",
    "precision_at_10",
    f"ndcg_at_{NDCG_DEFAULT_K}",
    "spearman_rho",
    "wmape",
)


def ndcg_at_k(y_true: Sequence[float], y_pred: Sequence[float], k: int = NDCG_DEFAULT_K) -> float:
    """NDCG@k: true values as relevance, ranked by PREDICTED value, DCG restricted to top-k
    predicted.

    Args:
        y_true: Realized target values (relevance scores), one per style.
        y_pred: Predicted values, one per style, same order as `y_true`.
        k: Cutoff. Clipped to `len(y_true)` if the eval set is smaller (see module docstring).

    Returns:
        DCG@k (ranked by predicted value) / IDCG@k (ranked by true value). `nan` if the eval set is
        empty; `0.0` if IDCG is 0 (every true value in the eval set is 0, so no ranking can gain).
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    n = y_true_arr.shape[0]
    if n == 0:
        return float("nan")
    k_eff = min(k, n)

    discounts = 1.0 / np.log2(np.arange(2, k_eff + 2))

    predicted_order = np.argsort(-y_pred_arr, kind="stable")
    dcg = float(np.sum(y_true_arr[predicted_order][:k_eff] * discounts))

    ideal_order = np.argsort(-y_true_arr, kind="stable")
    idcg = float(np.sum(y_true_arr[ideal_order][:k_eff] * discounts))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def precision_at_k(y_true: Sequence[float], y_pred: Sequence[float], k: int) -> float:
    """Precision@k: `|predicted top-k (by y_pred) intersect actual top-k (by y_true)| / k_eff`.

    See module docstring HEADLINE METRIC for why this is the project's primary reported metric, and
    K-CLIPPING for the `k_eff = min(k, n)` boundary rule.

    Args:
        y_true: Realized target values, one per style.
        y_pred: Predicted values, one per style, same order as `y_true`.
        k: Cutoff (nominal K, e.g. 3 or 10).

    Returns:
        `nan` if the eval set is empty, else the intersection-over-k_eff ratio in `[0, 1]`.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    n = y_true_arr.shape[0]
    if n == 0:
        return float("nan")
    k_eff = min(k, n)

    predicted_top = set(np.argsort(-y_pred_arr, kind="stable")[:k_eff].tolist())
    actual_top = set(np.argsort(-y_true_arr, kind="stable")[:k_eff].tolist())
    return len(predicted_top & actual_top) / k_eff


def hit_at_k_in_top_n(y_true: Sequence[float], y_pred: Sequence[float], k: int, n: int) -> float:
    """Hit@k-in-top-N: fraction of the predicted top-k styles that land ANYWHERE in the true top-N.

    See module docstring HEADLINE METRIC. Unlike `precision_at_k` (predicted top-K vs. actual
    top-K, same K), this compares a SMALL predicted set (k, e.g. 3) against a LARGER true set (N,
    e.g. 20 or 10) -- "did the model's top-3 bets land anywhere in the true top-N", not "did the
    model's top-3 bets land in the EXACT true top-3". The denominator is `k_eff` (the number of
    predicted picks actually available to hit with), not `n_eff` -- this is a precision-style
    metric over the predicted set, not a recall-style metric over the true set.

    Args:
        y_true: Realized target values, one per style.
        y_pred: Predicted values, one per style, same order as `y_true`.
        k: Predicted-set cutoff (nominal K, e.g. 3).
        n: True-set cutoff (nominal N, e.g. 20 or 10).

    Returns:
        `nan` if the eval set is empty, else the intersection-over-`k_eff` ratio in `[0, 1]`.
        `k_eff = min(k, n_total)` and `n_eff = min(n, n_total)` are clipped independently (see
        module docstring K-CLIPPING).
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    n_total = y_true_arr.shape[0]
    if n_total == 0:
        return float("nan")
    k_eff = min(k, n_total)
    n_eff = min(n, n_total)

    predicted_top = set(np.argsort(-y_pred_arr, kind="stable")[:k_eff].tolist())
    actual_top = set(np.argsort(-y_true_arr, kind="stable")[:n_eff].tolist())
    return len(predicted_top & actual_top) / k_eff


def spearman_rho(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Spearman rank correlation between `y_true` and `y_pred` across the full eval set.

    Returns:
        `nan` if fewer than 2 styles are in the eval set (rank correlation is undefined), else the
        Spearman rho in `[-1, 1]`.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if y_true_arr.shape[0] < 2:
        return float("nan")
    rho, _p_value = spearmanr(y_true_arr, y_pred_arr)
    return float(rho)


def wmape(y_true: Sequence[float], y_pred: Sequence[float], weight: Sequence[float]) -> float:
    """Weighted MAPE: `sum(weight * |actual - pred|) / sum(weight * |actual|)`.

    See `nss.models.backtest.WMAPE_WEIGHT_COL` for the weighting-column choice.

    Returns:
        `nan` if the eval set is empty or the weighted-actual denominator is exactly 0 (every
        weighted actual value is 0 -- division would be undefined, not "perfect prediction").
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    weight_arr = np.asarray(weight, dtype=float)
    if y_true_arr.shape[0] == 0:
        return float("nan")

    denom = float(np.sum(weight_arr * np.abs(y_true_arr)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(weight_arr * np.abs(y_true_arr - y_pred_arr)) / denom)


def _intensity(y_log1p: np.ndarray) -> np.ndarray:
    """Units per active article per week, from the project's `log1p` target scale."""
    return np.expm1(y_log1p)


def demand_capture_at_k(y_true: Sequence[float], y_pred: Sequence[float], k: int) -> float:
    """Demand capture@k: realised intensity of the predicted top-k over that of the true top-k.

    Pre-registered in `reports/v3/PREREGISTRATION.md` (Phase A). Both targets are on the `log1p`
    scale and are converted back to raw intensity (`expm1`) first, so the ratio is in demand units,
    not log units. `k_eff = min(k, n)`, as in the other top-k metrics.

    Returns:
        `nan` if the eval set is empty or the true top-k intensity sums to 0, else a ratio in [0, 1]
        (1 exactly when the predicted top-k has the same realised intensity as the true top-k).
    """
    true_int = _intensity(np.asarray(y_true, dtype=float))
    y_pred_arr = np.asarray(y_pred, dtype=float)
    n = true_int.shape[0]
    if n == 0:
        return float("nan")
    k_eff = min(k, n)
    picked = float(true_int[np.argsort(-y_pred_arr, kind="stable")[:k_eff]].sum())
    best = float(np.sort(true_int)[::-1][:k_eff].sum())
    return picked / best if best > 0.0 else float("nan")


def tolerance_hit_at_k(
    y_true: Sequence[float], y_pred: Sequence[float], k: int, margin: float
) -> float:
    """Tolerance hit@k: fraction of the predicted top-k whose realised intensity is at least
    `(1 - margin)` times the true #k style's intensity.

    Pre-registered in `reports/v3/PREREGISTRATION.md` (Phase A), with `margin` derived there from
    the measured near-tie distribution. With `margin = 0` it reduces to Precision@k up to ties.

    Returns:
        `nan` if the eval set is empty, else the hit fraction over `k_eff = min(k, n)` picks.
    """
    if not 0.0 <= margin < 1.0:
        raise ValueError("margin must be in [0, 1)")
    true_int = _intensity(np.asarray(y_true, dtype=float))
    y_pred_arr = np.asarray(y_pred, dtype=float)
    n = true_int.shape[0]
    if n == 0:
        return float("nan")
    k_eff = min(k, n)
    threshold = (1.0 - margin) * float(np.sort(true_int)[::-1][k_eff - 1])
    picks = true_int[np.argsort(-y_pred_arr, kind="stable")[:k_eff]]
    return float(np.mean(picks >= threshold))


def score_predictions(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    weight: Sequence[float] | None = None,
    ndcg_k: int = NDCG_DEFAULT_K,
    precision_ks: tuple[int, ...] = PRECISION_DEFAULT_KS,
) -> dict[str, float]:
    """Score one origin's eval set against one method's predictions on all 5 mandatory metrics.

    The single reusable scoring entry point (see module docstring) -- callers (the 4 baselines in
    `nss.models.backtest`, and any later model such as LightGBM) all go through this function with
    their own `y_pred` array; nothing about scoring is baseline-specific.

    Args:
        y_true: Realized target values, one per style.
        y_pred: Predicted values, one per style, same order as `y_true`.
        weight: Per-style WMAPE weight. Defaults to uniform (`1.0` for every style) if omitted.
        ndcg_k: NDCG cutoff. Defaults to `NDCG_DEFAULT_K` (10).
        precision_ks: Precision cutoffs to compute. Defaults to `PRECISION_DEFAULT_KS` ((3, 10)).

    Returns:
        A dict with keys `n_eval` (the number of styles actually scored) plus one key per metric in
        `METRIC_KEYS` (`hit_at_3_in_top20`, `hit_at_3_in_top10`, `precision_at_3`,
        `precision_at_10`, `ndcg_at_10`, `spearman_rho`, `wmape`).
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    weight_arr = np.ones_like(y_true_arr) if weight is None else np.asarray(weight, dtype=float)

    result: dict[str, float] = {"n_eval": float(y_true_arr.shape[0])}
    result[f"hit_at_{HIT_TOP_K}_in_top{HIT_TOP_N_PRIMARY}"] = hit_at_k_in_top_n(
        y_true_arr, y_pred_arr, k=HIT_TOP_K, n=HIT_TOP_N_PRIMARY
    )
    result[f"hit_at_{HIT_TOP_K}_in_top{HIT_TOP_N_SECONDARY}"] = hit_at_k_in_top_n(
        y_true_arr, y_pred_arr, k=HIT_TOP_K, n=HIT_TOP_N_SECONDARY
    )
    for k in precision_ks:
        result[f"precision_at_{k}"] = precision_at_k(y_true_arr, y_pred_arr, k)
    result[f"ndcg_at_{ndcg_k}"] = ndcg_at_k(y_true_arr, y_pred_arr, ndcg_k)
    result["spearman_rho"] = spearman_rho(y_true_arr, y_pred_arr)
    result["wmape"] = wmape(y_true_arr, y_pred_arr, weight_arr)
    return result
