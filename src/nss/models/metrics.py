"""Ranking and regression metrics for scoring rolling-origin backtest predictions.

Every function here scores ONE origin's eval set against ONE method's predictions: a 1-D array of
realized targets (`y_true`), a 1-D array of that method's predictions (`y_pred`), and (for WMAPE
only) a 1-D array of per-row weights. `score_predictions` is the single reusable entry point -- it
takes those three arrays and returns all 5 metrics in one dict. A later task that plugs in a
LightGBM prediction column calls `score_predictions` exactly the same way the 4 baselines in
`nss.models.backtest` do; nothing in this module is baseline-specific.

HEADLINE METRIC: Precision@3 (with Precision@10 as its ranks-4-10 companion) is the project's
headline metric -- it directly matches the deliverable framing ("pick a top-3 bet list, ranks 4-10
as a secondary watchlist"), so it is called out first when reporting, even though all 5 metrics are
computed for every origin and every method. `Precision@K = |predicted top-K styles (by predicted
value) intersect actual top-K styles (by realized target)| / K`.

NDCG@10: relevance is the realized target value itself (a continuous, non-negative log1p-intensity
quantity -- every target/prediction in this project is non-negative, so it is a valid NDCG
relevance score), styles are ranked by PREDICTED value, and DCG is restricted to the top-10
predicted styles (not the top-10 actual styles -- NDCG's whole point is to penalize a predicted
ranking that puts low-relevance styles where high-relevance ones belong).

Spearman rho: rank correlation between predicted and actual values across the FULL eval set for an
origin (unlike the top-K metrics above, this uses every style, not just the top of the ranking).

WMAPE: `sum(weight * |actual - pred|) / sum(weight * |actual|)`, weighted -- see
`nss.models.backtest.WMAPE_WEIGHT_COL` for the weighting-column choice and its rationale.

K-CLIPPING: if an eval set has fewer than K styles, both `ndcg_at_k` and `precision_at_k` use
`k_eff = min(k, n)` rather than raising or padding -- a top-K metric over fewer than K items is
still well-defined once K is clipped to the available population, and this project's eval sets are
expected to be far larger than 10 in practice (see backtest report), so clipping is a defensive
boundary case handled correctly by construction, not the common path.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import spearmanr

NDCG_DEFAULT_K = 10
PRECISION_DEFAULT_KS: tuple[int, ...] = (3, 10)

# The 5 mandatory metrics, in report order (Precision@3 first -- see module docstring HEADLINE
# METRIC). Kept as a module-level constant so aggregation code (`nss.models.backtest`) can iterate
# over "every metric" without hardcoding the list a second time.
METRIC_KEYS: tuple[str, ...] = (
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
        `METRIC_KEYS` (`precision_at_3`, `precision_at_10`, `ndcg_at_10`, `spearman_rho`, `wmape`).
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    weight_arr = np.ones_like(y_true_arr) if weight is None else np.asarray(weight, dtype=float)

    result: dict[str, float] = {"n_eval": float(y_true_arr.shape[0])}
    for k in precision_ks:
        result[f"precision_at_{k}"] = precision_at_k(y_true_arr, y_pred_arr, k)
    result[f"ndcg_at_{ndcg_k}"] = ndcg_at_k(y_true_arr, y_pred_arr, ndcg_k)
    result["spearman_rho"] = spearman_rho(y_true_arr, y_pred_arr)
    result["wmape"] = wmape(y_true_arr, y_pred_arr, weight_arr)
    return result
