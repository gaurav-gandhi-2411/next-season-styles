"""Circular block bootstrap (Politis and Romano 1992), pre-registered in PREREGISTRATION.md Q1.

The earlier moving-block bootstrap drew block starts from `0 .. n - L` without wrap-around, so the
first and last `L - 1` origins were under-sampled and intervals skewed. Here the series is a
circle: starts are uniform over `0 .. n - 1` and a block runs `s, s+1, ..., s+L-1 (mod n)`, so
every origin has the same expected weight. L, resample count and seed are unchanged (13, 2,000,
42), and the RNG call shape is `rng.integers(0, n, size=ceil(n / L))`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import norm

BLOCK = 13
N_RESAMPLES = 2000
SEED = 42
POWER_Z = float(norm.ppf(0.975) + norm.ppf(0.80))  # two-sided alpha 0.05, 80% power


def _clean(values: Sequence[float]) -> np.ndarray:
    return np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)


def circular_block_means(
    values: Sequence[float],
    block: int = BLOCK,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> np.ndarray:
    """Resample means of a chronological series under the circular block bootstrap."""
    x = _clean(values)
    n = x.size
    if n == 0:
        return np.full(n_resamples, np.nan)
    rng = np.random.default_rng(seed)
    blocks_needed = -(-n // block)
    offsets = np.arange(block)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        starts = rng.integers(0, n, size=blocks_needed)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        means[i] = x[idx].mean()
    return means


def circular_ci(values: Sequence[float], **kw: int) -> dict[str, float]:
    """Mean, 95% percentile interval, bootstrap SE and MDE@80% of a series' mean."""
    x = _clean(values)
    if x.size == 0:
        return {"n": 0}
    means = circular_block_means(x, **kw)
    lo, hi = np.percentile(means, [2.5, 97.5])
    se = float(np.std(means, ddof=1)) if x.size > 1 else 0.0
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "se": se,
        "mde_80": POWER_Z * se,
    }


def bootstrap_p_two_sided(values: Sequence[float], **kw: int) -> float:
    """Two-sided bootstrap p-value for "mean = 0", by percentile-interval duality.

    `p = min(1, 2 * min(#{m <= 0} + 1, #{m >= 0} + 1) / (B + 1))` over the resample means `m`. The
    two-sided `(1 - a)` percentile interval excludes zero exactly when `p <= a`, up to the +1
    correction, so a Holm threshold on `p` is a Holm threshold on the interval.
    """
    means = circular_block_means(values, **kw)
    b = means.size
    below = int(np.sum(means <= 0.0))
    above = int(np.sum(means >= 0.0))
    return float(min(1.0, 2.0 * (min(below, above) + 1) / (b + 1)))


def holm_thresholds(m: int, alpha: float = 0.05) -> list[float]:
    """Holm-Bonferroni step-down thresholds: the i-th smallest p-value is tested at alpha/(m-i)."""
    return [alpha / (m - i) for i in range(m)]


def holm_decisions(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm step-down: reject in ascending p order until the first failure, then stop."""
    order = sorted(pvalues, key=lambda k: pvalues[k])
    thresholds = holm_thresholds(len(order), alpha)
    out = dict.fromkeys(pvalues, False)
    for name, thr in zip(order, thresholds, strict=True):
        if pvalues[name] <= thr:
            out[name] = True
        else:
            break
    return out
