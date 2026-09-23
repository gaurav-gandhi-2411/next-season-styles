"""Rolling asymmetric conformalised quantile regression (PREREGISTRATION.md Section S).

For a forecast origin `t`, the correction is learned from the `window` most recent weekly origins
whose 13-week outcomes had closed by `t` (`c + horizon <= t`), all their rows pooled. The low and
high ends are corrected separately at level `1 - alpha / 2` each.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np
import polars as pl


def conformal_q(scores: np.ndarray, level: float) -> float:
    """The k-th smallest score, k = ceil(level * (n + 1)), capped at the maximum (k > n)."""
    n = scores.size
    if n == 0:
        raise ValueError("no calibration scores")
    k = min(n, int(np.ceil(level * (n + 1))))
    return float(np.sort(scores)[k - 1])


def calibration_window(
    t: date, history: Sequence[date], window: int, horizon_weeks: int
) -> list[date]:
    """The `window` most recent origins whose outcome window closed by `t`."""
    eligible = sorted(c for c in history if c + timedelta(weeks=horizon_weeks) <= t)
    return eligible[-window:]


def fit_asymmetric(
    scored: pl.DataFrame, window_weeks: Sequence[date], alpha: float
) -> dict[str, float]:
    """Q_lo and Q_hi from rows (y_true, q10, q90) at the calibration window's origins."""
    c = scored.filter(pl.col("origin_week").is_in(list(window_weeks)))
    y, lo, hi = (c[k].to_numpy() for k in ("y_true", "q10", "q90"))
    return {
        "Q_lo": conformal_q(lo - y, 1 - alpha / 2),
        "Q_hi": conformal_q(y - hi, 1 - alpha / 2),
        "n_scores": int(c.height),
    }
