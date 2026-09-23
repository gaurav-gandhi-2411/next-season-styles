from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from nss.models.metrics import demand_capture_at_k, tolerance_hit_at_k
from nss.models.phase_a_measure import block_means, derive_margin, select_primary


def _log(intensity: list[float]) -> np.ndarray:
    """The project's target scale: log1p of raw intensity."""
    return np.log1p(np.asarray(intensity, dtype=float))


def test_demand_capture_hand_computed() -> None:
    """Intensities [10, 8, 5, 1]; predicted order idx0, idx3, idx2, idx1.

    k=2: picks {0, 3} -> 10 + 1 = 11; true top-2 -> 10 + 8 = 18; 11/18.
    """
    y = _log([10, 8, 5, 1])
    pred = np.array([4.0, 1.0, 2.0, 3.0])
    assert demand_capture_at_k(y, pred, 2) == pytest.approx(11 / 18)


def test_demand_capture_is_in_raw_intensity_not_log() -> None:
    """The ratio uses expm1(y); in log space it would be (log 11 + log 2) / (log 11 + log 9)."""
    y = _log([10, 8, 1])
    pred = np.array([3.0, 1.0, 2.0])
    assert demand_capture_at_k(y, pred, 2) == pytest.approx(11 / 18)


def test_demand_capture_perfect_and_clipped() -> None:
    y = _log([3, 2, 1])
    assert demand_capture_at_k(y, y, 3) == pytest.approx(1.0)
    assert demand_capture_at_k(y, y, 20) == pytest.approx(1.0)  # k clipped to n
    assert math.isnan(demand_capture_at_k([], [], 3))
    assert math.isnan(demand_capture_at_k([0.0, 0.0], [1.0, 2.0], 1))  # zero denominator


def test_tolerance_hit_counts_near_ties() -> None:
    """True #3 = 100. margin 0.05 -> threshold 95. Picks 100 (hit), 96 (hit), 90 (miss)."""
    y = _log([120, 110, 100, 96, 90])
    pred = np.array([0.0, 0.0, 3.0, 2.0, 1.0])
    assert tolerance_hit_at_k(y, pred, 3, 0.05) == pytest.approx(2 / 3)
    assert tolerance_hit_at_k(y, pred, 3, 0.0) == pytest.approx(1 / 3)


def test_tolerance_hit_rejects_bad_margin() -> None:
    with pytest.raises(ValueError):
        tolerance_hit_at_k([1.0], [1.0], 1, 1.0)
    with pytest.raises(ValueError):
        tolerance_hit_at_k([1.0], [1.0], 1, -0.1)


def test_derive_margin_is_the_90th_percentile() -> None:
    t = pl.DataFrame({"gap_intensity_frac": [float(i) for i in range(11)]})
    assert derive_margin(t) == pytest.approx(9.0)


def test_block_means_matches_block_bootstrap_ci() -> None:
    """Same RNG draws as backtest.block_bootstrap_ci, so the CIs agree exactly."""
    from nss.models.backtest import block_bootstrap_ci

    x = list(np.random.default_rng(0).normal(size=30))
    _, lo, hi = block_bootstrap_ci(x, block_size=13)
    blo, bhi = np.percentile(block_means(x, block=13), [2.5, 97.5])
    assert (lo, hi) == pytest.approx((blo, bhi), abs=1e-12)


def test_select_primary_uses_normalised_mde_and_eligibility() -> None:
    """capture@20 has the smallest raw MDE but a floor above 0.5, so it is ineligible (R2)."""
    metrics = [
        "tolerance_hit_at_3",
        "demand_capture_at_3",
        "hit_at_3_in_top20",
        "demand_capture_at_10",
        "demand_capture_at_20",
    ]
    mde = [0.30, 0.10, 0.40, 0.09, 0.01]
    floor = [0.01, 0.50, 0.01, 0.60, 0.70]
    power = pl.DataFrame(
        {
            "metric": metrics,
            "n_paired": [30] * 5,
            "se_block": [m / 2.8 for m in mde],
            "mde_80": mde,
        }
    )
    summary = pl.DataFrame({"method": ["random_floor"] * 5, "metric": metrics, "mean": floor})
    out = select_primary(power, summary)
    # normalised: 0.303, 0.200, 0.404, (0.225 but R2 fails), (R2 fails)
    assert out.filter(pl.col("chosen"))["metric"].to_list() == ["demand_capture_at_3"]
    assert not out.filter(pl.col("metric") == "demand_capture_at_20")["passes"][0]
