from __future__ import annotations

import numpy as np

from nss.models.eval_power import ess_ac, ess_boot


def _ar1(n: int, phi: float, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()
    return x.tolist()


def test_iid_series_keeps_most_of_its_sample_size() -> None:
    x = np.random.default_rng(1).normal(size=400).tolist()
    assert ess_ac(x) > 0.6 * 400
    assert ess_boot(x, block_size=13) > 0.6 * 400


def test_strong_autocorrelation_shrinks_the_effective_sample_size() -> None:
    x = _ar1(400, 0.9)
    assert ess_ac(x) < 0.25 * 400
    assert ess_boot(x, block_size=13) < 0.5 * 400


def test_never_above_n_never_below_one_and_constant_series_is_n() -> None:
    assert ess_ac([1.0] * 20) == 20.0
    assert 1.0 <= ess_ac(_ar1(50, 0.99)) <= 50.0
    assert 1.0 <= ess_boot(_ar1(50, 0.99), 4) <= 50.0


def test_short_series_falls_back_to_n() -> None:
    assert ess_ac([1.0, 2.0]) == 2.0
    assert ess_boot([1.0, 2.0], 4) == 2.0
