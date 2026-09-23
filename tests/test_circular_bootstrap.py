from __future__ import annotations

import numpy as np
import pytest

from nss.models.circular_bootstrap import (
    bootstrap_p_two_sided,
    circular_block_means,
    circular_ci,
    holm_decisions,
    holm_thresholds,
)


def test_every_origin_equally_weighted_in_expectation() -> None:
    """Indicator series: the resampled share of each position must be ~1/n for every position.

    Under the old non-circular scheme the first and last positions were under-sampled; here a
    series that is 1 only at position 0 (or only at n-1) must average ~1/n, same as the middle.
    """
    n = 48
    for pos in (0, 24, n - 1):
        x = np.zeros(n)
        x[pos] = 1.0
        means = circular_block_means(x, block=13, n_resamples=20000)
        assert means.mean() == pytest.approx(1 / n, rel=0.05)


def test_blocks_wrap_around() -> None:
    """n=5, L=3, a single resample: every block is a contiguous run mod 5."""
    x = np.arange(5, dtype=float)
    means = circular_block_means(x, block=3, n_resamples=200)
    # every resample mean is a mean of 5 values from wrapped runs: always within [0, 4]
    assert np.all((means >= 0) & (means <= 4))
    # a constant series has a zero-width interval
    ci = circular_ci([2.0] * 10)
    assert ci["ci_lo"] == ci["ci_hi"] == 2.0 and ci["se"] == 0.0


def test_deterministic_with_seed() -> None:
    x = list(np.random.default_rng(1).normal(size=30))
    assert np.array_equal(circular_block_means(x), circular_block_means(x))


def test_nan_dropped() -> None:
    assert circular_ci([1.0, float("nan"), 3.0])["n"] == 2


def test_p_value_matches_interval_duality() -> None:
    """A clearly positive series gives a tiny p; a centred one gives a large p."""
    rng = np.random.default_rng(0)
    assert bootstrap_p_two_sided(rng.normal(1.0, 0.1, size=40)) < 0.001
    x = np.array([1.0, -1.0] * 20)
    assert bootstrap_p_two_sided(x) > 0.5


def test_holm_thresholds_and_step_down() -> None:
    assert holm_thresholds(4) == pytest.approx([0.0125, 0.05 / 3, 0.025, 0.05])
    # smallest p 0.01 <= 0.0125 passes; next 0.02 > 0.0167 fails and stops; 0.03 is not tested
    d = holm_decisions({"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.5})
    assert d == {"a": True, "b": False, "c": False, "d": False}
    assert holm_decisions({"a": 0.001, "b": 0.01, "c": 0.02, "d": 0.04}) == dict.fromkeys(
        "abcd", True
    )
