from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from nss.models.selective_prediction import (
    OriginPicks,
    block_bootstrap_difference,
    permutation_floor,
    statistic,
    verdict,
)


def _picks(hits: list[list[int]], conf: list[list[int]]) -> list[OriginPicks]:
    return [
        OriginPicks(date(2020, 1, 6), np.array(h), np.array(c), 100)
        for h, c in zip(hits, conf, strict=True)
    ]


def test_statistic_is_confident_rate_minus_all_rate_and_coverage() -> None:
    picks = _picks([[1, 1, 0], [0, 1, 0]], [[1, 1, 0], [0, 0, 1]])
    rate_conf, rate_all, diff, coverage = statistic(picks)
    assert rate_conf == pytest.approx(2 / 3)  # confident picks: hit, hit, miss
    assert rate_all == pytest.approx(3 / 6)
    assert diff == pytest.approx(2 / 3 - 0.5)
    assert coverage == pytest.approx(3 / 6)


def test_no_confident_picks_gives_nan_difference_not_zero() -> None:
    _, rate_all, diff, coverage = statistic(_picks([[1, 0, 0]], [[0, 0, 0]]))
    assert np.isnan(diff) and rate_all == pytest.approx(1 / 3) and coverage == 0.0


def test_permutation_floor_is_centred_near_zero_for_uninformative_labels() -> None:
    rng = np.random.default_rng(3)
    hits = [list(rng.integers(0, 2, 3)) for _ in range(40)]
    conf = [list(rng.permutation([1, 1, 0])) for _ in range(40)]
    floor = permutation_floor(_picks(hits, conf), n_perm=400)
    assert abs(floor["null_mean"]) < 0.05
    assert floor["null_lo"] < 0 < floor["null_hi"]


def test_informative_confidence_gives_a_positive_interval_and_low_p() -> None:
    hits = [[1, 1, 0]] * 30
    conf = [[1, 1, 0]] * 30  # the confident picks are exactly the hits
    picks = _picks(hits, conf)
    lo, _ = block_bootstrap_difference(picks, block=4, n_resamples=300)
    assert lo > 0
    assert permutation_floor(picks, n_perm=300)["p_one_sided"] < 0.05


def test_verdict_needs_a_positive_lower_bound_and_enough_coverage() -> None:
    assert verdict(0.01, 0.5) == "HELPS"
    assert verdict(0.0, 0.5) == "NOT_DEMONSTRATED"  # lower bound at zero is not above zero
    assert verdict(0.2, 0.2) == "NOT_DEMONSTRATED"  # coverage below 0.30
