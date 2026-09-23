from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from nss.models.phase_s_calibration import WINDOW, conformal_q, window_for


def test_conformal_quantile_is_finite_sample_corrected() -> None:
    """n = 9 scores 1..9, level 0.8: ceil(0.8 * 10) / 9 = 8/9 -> the 8th smallest score, 8."""
    assert conformal_q(np.arange(1.0, 10.0), 0.8) == 8.0


def test_conformal_quantile_caps_at_max() -> None:
    """Too few scores for the level: the correction exceeds 1 and falls back to the maximum."""
    assert conformal_q(np.array([1.0, 2.0, 3.0]), 0.9) == 3.0


def test_window_is_embargoed_and_most_recent() -> None:
    """Only origins whose 13-week outcome closed by t; the WINDOW latest of them."""
    t = date(2020, 1, 6)
    history = [t - timedelta(weeks=k) for k in range(30, -1, -1)]
    win = window_for(t, history)
    assert len(win) == WINDOW
    assert max(win) == t - timedelta(weeks=13)
    assert min(win) == t - timedelta(weeks=13 + WINDOW - 1)
    assert all(c + timedelta(weeks=13) <= t for c in win)
