from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nss.features.model_features import _FEATURE_COLS
from nss.models.backtest import Origin
from nss.prod.calibration import calibration_window, conformal_q, fit_asymmetric
from nss.prod.config import load_config
from nss.prod.contracts import STYLE_KEY_COLS
from nss.prod.training import embargoed_train_weeks, psi_reference

CONFIG = "configs/champion.yaml"


def test_config_matches_the_locked_champion() -> None:
    """The config must carry exactly the hyperparameters the pre-registered analysis locked."""
    from nss.models.final_forecast import FINAL_MODEL_CONFIG

    m = load_config(CONFIG).model
    assert {k: getattr(m, k) for k in FINAL_MODEL_CONFIG} == FINAL_MODEL_CONFIG
    assert m.seed == 42 and m.objective == "regression"


def test_config_feature_columns_match_the_builder() -> None:
    assert load_config(CONFIG).features.columns == [*STYLE_KEY_COLS, *_FEATURE_COLS]


def test_config_hash_is_stable_and_sensitive() -> None:
    cfg = load_config(CONFIG)
    assert cfg.config_hash() == load_config(CONFIG).config_hash()
    changed = cfg.model_copy(update={"model": cfg.model.model_copy(update={"num_leaves": 31})})
    assert changed.config_hash() != cfg.config_hash()


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    text = Path(CONFIG).read_text(encoding="utf-8").replace("seed: 42", "seed: 42\n  typo_key: 1")
    p = tmp_path / "bad.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_embargo_uses_only_closed_label_windows() -> None:
    start = date(2019, 1, 7)
    grid = [Origin(start + timedelta(weeks=4 * i), True, False) for i in range(10)]
    weeks = embargoed_train_weeks(grid, 8, horizon_weeks=13)
    assert weeks  # some history is usable
    assert all(w + timedelta(weeks=13) <= grid[8].origin_week for w in weeks)
    assert grid[5].origin_week not in weeks  # 12 weeks before grid[8]: its window is still open


def test_conformal_quantile_is_the_kth_order_statistic() -> None:
    assert conformal_q(np.arange(1.0, 10.0), 0.8) == 8.0  # k = ceil(0.8 * 10) = 8
    assert conformal_q(np.array([1.0, 2.0]), 0.99) == 2.0  # k > n caps at the maximum
    with pytest.raises(ValueError):
        conformal_q(np.array([]), 0.9)


def test_calibration_window_and_fit() -> None:
    t = date(2020, 1, 6)
    history = [t - timedelta(weeks=k) for k in range(20, -1, -1)]
    win = calibration_window(t, history, window=4, horizon_weeks=13)
    assert win == [t - timedelta(weeks=k) for k in (16, 15, 14, 13)]
    rows = pl.DataFrame(
        {
            "origin_week": [win[0]] * 9 + [t] * 3,  # rows at t must be ignored
            "y_true": [0.0] * 12,
            "q10": list(np.arange(1.0, 10.0)) + [99.0] * 3,  # lower scores q10 - y = 1..9
            "q90": [0.0] * 12,
        }
    )
    q = fit_asymmetric(rows, win, alpha=0.2)
    assert q["n_scores"] == 9
    assert q["Q_lo"] == 9.0  # level 0.9: k = ceil(0.9 * 10) = 9 -> 9th smallest


def test_psi_reference_is_a_distribution() -> None:
    ref = psi_reference(np.random.default_rng(0).normal(size=1000))
    assert len(ref["edges"]) == 11
    assert sum(ref["proportions"]) == pytest.approx(1.0)
