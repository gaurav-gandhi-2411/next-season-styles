from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from nss.prod.monitoring import (
    PSI_ALERT,
    PSI_WARN,
    ScoringLog,
    coverage_alert,
    coverage_by_week,
    drift_report,
    psi,
    psi_status,
)
from nss.prod.training import psi_reference

RNG = np.random.default_rng(42)
REF = psi_reference(RNG.normal(0, 1, 20_000))


def test_psi_stable_on_same_distribution() -> None:
    value = psi(REF, RNG.normal(0, 1, 5_000))
    assert value < PSI_WARN and psi_status(value) == "stable"


def test_psi_alerts_on_a_shifted_distribution() -> None:
    value = psi(REF, RNG.normal(1.0, 1, 5_000))  # one-SD mean shift
    assert value >= PSI_ALERT and psi_status(value) == "alert"


def test_psi_alerts_on_a_collapsed_distribution() -> None:
    value = psi(REF, np.full(5_000, 0.0))  # every value in one bin
    assert psi_status(value) == "alert"


def test_psi_no_data_is_explicit() -> None:
    assert psi_status(psi(REF, np.array([np.nan]))) == "no_data"


def test_drift_report_flags_only_the_drifted_feature() -> None:
    ref = {"features": {"stable": REF, "drifted": REF}, "prediction": REF}
    feats = pl.DataFrame({"stable": RNG.normal(0, 1, 5_000), "drifted": RNG.normal(2, 1, 5_000)})
    rep = drift_report(ref, feats, RNG.normal(0, 1, 5_000))
    status = dict(zip(rep["name"], rep["status"], strict=True))
    assert status == {"stable": "stable", "drifted": "alert", "point_prediction": "stable"}


def _log_and_outcomes(
    coverage_per_week: list[float], n: int = 3000
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Synthetic scoring log with interval [0, 1]; `coverage` of each week's outcomes inside."""
    logs, outs = [], []
    for i, cov in enumerate(coverage_per_week):
        w = date(2020, 1, 6) + timedelta(weeks=i)
        keys = [f"s{j}" for j in range(n)]
        inside = RNG.random(n) < cov
        logs.append(pl.DataFrame({"style_key": keys, "as_of": [w] * n, "interval_lower": 0.0,
                                  "interval_upper": 1.0}))  # fmt: skip
        outs.append(pl.DataFrame({"style_key": keys, "as_of": [w] * n,
                                  "realised": np.where(inside, 0.5, 2.0)}))  # fmt: skip
    return pl.concat(logs), pl.concat(outs)


def test_coverage_in_band_does_not_alert() -> None:
    log, real = _log_and_outcomes([0.80] * 6)
    res = coverage_alert(coverage_by_week(log, real))
    assert res["alert"] is False and res["status"] == "ok"
    assert abs(res["coverage"] - 0.80) < 0.02


def test_under_coverage_alerts_low() -> None:
    log, real = _log_and_outcomes([0.80, 0.80, 0.60, 0.62, 0.58, 0.61])
    res = coverage_alert(coverage_by_week(log, real))
    assert res["alert"] is True and res["status"] == "low"


def test_over_coverage_alerts_high() -> None:
    log, real = _log_and_outcomes([0.97] * 4)
    assert coverage_alert(coverage_by_week(log, real))["status"] == "high"


def test_alert_uses_only_the_most_recent_window() -> None:
    """Old bad weeks followed by four good ones: no alert."""
    log, real = _log_and_outcomes([0.50, 0.50, 0.80, 0.80, 0.80, 0.80])
    assert coverage_alert(coverage_by_week(log, real))["alert"] is False


def test_too_few_weeks_is_not_an_alert() -> None:
    log, real = _log_and_outcomes([0.40, 0.40])
    res = coverage_alert(coverage_by_week(log, real))
    assert res["alert"] is False and res["status"] == "insufficient"


def test_scoring_log_roundtrip(tmp_path: Path) -> None:
    scored = pl.DataFrame({"style_key": ["a", "b"], "as_of": [date(2020, 1, 6)] * 2,
                           "forecast": [1.0, 2.0], "interval_lower": [0.5, 1.0],
                           "interval_upper": [2.0, 3.0], "rank": [2, 1],
                           "model_version": ["v1", "v1"], "extra": [0, 0]})  # fmt: skip
    log = ScoringLog(str(tmp_path / "log"))
    log.append(scored)
    log.append(scored.with_columns(pl.lit("v2").alias("model_version")))
    got = log.read()
    assert got.height == 4 and sorted(got["model_version"].unique()) == ["v1", "v2"]
    assert "extra" not in got.columns and "logged_utc" in got.columns
