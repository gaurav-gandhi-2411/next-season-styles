"""Monitoring logic: scoring log, rolling realised coverage, PSI drift. Logic only; C2 wires it.

Thresholds are fixed here, before any live data:

- **Rolling realised coverage.** Once a forecast week's 13-week outcome is known, its coverage is
  the share of styles whose outcome fell inside the interval. The alert looks at the **4 most
  recent forecast weeks with known outcomes**, pooled (about 12,000 predictions), and fires when
  that coverage leaves **[0.70, 0.90]**. Four weeks matches the calibration window. The band is
  wider than the model's 0.75-0.85 target because per-week coverage is known to swing
  (backtest 0.58-0.98, driven by the calibration lag), and the alert is meant for sustained
  misses. How often it would have fired in the backtest is reported, not assumed
  (`scripts/monitoring_alert_backtest.py`).
- **Drift (PSI)** of every numeric feature and of the point prediction, against the training
  distribution stored in the artifact (`reference.json`, decile bins): **PSI < 0.10 stable,
  0.10-0.25 warn, >= 0.25 alert**, the usual convention for PSI.
"""

from __future__ import annotations

import io
import uuid
from datetime import UTC, date, datetime
from typing import Any

import fsspec
import numpy as np
import polars as pl

from nss.features.targets import HORIZON_WEEKS, compute_forward_target

COVERAGE_WINDOW_WEEKS = 4
COVERAGE_BAND = (0.70, 0.90)
PSI_WARN = 0.10
PSI_ALERT = 0.25
PSI_EPS = 1e-4  # floor on bin shares, so an empty bin does not make PSI infinite

LOG_COLUMNS = (
    "style_key",
    "as_of",
    "forecast",
    "interval_lower",
    "interval_upper",
    "rank",
    "model_version",
)


class ScoringLog:
    """Every prediction, with its model version. One immutable file per scoring batch."""

    def __init__(self, root: str) -> None:
        self.fs, self.base = fsspec.core.url_to_fs(root)
        self.base = self.base.rstrip("/")

    def append(self, scored: pl.DataFrame) -> str:
        now = datetime.now(UTC)
        rows = scored.select(LOG_COLUMNS).with_columns(pl.lit(now).alias("logged_utc"))
        self.fs.makedirs(self.base, exist_ok=True)
        path = f"{self.base}/scores_{now:%Y%m%dT%H%M%S%fZ}_{uuid.uuid4().hex[:8]}.parquet"
        buf = io.BytesIO()
        rows.write_parquet(buf)
        with self.fs.open(path, "wb") as f:
            f.write(buf.getvalue())
        return path

    def read(self) -> pl.DataFrame:
        if not self.fs.exists(self.base):
            return pl.DataFrame()
        files = sorted(p for p in self.fs.ls(self.base, detail=False) if p.endswith(".parquet"))
        parts = []
        for p in files:
            with self.fs.open(p, "rb") as f:
                parts.append(pl.read_parquet(io.BytesIO(f.read())))
        return pl.concat(parts) if parts else pl.DataFrame()


# ---------------------------------------------------------------------------
# realised coverage
# ---------------------------------------------------------------------------


def realised_outcomes(panel: pl.DataFrame, as_of_weeks: list[date]) -> pl.DataFrame:
    """Realised 13-week intensity per (style, as_of), for weeks whose window has fully closed."""
    parts = [compute_forward_target(panel, w, horizon_weeks=HORIZON_WEEKS) for w in as_of_weeks]
    return (
        pl.concat(parts)
        .filter(pl.col("target").is_not_null())
        .select("style_key", pl.col("origin_week").alias("as_of"), pl.col("target").exp() - 1)
        .rename({"target": "realised"})
    )


def coverage_by_week(log: pl.DataFrame, realised: pl.DataFrame) -> pl.DataFrame:
    """Per forecast week with known outcomes: n and the share inside the logged interval."""
    j = log.join(realised, on=["style_key", "as_of"], how="inner")
    return (
        j.with_columns(
            (
                (pl.col("interval_lower") <= pl.col("realised"))
                & (pl.col("realised") <= pl.col("interval_upper"))
            ).alias("covered")
        )
        .group_by("as_of")
        .agg(pl.len().alias("n"), pl.col("covered").sum().alias("n_covered"))
        .sort("as_of")
    )


def coverage_alert(
    per_week: pl.DataFrame,
    window: int = COVERAGE_WINDOW_WEEKS,
    band: tuple[float, float] = COVERAGE_BAND,
) -> dict[str, Any]:
    """Pooled coverage over the `window` most recent weeks; alert when it leaves `band`."""
    recent = per_week.sort("as_of").tail(window)
    if recent.height < window:
        return {"alert": False, "status": "insufficient", "weeks": recent.height, "coverage": None}
    cov = float(recent["n_covered"].sum() / recent["n"].sum())
    status = "low" if cov < band[0] else "high" if cov > band[1] else "ok"
    return {
        "alert": status != "ok",
        "status": status,
        "coverage": cov,
        "weeks": [str(w) for w in recent["as_of"].to_list()],
        "n": int(recent["n"].sum()),
        "band": list(band),
    }


# ---------------------------------------------------------------------------
# drift
# ---------------------------------------------------------------------------


def psi(reference: dict[str, list[float]], values: np.ndarray) -> float:
    """Population stability index of `values` against a stored reference (edges, proportions)."""
    edges = np.asarray(reference["edges"], dtype=float)
    expected = np.asarray(reference["proportions"], dtype=float)
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0 or edges.size < 2:
        return float("nan")
    actual = np.histogram(np.clip(v, edges[0], edges[-1]), bins=edges)[0] / v.size
    e = np.maximum(expected, PSI_EPS)
    a = np.maximum(actual, PSI_EPS)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_status(value: float) -> str:
    if np.isnan(value):
        return "no_data"
    return "alert" if value >= PSI_ALERT else "warn" if value >= PSI_WARN else "stable"


def drift_report(
    reference: dict[str, Any], features: pl.DataFrame, point_log1p: np.ndarray
) -> pl.DataFrame:
    """PSI and status for every referenced feature and for the point prediction."""
    rows = []
    for name, ref in reference["features"].items():
        if name in features.columns:
            value = psi(ref, features[name].cast(pl.Float64).to_numpy())
            rows.append(
                {"name": name, "kind": "feature", "psi": value, "status": psi_status(value)}
            )
    value = psi(reference["prediction"], point_log1p)
    rows.append(
        {
            "name": "point_prediction",
            "kind": "prediction",
            "psi": value,
            "status": psi_status(value),
        }
    )
    return pl.DataFrame(rows)
