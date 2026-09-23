"""Production path for the champion forecaster (SPEC Phase C1).

Config-driven training to versioned artifacts, data contracts, a file-based registry with gated
promotion, batch and API inference, and monitoring logic. Imports only the light stack (polars,
numpy, scipy, lightgbm, pydantic, pandera, fastapi, fsspec): no torch, shap or plotting.
"""

from __future__ import annotations

INTERVAL_SEMANTICS = (
    "Long-run 80% interval: over many forecast weeks about 80% of outcomes fall inside it "
    "(backtest pooled coverage 0.803; 0.748 outside the COVID period). Any single week can be "
    "far off (backtest per-week coverage 0.58-0.98), because the calibration learns from "
    "outcomes 13-16 weeks old."
)
