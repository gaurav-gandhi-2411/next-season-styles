"""Production path for the champion forecaster (SPEC Phase C1).

Config-driven training to versioned artifacts, data contracts, a file-based registry with gated
promotion, batch and API inference, and monitoring logic. Imports only the light stack (polars,
numpy, scipy, lightgbm, pydantic, pandera, fastapi, fsspec): no torch, shap or plotting.
"""

from __future__ import annotations

# lightgbm must be imported before pyarrow. With lightgbm 4.7.0 and pyarrow 18.1.0 on Windows,
# importing pyarrow first (pandera and pandas both load it) makes LightGBM's native
# Dataset.set_field('label') fail with "access violation reading 0x0" on a plain float64 numpy
# label. Reproduced outside pytest; the native mechanism was not identified. Importing it here
# makes every nss.prod entry point safe; tests/test_prod_import_order.py guards the order.
import lightgbm  # noqa: F401  # isort: skip

INTERVAL_SEMANTICS = (
    "Long-run 80% interval: over many forecast weeks about 80% of outcomes fall inside it "
    "(backtest pooled coverage 0.803; 0.748 outside the COVID period). Any single week can be "
    "far off (backtest per-week coverage 0.58-0.98), because the calibration learns from "
    "outcomes 13-16 weeks old."
)
