"""Cross-process determinism regression test for `nss.models.lambdarank_model` (Task G2).

Same rationale/mechanism as `tests/test_determinism_cross_process.py` (the L2 model's own
cross-process test): a single Python process cannot exercise the cross-process class of
nondeterminism at all (see that module's docstring for the root cause -- `pl.Categorical`'s
category->code dictionary construction, fixed via `pl.Enum` in `nss.features.model_features`),
hence `subprocess.run`, invoking `tests/_cross_process_determinism_worker_lambdarank.py` as two
independent `python` processes via `sys.executable`. `nss.models.lambdarank_model.train_lambdarank`
reuses `nss.models.lightgbm_model.LGBM_DETERMINISM_PARAMS` verbatim (see that module's own
docstring DETERMINISM section) -- this test verifies that guarantee holds for `lgb.LGBMRanker` too,
not just `lgb.LGBMRegressor`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_WORKER_PATH = Path(__file__).parent / "_cross_process_determinism_worker_lambdarank.py"


def _run_worker() -> list[dict[str, object]]:
    """Run the worker script as a fresh OS process; parse its stdout JSON."""
    result = subprocess.run(
        [sys.executable, str(_WORKER_PATH)],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    records: list[dict[str, object]] = json.loads(result.stdout)
    return records


def test_lambdarank_predictions_are_bit_identical_across_separate_process_invocations() -> None:
    """Two genuinely separate `python` process runs of the same `lambdarank` train+predict
    pipeline, on the same synthetic data and hyperparameters, must produce EXACTLY the same
    predictions for every (style_key, origin_week) row -- `np.array_equal`, not `np.allclose`, the
    same bar as the L2 model's own cross-process test."""
    run1 = _run_worker()
    run2 = _run_worker()

    assert len(run1) > 0, "worker produced no rows -- fixture or schedule is degenerate"
    assert len(run1) == len(run2)

    keys1 = [(r["style_key"], r["origin_week"]) for r in run1]
    keys2 = [(r["style_key"], r["origin_week"]) for r in run2]
    assert set(keys1) == set(keys2)

    def _sort_key(r: dict[str, object]) -> tuple[object, object]:
        return (r["style_key"], r["origin_week"])

    preds1 = np.array([r["prediction"] for r in sorted(run1, key=_sort_key)])
    preds2 = np.array([r["prediction"] for r in sorted(run2, key=_sort_key)])

    assert np.array_equal(preds1, preds2), (
        f"lambdarank predictions differ across separate process runs: "
        f"max abs diff = {np.abs(preds1 - preds2).max()}"
    )
