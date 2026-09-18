"""Cross-process determinism regression test (D3a).

Reproduces, in miniature, the exact scenario the original bug report described: the SAME
already-trained-model-shaped inference (`build_model_frame` -> `train_lightgbm` ->
`predict_lightgbm`, the same functions `nss.models.final_forecast` calls) run twice as two
genuinely separate OS processes, on the identical data and hyperparameters, must produce
bit-identical predictions. This is a REPRODUCIBILITY test, not a model-quality test -- it makes no
claim about accuracy, only that repeated invocations agree with each other exactly.

Root cause (see `nss.features.model_features`'s DETERMINISM (A5 FOLLOW-UP) docstring section for
the full mechanism and the minimal repro that confirmed it): `pl.Categorical`'s category->code
dictionary was built via an internal, non-deterministic-across-processes unique-value collection,
so the SAME category string could get a DIFFERENT integer code in two separate process runs on the
identical input -- flipping which side of a LightGBM categorical split a row falls on. Fixed via
`pl.Enum` with an explicit, alphabetically-sorted vocabulary (a pure function of the sorted
category list, no hashing/dictionary-building involved).

A single Python process cannot exercise this bug at all (calling the pipeline twice within one
process reuses whatever the process's own internal state already settled on) -- hence the
`subprocess.run` design here, invoking `tests/_cross_process_determinism_worker.py` as two
independent `python` processes via `sys.executable`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_WORKER_PATH = Path(__file__).parent / "_cross_process_determinism_worker.py"


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


def test_predictions_are_bit_identical_across_separate_process_invocations() -> None:
    """Two genuinely separate `python` process runs of the same train+predict pipeline, on the
    same synthetic data and hyperparameters, must produce EXACTLY the same predictions for every
    (style_key, origin_week) row -- `np.array_equal`, not `np.allclose` (same bar as
    `tests/test_lightgbm_model.py::test_train_lightgbm_is_bit_identical_across_repeated_runs`,
    extended here across process boundaries rather than within one process).
    """
    run1 = _run_worker()
    run2 = _run_worker()

    assert len(run1) > 0, "worker produced no rows -- fixture or schedule is degenerate"
    assert len(run1) == len(run2)

    # Compare as a (style_key, origin_week) -> prediction mapping rather than assuming identical
    # row ORDER between the two subprocess runs (row order downstream of a join is a separate
    # concern from category-code/value determinism, which is what this test targets -- see
    # nss.models.lightgbm_model.build_model_frame's maintain_order="left" for the row-order
    # guarantee, verified as a byproduct below since a row-order mismatch would also fail the key
    # `set` equality check).
    keys1 = [(r["style_key"], r["origin_week"]) for r in run1]
    keys2 = [(r["style_key"], r["origin_week"]) for r in run2]
    assert set(keys1) == set(keys2)

    def _sort_key(r: dict[str, object]) -> tuple[object, object]:
        return (r["style_key"], r["origin_week"])

    preds1 = np.array([r["prediction"] for r in sorted(run1, key=_sort_key)])
    preds2 = np.array([r["prediction"] for r in sorted(run2, key=_sort_key)])

    assert np.array_equal(preds1, preds2), (
        f"predictions differ across separate process runs: "
        f"max abs diff = {np.abs(preds1 - preds2).max()}"
    )


def test_row_order_is_also_identical_across_separate_process_invocations() -> None:
    """Defense-in-depth: the ROW ORDER of the worker's output (not just the values) is identical
    across separate process runs too -- see the explicit sorts added to
    `nss.features.style_panel` and `nss.features.model_features.build_features`, and
    `maintain_order="left"` on `nss.models.lightgbm_model.build_model_frame`'s target join.
    """
    run1 = _run_worker()
    run2 = _run_worker()

    keys1 = [(r["style_key"], r["origin_week"]) for r in run1]
    keys2 = [(r["style_key"], r["origin_week"]) for r in run2]
    assert keys1 == keys2
