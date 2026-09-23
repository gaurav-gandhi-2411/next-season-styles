from __future__ import annotations

import subprocess
import sys

import pytest

# See src/nss/prod/__init__.py: pyarrow loaded before lightgbm makes LightGBM training fault.
# Checked by behaviour, not by sys.modules order: CPython re-inserts a package's key when its
# import finishes, so `lightgbm` legitimately appears after the pyarrow it imports itself.
ENTRY_POINTS = ["nss.prod.api", "nss.prod.cli", "nss.prod.fixture", "nss.prod.contracts"]
FIT = (
    "import numpy as np, lightgbm as lgb; "
    "X = np.random.default_rng(0).normal(size=(200, 3)); y = X[:, 0] * 2.0; "
    "lgb.LGBMRegressor(n_estimators=5, verbosity=-1).fit(X, y); print('ok')"
)


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


@pytest.mark.parametrize("module", ENTRY_POINTS)
def test_training_works_after_importing_an_entry_point(module: str) -> None:
    out = _run(f"import {module}; import pyarrow; {FIT}")
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr[-500:]


@pytest.mark.skipif(sys.platform != "win32", reason="the fault was reproduced on Windows only")
def test_negative_control_the_hazard_is_real() -> None:
    """Without nss.prod's guard, pyarrow before lightgbm faults. If this starts passing, LightGBM
    or pyarrow fixed it and the guard in nss/prod/__init__.py can be reconsidered."""
    out = _run(f"import pyarrow; {FIT}")
    assert out.returncode != 0 and "access violation" in out.stderr
