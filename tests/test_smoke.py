from __future__ import annotations

import importlib


def test_package_imports() -> None:
    """The nss package and its subpackages must be importable."""
    for module in ["nss", "nss.data", "nss.features", "nss.models", "nss.viz"]:
        importlib.import_module(module)
