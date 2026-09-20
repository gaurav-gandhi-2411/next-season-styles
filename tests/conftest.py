from __future__ import annotations

from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "needs_data(*paths): skip when any of these gitignored local files is absent, as on a "
        "clean clone (the H&M dataset and the fetched exemplar images are not committed)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Turn a missing local data file into an informative skip instead of a FileNotFoundError.

    Only tests that declare `@pytest.mark.needs_data(<path>, ...)` are affected, and only their
    skip condition is added; no assertion is changed or relaxed. Paths are relative to the repo
    root, where pytest is run.
    """
    for item in items:
        marker = item.get_closest_marker("needs_data")
        if marker is None:
            continue
        missing = [p for p in marker.args if not Path(p).exists()]
        if missing:
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        f"needs {', '.join(missing)}, which is not committed (gitignored data "
                        "tier); absent on a clean clone. See the README's data section."
                    )
                )
            )
