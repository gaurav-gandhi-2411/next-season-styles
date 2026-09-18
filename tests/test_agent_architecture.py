from __future__ import annotations

from pathlib import Path

from nss.viz.agent_architecture import main

# A blank/near-blank PNG (e.g. an empty axes) is a few KB; a real multi-box, multi-arrow diagram
# at dpi=150 is consistently well over 50KB -- a cheap, meaningful non-triviality floor without
# pinning exact pixel content (rule: light sanity check, not a formal visual-regression test).
MIN_NON_TRIVIAL_BYTES = 50_000


def test_main_writes_non_trivial_png(tmp_path: Path) -> None:
    """`main()` runs end-to-end and writes a real (non-empty, non-trivial) PNG file."""
    out_path = tmp_path / "agent_architecture.png"
    result = main(out_path)

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > MIN_NON_TRIVIAL_BYTES


def test_main_creates_missing_parent_dirs(tmp_path: Path) -> None:
    """`main()` creates any missing parent directories for `out_path`."""
    out_path = tmp_path / "nested" / "dir" / "agent_architecture.png"
    result = main(out_path)

    assert result.exists()
