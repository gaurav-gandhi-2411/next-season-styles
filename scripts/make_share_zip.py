"""Build a shareable zip of the repo (task L4): tracked files only, denylist enforced, no secrets.

Takes `git ls-files` (so `.env`, `vertex-sa.json`, `data/`, `.venv`, caches and other ignored files
are never candidates), then passes every file through `nss.share_safety.assert_shareable`
(file-name, directory and content checks) as a second, independent barrier. Any hit aborts and no
zip is written. `.git` is not included (history is available from the remote repository).

Usage:
    uv run python scripts/make_share_zip.py            # or: make share-zip
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

from nss.share_safety import (
    UnshareableFileError,
    assert_shareable,
    denied_reason,
    manual_handling_warning,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "dist" / "next-season-styles-share.zip"


def tracked_files(root: Path = REPO_ROOT) -> list[Path]:
    """Files git tracks under `root` that exist on disk (paths relative to `root`)."""
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True
    ).stdout.decode("utf-8")
    return [Path(p) for p in out.split("\0") if p and (root / p).is_file()]


def build_zip(root: Path = REPO_ROOT, out_path: Path = OUT_PATH) -> tuple[int, int]:
    """Write the zip; returns `(n_files, size_bytes)`. Raises before writing on any denylist hit."""
    files: list[Path] = []
    for rel in tracked_files(root):
        reason = denied_reason(root / rel, root=root)
        if reason is not None and reason.startswith("inside a denied directory"):
            continue  # e.g. tracked `data/**/.gitkeep`: expected, simply left out of the zip
        assert_shareable(root / rel, root=root)  # raises UnshareableFileError; nothing written yet
        files.append(rel)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in files:
            zf.write(root / rel, arcname=f"next-season-styles/{rel.as_posix()}")
    return len(files), out_path.stat().st_size


def main() -> int:
    """Build the zip, print its size and the manual-handling warning."""
    print(manual_handling_warning(REPO_ROOT))
    try:
        n, size = build_zip()
    except UnshareableFileError as exc:
        print(f"ABORTED, no zip written: {exc}")
        return 1
    print(f"wrote {OUT_PATH} ({n} files, {size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
