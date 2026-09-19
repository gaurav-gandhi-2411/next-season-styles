# ruff: noqa: E501  -- long lines are the bundle README template text
"""Assemble and verify the reviewer bundle `reports/SUBMISSION/` (task K9).

Copies exactly the files the reviewer receives from their source-of-truth locations (so the bundle
can never drift from the repo), writes the bundle README, then verifies every file: images decode,
HTML has no external references, markdown files are non-empty, and the byte-for-byte copies match
their sources.

Usage:
    uv run python scripts/build_submission.py
"""

from __future__ import annotations

import filecmp
import re
import shutil
import sys
from pathlib import Path

from PIL import Image

BUNDLE = Path("reports/SUBMISSION")
SOURCES: dict[str, Path] = {
    "FINAL_concepts.png": Path("reports/figures/FINAL_concepts.png"),
    "evidence_chain.png": Path("reports/figures/evidence_chain.png"),
    "seasonal_comparison.png": Path("reports/figures/seasonal_comparison.png"),
    "DEMO.html": Path("reports/DEMO.html"),
    "WRITEUP.md": Path("reports/WRITEUP.md"),
    "SUBMISSION_CHECKLIST.md": Path("reports/SUBMISSION_CHECKLIST.md"),
}
README = """# next-season-styles: submission bundle

Start with `DEMO.html` (open it in any browser; it is self-contained, needs no server or internet).

| File | What it is |
|---|---|
| `DEMO.html` | The concepts, how each traces to its forecast, the model evidence, the seasonal view, and what could not be verified, for a non-specialist |
| `FINAL_concepts.png` | The three generated concepts, one per forecast style |
| `evidence_chain.png` | Per concept: references, brief, checks, verdict |
| `seasonal_comparison.png` | The same pipeline for autumn/winter and for summer |
| `WRITEUP.md` | The full argument and its limits (about 1,900 words) |
| `SUBMISSION_CHECKLIST.md` | Each required item mapped to the file that satisfies it |

**Repository:** this bundle is cut from the project's git repository, branch `main` (run
`git log -1` for the exact commit). The code, tests, agents, skills, MCP server and every table cited
here live there.

- Reviewers without a GPU: `uv run --no-sync python scripts/run_pipeline.py --dry-run` (about 1 to
  2.5 minutes, writes only to a scratch directory; needs the H&M data in `data/`).
- Full reproduction (CUDA GPU): `uv run --no-sync python scripts/run_pipeline.py`.
- MCP server check: `uv run --no-sync python scripts/mcp_smoke_test.py` (expected output is in the
  repository README).
"""


def build() -> list[str]:
    """Copy sources into the bundle, write the README; return the bundle's file names."""
    BUNDLE.mkdir(parents=True, exist_ok=True)
    for name, src in SOURCES.items():
        if not src.exists():
            raise SystemExit(f"missing source for {name}: {src}")
        shutil.copyfile(src, BUNDLE / name)
    (BUNDLE / "README.md").write_text(README, encoding="utf-8")
    return sorted(p.name for p in BUNDLE.iterdir())


def verify() -> list[str]:
    """Check every bundle file opens and is self-contained; return a list of problems."""
    problems: list[str] = []
    expected = {*SOURCES, "README.md"}
    present = {p.name for p in BUNDLE.iterdir()}
    if present != expected:
        problems.append(
            f"bundle contents differ: extra={present - expected} missing={expected - present}"
        )
    for name, src in SOURCES.items():
        f = BUNDLE / name
        if not f.exists():
            continue
        if not filecmp.cmp(f, src, shallow=False):
            problems.append(f"{name} differs from its source {src}")
        if name.endswith(".png"):
            with Image.open(f) as im:
                im.verify()
        elif name.endswith(".html"):
            html = f.read_text(encoding="utf-8")
            if re.search(r"""(?:src|href)=["']https?://""", html) or "<script" in html:
                problems.append(f"{name} has an external reference or a script")
            if html.count("data:image/") < 10:
                problems.append(f"{name} has fewer inlined images than expected")
        elif f.stat().st_size < 500:
            problems.append(f"{name} is suspiciously small")
    return problems


def main() -> int:
    """Build then verify; non-zero exit on any problem."""
    names = build()
    problems = verify()
    print("bundle:", ", ".join(names))
    for p in problems:
        print("PROBLEM:", p)
    print("OK" if not problems else "FAILED")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
