"""J4: does the MCP `generate_concept` tool, in production mode, reproduce a submitted concept?

Generates the submitted sweater and dress through the real MCP tool (GPU server over stdio,
concept-designer's allowlist enforced) with their submitted scale and seed, then compares each
with the committed candidate the submission was chosen from (`data/generated/n9/...`) by sha256
and by pixel difference. Writes `reports/tables/v3_generation_reproduction.csv`.

    uv run --no-sync python scripts/verify_generation_reproduction.py
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_demo import _call_tool, parse_allowlist  # noqa: E402

from nss.generate.final_selection_figures import ALL_SELECTED  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "tables" / "v3_generation_reproduction.csv"
STYLES = list(ALL_SELECTED)[:2]  # sweater, dress


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def run() -> None:
    """Generate each submitted (style, scale, seed) via the tool and diff it against the record."""
    allow = {p.stem: parse_allowlist(p.read_text("utf-8")) for p in (ROOT / "agents").glob("*.md")}
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "nss.mcp_server"],
        cwd=str(ROOT),
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )
    rows = []
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        for style in STYLES:
            ref = ROOT / ALL_SELECTED[style]
            m = re.search(r"s([0-9.]+)_seed(\d+)", ref.stem)
            scale, seed = float(m.group(1)), int(m.group(2))
            paths, _ = await _call_tool(
                session,
                allow,
                "concept-designer",
                "generate_concept",
                {"style_key": style, "ip_adapter_scale": scale, "seed": seed, "n": 1},
            )
            new = ROOT / paths[0]
            a = np.asarray(Image.open(ref).convert("RGB"), dtype=np.int16)
            b = np.asarray(Image.open(new).convert("RGB"), dtype=np.int16)
            same_shape = a.shape == b.shape
            diff = np.abs(a - b) if same_shape else None
            rows.append(
                {
                    "style": style,
                    "scale": scale,
                    "seed": seed,
                    "recorded": ref.relative_to(ROOT).as_posix(),
                    "regenerated": new.relative_to(ROOT).as_posix(),
                    "sha256_equal": _sha(ref) == _sha(new),
                    "same_shape": same_shape,
                    "max_abs_pixel_diff": int(diff.max()) if diff is not None else -1,
                    "mean_abs_pixel_diff": float(diff.mean()) if diff is not None else -1.0,
                }
            )
            print(rows[-1], flush=True)
    pl.DataFrame(rows).write_csv(OUT)


if __name__ == "__main__":
    asyncio.run(run())
