"""MCP smoke test a reviewer can repeat (task K5).

Reads the `mcpServers` config block from the README's "MCP server" section AS WRITTEN (only the
documented `cwd` placeholder is substituted with this checkout's path, which is exactly what the
README tells the reader to do), launches the server with that command over stdio, connects a real
MCP client, lists the tools, and calls `forecast_styles`, `get_style_profile` and `score_concept`.

Usage (from the repo root; needs the data/ artifacts the tools read):
    uv run --no-sync python scripts/mcp_smoke_test.py

Exit code 0 iff both tools return real payloads; prints the raw JSON so it can be pasted into the
README's expected-output block.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
CWD_PLACEHOLDER = "/absolute/path/to/next-season-styles"
SERVER_NAME = "next-season-styles"


def documented_server_config() -> dict[str, object]:
    """The `mcpServers[SERVER_NAME]` entry from the README's fenced json block."""
    text = README.read_text(encoding="utf-8")
    for block in re.findall(r"```json\n(.*?)\n```", text, flags=re.S):
        cfg = json.loads(block)
        if SERVER_NAME in cfg.get("mcpServers", {}):
            return dict(cfg["mcpServers"][SERVER_NAME])
    raise SystemExit(f"README has no mcpServers.{SERVER_NAME} config block")


def _elide(obj: object, keep: int = 2) -> object:
    """Recursively keep only the first `keep` items of long lists, saying so explicitly."""
    if isinstance(obj, dict):
        return {k: _elide(v, keep) for k, v in obj.items()}
    if isinstance(obj, list):
        head = [_elide(v, keep) for v in obj[:keep]]
        return head + [f"... {len(obj) - keep} more entries elided"] if len(obj) > keep else head
    return obj


def _payload(result: object) -> object:
    """Structured JSON payload of a tool result (falls back to concatenated text content)."""
    structured = getattr(result, "structured_content", None) or getattr(
        result, "structuredContent", None
    )
    if structured is not None:
        return structured.get("result", structured) if isinstance(structured, dict) else structured
    text = "".join(getattr(c, "text", "") for c in getattr(result, "content", []))
    return json.loads(text)


SWEATER_STYLE = "Ladieswear || Sweater || Knitwear || Beige || Melange"


async def run() -> int:
    """Connect over stdio with the documented config and call three tools."""
    cfg = documented_server_config()
    cwd = str(cfg["cwd"]).replace(CWD_PLACEHOLDER, str(REPO_ROOT))
    params = StdioServerParameters(command=str(cfg["command"]), args=list(cfg["args"]), cwd=cwd)  # type: ignore[arg-type]
    print(f"launching: {params.command} {' '.join(params.args)}  (cwd={cwd})")
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        names = sorted(t.name for t in tools.tools)
        print("tools:", names)
        forecast = await session.call_tool(
            "forecast_styles",
            {"origin_date": "2020-09-21", "horizon_weeks": 13, "table": "incumbent", "top_n": 3},
        )
        rows = _payload(forecast)
        print("forecast_styles ->", json.dumps(_elide(rows[:1]), indent=2))
        style_key = rows[0]["style_key"]  # type: ignore[index]
        profile = await session.call_tool("get_style_profile", {"style_key": style_key})
        print("get_style_profile ->", json.dumps(_elide(_payload(profile)), indent=2))
        concept = REPO_ROOT / "reports" / "concepts" / "beige-knit-sweater_s0.35_seed45.png"
        scored = await session.call_tool(
            "score_concept", {"concept_path": str(concept), "style_key": SWEATER_STYLE}
        )
        gates = _payload(scored)
        print("score_concept ->", json.dumps(gates, indent=2))
        ok = (
            len(names) == 8
            and bool(rows)
            and not getattr(profile, "is_error", False)
            and gates["human_visual_check"]["required"] is True  # type: ignore[index]
            and gates["gate1b"]["clone_control_failed_as_required"] is True  # type: ignore[index]
        )
    print(f"forecast_styles returned {len(rows)} rows (only the first is shown above)")  # type: ignore[arg-type]
    print("MCP smoke test:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
