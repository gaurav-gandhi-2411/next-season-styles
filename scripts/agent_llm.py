"""J1: run the orchestrator as a real LLM through headless Claude Code (`claude -p`).

The sub-agents are `agents/*.md` (Claude Code sub-agent format: name / description / tools
frontmatter, role text unchanged). They are registered with `--agents` built from those files, so
there is one source of truth and the same files also work dropped into `.claude/agents/`. Two MCP
servers are attached with `--strict-mcp-config`: `nss_gpu` (generation only) and `nss_cpu` (every
other tool; the GPU is hidden so scoring never competes with SDXL for 8 GB). The LLM makes the
routing, retry and escalation decisions; every tool stays deterministic.

Authentication is the Claude Max login already on this machine. `ANTHROPIC_API_KEY` is removed from
the child's environment and never read; nothing here can use an API key.

    uv run --no-sync python scripts/agent_llm.py run       # the fully live LLM-orchestrated run
    uv run --no-sync python scripts/agent_llm.py render F  # re-render a saved stream as markdown
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = ROOT / "agents"
LOG_DIR = ROOT / "data" / "generated" / "v3_logs"
TRANSCRIPT = ROOT / "reports" / "agent_run_transcript_llm.md"
SUBAGENT_MODEL = "sonnet"  # executors; the orchestrator's model is chosen per run
RESULT_LIMIT = 1800  # characters of one tool result shown in the transcript


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split `---\\nkey: value\\n---\\nbody`; raises if the file has no frontmatter."""
    if not text.startswith("---\n"):
        raise ValueError("agent file has no frontmatter")
    head, _, body = text[4:].partition("\n---\n")
    fields = {}
    for line in head.splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields, body.lstrip("\n")


def load_agents() -> dict[str, dict[str, Any]]:
    """`--agents` JSON: name -> description, prompt (the role text), tools, model."""
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(AGENTS_DIR.glob("*.md")):
        fields, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        if fields["name"] != path.stem:
            raise ValueError(f"{path.name}: frontmatter name {fields['name']!r} != file name")
        out[path.stem] = {
            "description": fields["description"],
            "prompt": body,
            "tools": [t.strip() for t in fields["tools"].split(",")],
            "model": SUBAGENT_MODEL,
        }
    return out


def mcp_config() -> dict[str, Any]:
    """The two MCP servers (same `nss.mcp_server`; only the visible GPU differs)."""
    base = {"command": sys.executable, "args": ["-m", "nss.mcp_server"], "cwd": str(ROOT)}
    return {
        "mcpServers": {
            "nss_gpu": {**base, "env": {"HF_HUB_OFFLINE": "1"}},
            "nss_cpu": {**base, "env": {"HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": ""}},
        }
    }


def allowed_tools() -> list[str]:
    """Every MCP tool a sub-agent lists, plus `Agent` for the orchestrator, nothing else."""
    tools = {t for a in load_agents().values() for t in a["tools"]}
    return sorted(tools)


def build_command(prompt: str, model: str, max_turns: int, mcp_path: Path) -> list[str]:
    """The `claude -p` command line (no shell; JSON goes in as one argument)."""
    return [
        "claude",
        "-p",
        prompt,
        "--agent",
        "orchestrator",
        "--agents",
        json.dumps(load_agents()),
        "--mcp-config",
        str(mcp_path),
        "--strict-mcp-config",
        "--setting-sources",
        "",  # no user/project settings, CLAUDE.md or hooks: the agents get their own role text only
        "--tools",
        "Agent",  # built-in tools: only the sub-agent launcher; data tools come from MCP alone
        "--allowedTools",
        *allowed_tools(),
        "--permission-mode",
        "dontAsk",  # anything not allowed above is refused, never prompted (headless)
        "--model",
        model,
        "--max-turns",
        str(max_turns),
        "--output-format",
        "stream-json",
        "--verbose",
        "--no-session-persistence",
    ]


def clean_env() -> dict[str, str]:
    """The environment for the child: `ANTHROPIC_API_KEY` removed (Claude Max login only)."""
    return {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}


def _short(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if len(text) > RESULT_LIMIT:
        return f"{text[:RESULT_LIMIT]}... [{len(text)} chars, shortened for the transcript]"
    return text


def _result_text(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content)


def render(events: list[dict[str, Any]], header: str) -> str:
    """Markdown transcript of a stream-json run: who said what, every tool call and its result."""
    spawned: dict[str, str] = {}  # Agent tool_use id -> subagent type
    parts = [header]
    for ev in events:
        kind = ev.get("type")
        parent = ev.get("parent_tool_use_id")
        who = spawned.get(parent, "orchestrator") if parent else "orchestrator"
        if kind == "assistant":
            for block in ev["message"]["content"]:
                if block["type"] == "text" and block["text"].strip():
                    parts.append(f"**[{who}, LLM]** {block['text'].strip()}\n")
                elif block["type"] == "tool_use":
                    name, args = block["name"], block["input"]
                    if name == "Agent":
                        sub = args.get("subagent_type", "?")
                        spawned[block["id"]] = sub
                        parts.append(
                            f"**[{who}, LLM] delegates to `{sub}`** -- prompt: "
                            f"{_short(args.get('prompt'))}\n"
                        )
                    else:
                        parts.append(
                            f"**[{who}, LLM] calls MCP tool `{name}`** with " f"`{_short(args)}`\n"
                        )
        elif kind == "user":
            msg = ev.get("message", {})
            for block in msg.get("content", []) if isinstance(msg.get("content"), list) else []:
                if block.get("type") == "tool_result":
                    tid = block.get("tool_use_id")
                    label = (
                        f"`{spawned[tid]}` returns to orchestrator"
                        if tid in spawned
                        else "tool result"
                    )
                    parts.append(f"> {label}: {_short(_result_text(block.get('content')))}\n")
        elif kind == "result":
            parts.append(
                f"\n## Run summary\n\n- stop reason: `{ev.get('subtype')}`; "
                f"turns: {ev.get('num_turns')}; "
                f"wall time {ev.get('duration_ms', 0) / 1000:.0f} s; "
                f"list-price cost (not billed on Max): ${ev.get('total_cost_usd', 0):.2f}\n"
                f"- sub-agent stats: `{json.dumps(ev.get('subagent_stats', {}))[:400]}`\n"
            )
    return "\n".join(parts)


def run_stream(prompt: str, model: str, max_turns: int, tag: str) -> list[dict[str, Any]]:
    """Run `claude -p` headless, save the raw stream, return the parsed events."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    mcp_path = LOG_DIR / f"agent_llm_mcp_{tag}.json"
    mcp_path.write_text(json.dumps(mcp_config(), indent=1), encoding="utf-8")
    stream_path = LOG_DIR / f"agent_llm_stream_{tag}.jsonl"
    cmd = build_command(prompt, model, max_turns, mcp_path)
    t0 = time.time()
    with stream_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(  # noqa: S603 -- fixed argv, no shell
            cmd,
            stdout=fh,
            stderr=subprocess.PIPE,
            text=True,
            env=clean_env(),
            cwd=ROOT,
            check=False,
        )
    print(f"claude exited {proc.returncode} after {time.time() - t0:.0f} s", flush=True)
    if proc.stderr.strip():
        print("stderr:", proc.stderr[-1500:], flush=True)
    return [json.loads(x) for x in stream_path.read_text("utf-8").splitlines() if x.strip()]


def main(argv: list[str]) -> None:
    """`run`: the live LLM-orchestrated run (writes the transcript); `render F`: re-render."""
    header = (
        "# Agent workflow, LLM-orchestrated path\n\n"
        "**This is the LLM-orchestrated path**, distinct from the deterministic driver "
        "(`reports/agent_run_transcript_live.md`). The orchestrator and every sub-agent below is a "
        "Claude model run through headless Claude Code (`claude -p`, Claude Max login, no API "
        "key); the routing, retry and escalation decisions are the models' own. Every tool is the "
        "same deterministic MCP tool the scripted path calls. Sub-agent definitions: "
        "`agents/*.md` (Claude Code sub-agent format). The human visual check is not performed "
        "by anything here.\n"
    )
    if argv and argv[0] == "render":
        events = [json.loads(x) for x in Path(argv[1]).read_text("utf-8").splitlines() if x.strip()]
        TRANSCRIPT.write_text(render(events, header), encoding="utf-8")
        return
    prompt = Path(argv[1]).read_text("utf-8") if len(argv) > 1 else sys.stdin.read()
    model = argv[2] if len(argv) > 2 else "opus"
    events = run_stream(prompt, model, 80, "run")
    TRANSCRIPT.write_text(render(events, header + f"\nOrchestrator model: `{model}`.\n"), "utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
