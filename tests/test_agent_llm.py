"""J1: the agent files follow the Claude Code sub-agent format and the headless command is safe."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import agent_llm  # noqa: E402 -- see sys.path.insert above
from agent_demo import parse_allowlist  # noqa: E402

AGENTS = agent_llm.load_agents()


def test_every_agent_file_has_frontmatter_and_the_role_text_after_it() -> None:
    assert set(AGENTS) == {
        "orchestrator",
        "data-analyst",
        "forecaster",
        "style-profiler",
        "concept-designer",
        "critic",
    }
    for name, agent in AGENTS.items():
        assert agent["description"] and agent["tools"]
        assert agent["prompt"].startswith(f"# Agent: {name}")  # role text unchanged, no frontmatter


@pytest.mark.parametrize("name", sorted(AGENTS))
def test_frontmatter_tools_match_the_body_allowlist(name: str) -> None:
    """The MCP tools in the frontmatter are exactly the allowlist section, on the right server."""
    text = (agent_llm.AGENTS_DIR / f"{name}.md").read_text(encoding="utf-8")
    body_tools = parse_allowlist(text)
    fm_tools = set(AGENTS[name]["tools"])
    if name == "orchestrator":
        assert fm_tools == {"Agent"} and not body_tools  # calls no data tool
        return
    expected = {
        f"mcp__{'nss_gpu' if t == 'generate_concept' else 'nss_cpu'}__{t}" for t in body_tools
    }
    assert fm_tools == expected


def test_only_generation_is_on_the_gpu_server() -> None:
    cfg = agent_llm.mcp_config()["mcpServers"]
    assert cfg["nss_cpu"]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert "CUDA_VISIBLE_DEVICES" not in cfg["nss_gpu"]["env"]
    gpu_tools = {t for a in AGENTS.values() for t in a["tools"] if t.startswith("mcp__nss_gpu__")}
    assert gpu_tools == {"mcp__nss_gpu__generate_concept"}


def test_command_is_least_privilege_and_uses_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-never-reach-the-child")
    assert "ANTHROPIC_API_KEY" not in agent_llm.clean_env()
    cmd = agent_llm.build_command("go", "sonnet", 10, Path("m.json"))
    assert cmd[cmd.index("--tools") + 1] == "Agent"  # no Bash/Read/Write/Edit
    assert cmd[cmd.index("--setting-sources") + 1] == ""
    assert "--strict-mcp-config" in cmd and cmd[cmd.index("--permission-mode") + 1] == "dontAsk"
    assert json.loads(cmd[cmd.index("--agents") + 1]).keys() == AGENTS.keys()
    assert not any("ANTHROPIC_API_KEY" in c or "sk-" in c for c in cmd)


def test_missing_frontmatter_is_an_error() -> None:
    with pytest.raises(ValueError, match="no frontmatter"):
        agent_llm.parse_frontmatter("# Agent: x\n")
