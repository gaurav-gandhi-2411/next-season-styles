"""J1: the agent files follow the Claude Code sub-agent format and the headless command is safe."""

from __future__ import annotations

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
        # calls no data tool; may spawn only the five sub-agents (not Explore or a general agent)
        assert not body_tools
        assert fm_tools == {
            "Agent(forecaster, data-analyst, style-profiler, concept-designer, critic)"
        }
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
    assert cmd[cmd.index("--setting-sources") + 1] == "project"
    assert "--strict-mcp-config" in cmd and cmd[cmd.index("--permission-mode") + 1] == "dontAsk"
    assert len(" ".join(cmd)) < 20_000  # far below Windows' 32,767-character limit
    assert not any("ANTHROPIC_API_KEY" in c or "sk-" in c for c in cmd)


def test_missing_frontmatter_is_an_error() -> None:
    with pytest.raises(ValueError, match="no frontmatter"):
        agent_llm.parse_frontmatter("# Agent: x\n")


def test_staged_agents_are_byte_identical_to_the_sources(tmp_path: Path) -> None:
    project = agent_llm.stage_project("pytest")
    for src in agent_llm.AGENTS_DIR.glob("*.md"):
        assert (project / ".claude" / "agents" / src.name).read_bytes() == src.read_bytes()


@pytest.mark.parametrize("name", sorted(AGENTS))
def test_frontmatter_is_valid_yaml(name: str) -> None:
    """A colon inside a description made the orchestrator's frontmatter invalid YAML, so Claude
    Code silently did not register the agent; the simple parser above would not have noticed."""
    yaml = pytest.importorskip("yaml")
    text = (agent_llm.AGENTS_DIR / f"{name}.md").read_text(encoding="utf-8")
    head = text[4:].partition("\n---\n")[0]
    parsed = yaml.safe_load(head)
    assert isinstance(parsed, dict) and parsed["name"] == name
    assert parsed["description"] == AGENTS[name]["description"]


def test_mcp_server_chdirs_to_the_repo_root_itself() -> None:
    """Claude Code ignores a `cwd` key, so the boot command must change directory."""
    for server in agent_llm.mcp_config()["mcpServers"].values():
        assert "cwd" not in server
        code = server["args"][-1]
        assert f"os.chdir({str(agent_llm.ROOT)!r})" in code and "nss.mcp_server" in code


def test_split_tools_keeps_the_agent_allowlist_together() -> None:
    assert agent_llm.split_tools("Agent(a, b), mcp__x__y, mcp__x__z") == [
        "Agent(a, b)",
        "mcp__x__y",
        "mcp__x__z",
    ]
