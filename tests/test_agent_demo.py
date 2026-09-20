"""Light tests for scripts/agent_demo.py's pure logic.

Only the pure functions are tested here (allowlist parsing, the critic retry-loop replay, the
formatting helpers). `run_demo`/`main` drive a real subprocess MCP server that loads CLIP, DINOv2,
SmolVLM and Florence-2, which is exactly the live behaviour the script exists to demonstrate, not
something to mock out in a unit test.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from agent_demo import (  # noqa: E402 -- see sys.path.insert above
    ToolCallRecord,
    critic_replay,
    failed_gates,
    format_qc_attempt,
    format_tool_call,
    intended_position,
    parse_allowlist,
    render_forecast_table,
    shorten_long_strings,
)

AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"


def _row(seed: int, **fails: bool) -> dict[str, Any]:
    """A recorded-candidate row: every gating gate passes unless `<col>=False` is given."""
    row: dict[str, Any] = {
        "seed": seed,
        "scale": 0.35,
        "gate1_pass": True,
        "gate1b_pass": True,
        "integrity_floor_pass": True,
        "gate2_pass": True,
        "gate3_pass": True,
        "clip_mean_sim": 0.8,
        "clip_p90_limit": 0.9,
        "dinov2_mean_sim": 0.7,
        "dinov2_p90_limit": 0.8,
        "clip_max_sim": 0.85,
        "clip_gate1b_limit": 0.95,
        "dinov2_max_sim": 0.75,
        "dinov2_gate1b_limit": 0.9,
        "floor_max_sim": 0.75,
        "global_floor_limit": 0.78,
        "integrity_style_pass": False,
        "smolvlm_fidelity": 0.28,
        "gate2_advisory_pass": True,
        "smolvlm_gate3_answers": "YN",
    }
    row.update(fails)
    return row


def test_parse_allowlist_reads_only_the_allowlist_section() -> None:
    """Bullets outside the "MCP tools it may call" section are not tools."""
    md = (
        "## Role\n\n- `not_a_tool` mentioned in prose\n\n"
        "## MCP tools it may call (allowlist)\n\n"
        "- `score_concept` -- scores\n- `get_style_profile` -- reads\n\n"
        "No other tools.\n\n## Failure\n\n- `generate_concept` is banned\n"
    )
    assert parse_allowlist(md) == {"score_concept", "get_style_profile"}
    assert parse_allowlist("## Role\n\nnothing here\n") == frozenset()


def test_shipped_agent_allowlists_cover_the_demo_calls() -> None:
    """The agent definitions grant exactly the tools the transcript has each agent call."""
    allow = {
        p.stem: parse_allowlist(p.read_text(encoding="utf-8")) for p in AGENTS_DIR.glob("*.md")
    }
    assert "forecast_styles" in allow["forecaster"] and "forecast_concept" in allow["forecaster"]
    assert "query_transactions" in allow["data-analyst"]
    assert "get_style_profile" in allow["style-profiler"]
    assert "score_concept" in allow["critic"]
    assert allow["concept-designer"] == {"generate_concept"}
    assert allow["orchestrator"] == frozenset()


def test_failed_gates_treats_missing_and_null_as_failed() -> None:
    """An unmeasured gate is never a pass (fail-closed)."""
    assert failed_gates(_row(1)) == []
    assert failed_gates(_row(1, integrity_floor_pass=False, gate3_pass=False)) == [
        "integrity",
        "gate3",
    ]
    row = _row(1)
    row["gate3_pass"] = None
    del row["gate2_pass"]
    assert failed_gates(row) == ["gate2", "gate3"]


def test_critic_replay_reject_then_retry_passes() -> None:
    """A rejected first attempt is followed by the next recorded seed, which passes."""
    rows = [_row(42, integrity_floor_pass=False), _row(43), _row(44)]
    attempts, outcome, unexamined = critic_replay(rows)
    assert [a["verdict"] for a in attempts] == ["REJECT", "PASS_PENDING_HUMAN"]
    assert attempts[0]["failed"] == ["integrity"]
    assert outcome == "PASS_PENDING_HUMAN"
    assert [r["seed"] for r in unexamined] == [44]


def test_critic_replay_cap_is_one_original_plus_two_retries() -> None:
    """Three rejections exhaust the cap; a passing 4th candidate is beyond it, not used."""
    rows = [_row(s, gate2_pass=False) for s in (42, 43, 44)] + [_row(45)]
    attempts, outcome, unexamined = critic_replay(rows)
    assert len(attempts) == 3 and outcome.startswith("FAILED")
    assert [r["seed"] for r in unexamined] == [45]


def test_format_qc_attempt_names_verdict_and_failed_gates() -> None:
    """A REJECT line lists which gates failed; a pass line has no failure note."""
    rejected = critic_replay([_row(42, gate3_pass=False)])[0][0]
    assert "**REJECT** -- failed: gate3" in format_qc_attempt(rejected, 1)
    passed = critic_replay([_row(43)])[0][0]
    rendered = format_qc_attempt(passed, 2)
    assert "**PASS_PENDING_HUMAN**" in rendered and "failed:" not in rendered.split("\n")[0]


def test_intended_position() -> None:
    """1-based position in the top 5, None when outside it."""
    top5 = [{"style_key": k} for k in "ABCDE"]
    assert intended_position(top5, "C") == 3
    assert intended_position(top5, "Z") is None


def test_shorten_long_strings_only_touches_long_strings() -> None:
    """Numbers, booleans and short strings survive; a long one is cut with its length noted."""
    out = shorten_long_strings({"a": 0.5, "b": True, "c": "short", "d": "x" * 500}, limit=50)
    assert out["a"] == 0.5 and out["b"] is True and out["c"] == "short"
    assert out["d"].startswith("x" * 50) and "500 chars" in out["d"]


def test_format_tool_call_includes_agent_tool_and_payloads() -> None:
    """The rendered block names the agent/tool and embeds the exact request/response JSON."""
    record = ToolCallRecord(
        agent="forecaster",
        tool="forecast_styles",
        request={"table": "incumbent", "top_n": 1},
        response_json='{"result": [{"rank": 1}]}',
    )
    rendered = format_tool_call(record)
    assert "forecaster" in rendered
    assert "[LIVE]" in rendered
    assert "`forecast_styles`" in rendered
    assert "real call over the MCP stdio protocol" in rendered
    assert '"table": "incumbent"' in rendered
    assert '{"result": [{"rank": 1}]}' in rendered


def test_render_forecast_table_one_row_per_style() -> None:
    """The forecast table has one markdown row per input row, in the given order."""
    rows = [
        {"rank": 1, "style_key": "A", "predicted_intensity": 33.68792198187379},
        {"rank": 2, "style_key": "B", "predicted_intensity": 18.970321246493675},
    ]
    rendered = render_forecast_table(rows)
    data_lines = rendered.splitlines()[2:]  # skip the header row + the "---" separator row
    assert len(data_lines) == 2
    assert "| 1 | A | 33.6879 |" in rendered
    assert "| 2 | B | 18.9703 |" in rendered


def test_call_tool_refuses_a_tool_outside_the_allowlist() -> None:
    """A call the agent definition does not permit raises before anything is sent."""
    import asyncio

    from agent_demo import _call_tool

    with pytest.raises(PermissionError, match="allowlist"):
        asyncio.run(
            _call_tool(
                None,  # type: ignore[arg-type] -- never reached: the allowlist check comes first
                {"critic": frozenset({"score_concept"})},
                "critic",
                "generate_concept",
                {},
            )
        )
