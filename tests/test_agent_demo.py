"""Light tests for scripts/agent_demo.py's pure transcript-formatting functions (task D4).

Only the formatting/rendering logic is tested here -- `run_demo`/`main` drive a real subprocess
MCP server and download real CLIP/DINOv2 weights, which is exactly the live behaviour this script
exists to demonstrate, not something to mock out in a unit test.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from agent_demo import (  # noqa: E402 -- see sys.path.insert above
    ToolCallRecord,
    format_qc_attempt,
    format_tool_call,
    render_forecast_table,
)


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
    assert "`forecast_styles`" in rendered
    assert "real call over the MCP stdio protocol" in rendered
    assert '"table": "incumbent"' in rendered
    assert '{"result": [{"rank": 1}]}' in rendered


def test_format_qc_attempt_reject_with_retry() -> None:
    """A rejected attempt with a next_scale renders the REJECT verdict and the retry parameter."""
    row = {
        "attempt_number": 0,
        "seed": 43,
        "ip_adapter_scale": 0.2,
        "clip_margin": 0.09480527128492089,
        "clip_in_band": True,
        "dino_margin": 0.7245466072644506,
        "dino_in_band": False,
        "margin_band_pass": False,
        "mean_attribute_fidelity": 0.5,
        "n_contributing_judges": 1,
        "fidelity_pass": False,
        "overall_pass": False,
        "retry_triggered": True,
        "next_scale": 0.9,
    }
    rendered = format_qc_attempt(row)
    assert "Attempt 0" in rendered
    assert "clip_margin=0.0948" in rendered
    assert "dino_margin=0.7245" in rendered
    assert "**REJECT**" in rendered
    assert "retry with ip_adapter_scale=0.9" in rendered


def test_format_qc_attempt_reject_cap_exhausted() -> None:
    """The final rejected attempt (no next_scale) reports the retry cap as exhausted."""
    row = {
        "attempt_number": 2,
        "seed": 43,
        "ip_adapter_scale": 0.8,
        "clip_margin": 0.135935686315809,
        "clip_in_band": False,
        "dino_margin": 0.6824453732797078,
        "dino_in_band": False,
        "margin_band_pass": False,
        "mean_attribute_fidelity": 0.75,
        "n_contributing_judges": 1,
        "fidelity_pass": True,
        "overall_pass": False,
        "retry_triggered": False,
        "next_scale": None,
    }
    rendered = format_qc_attempt(row)
    assert "**REJECT**" in rendered
    assert "retry cap exhausted" in rendered


def test_format_qc_attempt_pass_has_no_retry_note() -> None:
    """A passing attempt renders PASS with no retry/cap note appended."""
    row = {
        "attempt_number": 0,
        "seed": 1,
        "ip_adapter_scale": 0.5,
        "clip_margin": 0.05,
        "clip_in_band": True,
        "dino_margin": 0.2,
        "dino_in_band": True,
        "margin_band_pass": True,
        "mean_attribute_fidelity": 0.9,
        "n_contributing_judges": 1,
        "fidelity_pass": True,
        "overall_pass": True,
        "retry_triggered": False,
        "next_scale": None,
    }
    rendered = format_qc_attempt(row)
    assert rendered.endswith("**PASS**")


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
