"""Agent-layer demonstration driver (tasks D4, rebuilt in U5 against the CURRENT gates).

Walks through the `agents/` layer's typical "find next season's winning styles and generate
concepts" request the way the orchestrator/sub-agents would per `agents/*.md`, and writes a
narrated transcript to `reports/agent_run_transcript.md`.

Every step is labelled in the transcript as one of:

- **LIVE**: a real MCP tool call over the stdio protocol (official `mcp` SDK `ClientSession`
  against a real `nss.mcp_server` subprocess, CPU only: `CUDA_VISIBLE_DEVICES` is emptied for the
  server). The driver refuses a call whose tool is not in the calling agent's allowlist, parsed
  from `agents/<agent>.md` (so a stale agent definition fails loudly instead of being narrated
  past).
- **REPLAY**: read from a table recorded by an earlier real run (the N9 candidate scores, the
  summer forecast, the recorded closed-loop table). Concept GENERATION is always a replay: no GPU
  is touched by this driver, `generate_concept` is never called.
- **HUMAN (recorded)**: the human visual check. It is a person's judgment, recorded in
  `nss.generate.h4_deliverables.HUMAN_CHECK`; this driver cannot perform it and says so.

The critic's retry loop is REPLAYED over the recorded N9 candidates of each style in seed order
with `agents/critic.md`'s rules (stop at the first candidate that clears every gating gate; at most
1 original + 2 retries). Nothing is fabricated: each verdict is derived from the recorded gate
columns of `reports/tables/n9_candidates_scored.csv`, and the shipped images plus the rejected
candidates are also re-scored LIVE through `score_concept` and compared to the recorded columns.

Run standalone, from the repo root (or any directory holding `data/`, `reports/`, `models/`; set
`NSS_DEMO_CWD` to point at one):
    uv run --no-sync python scripts/agent_demo.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from nss.generate import final_registry, n9_generate
from nss.generate.h4_deliverables import ALL_SELECTED, HUMAN_BRIEF_MET, HUMAN_CHECK

REPO_ROOT = Path(__file__).resolve().parent.parent
# Working directory of the MCP server and of this driver: the tools read `data/` and `reports/`
# by relative path, and `data/` is not in git, so a checkout without it points this elsewhere.
WORK_DIR = Path(os.environ.get("NSS_DEMO_CWD", REPO_ROOT))
TRANSCRIPT_PATH = REPO_ROOT / "reports" / "agent_run_transcript.md"
AGENTS_DIR = REPO_ROOT / "agents"
SCORED_PATH = Path("reports/tables/n9_candidates_scored.csv")
SELECTION_PATH = Path("reports/tables/final_selection_h4.csv")
CLOSED_LOOP_PATH = Path("reports/tables/q2_concept_forecast_retrieval.csv")
SUMMER_FORECAST_PATH = Path("reports/tables/seasonal_summer_forecast.csv")

USER_REQUEST = "find next season's winning styles and generate concepts"
RETRY_CAP = 2  # critic's max-2-retries-per-concept cap, per agents/critic.md.
GATING_COLUMNS = (
    ("gate1", "gate1_pass"),
    ("gate1b", "gate1b_pass"),
    ("integrity", "integrity_floor_pass"),  # GLOBAL floor gates; per-style floor is advisory
    ("gate2", "gate2_pass"),  # SmolVLM gates; Florence-2 is advisory
    ("gate3", "gate3_pass"),
)
LONG_STRING_LIMIT = 160
SWEATER, DRESS, TOP = final_registry.STYLE_ORDER
SUMMER = final_registry.SUMMER
EMERGING_TABLE_TOP_N = 10  # the whole emerging table: the current final three sit at ranks 2, 3, 5


@dataclass
class ToolCallRecord:
    """One real MCP tool call's request/response, captured for the transcript.

    Attributes:
        agent: Which sub-agent made the call (matches its `agents/*.md` allowlist).
        tool: The MCP tool name invoked.
        request: The exact arguments dict sent to `session.call_tool`.
        response_json: Pretty-printed JSON text of the server's `structured_content` response --
            the real wire response (long free-text strings shortened, see `shorten_long_strings`),
            or, when `note` is set, a projection of it onto the fields the note names.
        note: Empty for a verbatim response; else what the shown response was projected to.
    """

    agent: str
    tool: str
    request: dict[str, Any]
    response_json: str
    note: str = ""


def parse_allowlist(agent_md: str) -> frozenset[str]:
    """MCP tool names listed in an agent definition's "MCP tools it may call" section.

    Args:
        agent_md: The text of `agents/<agent>.md`.

    Returns:
        The backticked tool names that start a bullet in that section (empty for an agent that
        calls no tool directly, e.g. the orchestrator), read up to the next `## ` heading.
    """
    match = re.search(r"^## MCP tools it may call[^\n]*\n(.*?)(?=^## |\Z)", agent_md, re.M | re.S)
    if match is None:
        return frozenset()
    return frozenset(re.findall(r"^- `([a-z_]+)`", match.group(1), re.M))


def shorten_long_strings(obj: Any, limit: int = LONG_STRING_LIMIT) -> Any:
    """Recursively shorten strings longer than `limit`, marking the cut with the original length.

    Only for readability of the transcript (a Florence-2 caption is repeated once per attribute).
    Numbers, booleans and short strings are never touched.
    """
    if isinstance(obj, str) and len(obj) > limit:
        return f"{obj[:limit]}... [{len(obj)} chars, shortened for the transcript]"
    if isinstance(obj, dict):
        return {k: shorten_long_strings(v, limit) for k, v in obj.items()}
    if isinstance(obj, list):
        return [shorten_long_strings(v, limit) for v in obj]
    return obj


def format_tool_call(record: ToolCallRecord) -> str:
    """Render one real MCP tool call as a markdown request/response block.

    Args:
        record: The call's agent, tool name, request args, and response JSON text.

    Returns:
        A markdown string documenting the call was made over the real MCP protocol, with the
        exact request and response payloads.
    """
    return (
        f"**[LIVE] {record.agent}** calls MCP tool `{record.tool}` "
        "(real call over the MCP stdio protocol -- `ClientSession.call_tool` against a live "
        "`nss.mcp_server` subprocess):\n\n"
        f"Request:\n```json\n{json.dumps(record.request, indent=2)}\n```\n\n"
        f"Response{record.note}:\n```json\n{record.response_json}\n```\n"
    )


def project_forecast_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`forecast_styles` rows reduced to the ranking and guard fields (drops SHAP columns)."""
    keep = (
        "rank",
        "style_key",
        "predicted_intensity",
        "growth_ratio",
        "guard1_pass",
        "guard2_pass",
        "guard3_pass",
    )
    return [{k: r[k] for k in keep} for r in rows]


def project_closed_loop(fc: dict[str, Any], style_id: str) -> dict[str, Any]:
    """`forecast_concept` response with the top-5 as one line each, the intended style marked."""
    top5 = [
        f"{'>> ' if t['style_key'] == style_id else '   '}{i}. {t['style_key']}  "
        f"(similarity {t['similarity']:.3f}, forecast {t['forecast']:.1f}, rank {t['rank']})"
        for i, t in enumerate(fc["top5"], start=1)
    ]
    keep = (
        "sentence",
        "style_key",
        "forecast_units_per_product_per_week",
        "rank",
        "n_styles",
        "match_level",
        "confidence",
        "similarity",
        "margin",
        "n_indexed_styles",
    )
    return {
        **{k: fc[k] for k in keep},
        "top5 (>> = the style this concept was designed from)": top5,
    }


def failed_gates(row: dict[str, Any]) -> list[str]:
    """Names of the GATING gates a recorded `n9_candidates_scored.csv` row fails, in gate order.

    A missing or null gate result counts as failed: an unmeasured gate is never a pass.
    """
    return [name for name, col in GATING_COLUMNS if row.get(col) is not True]


def critic_replay(
    rows: list[dict[str, Any]], cap: int = RETRY_CAP
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    """Replay `agents/critic.md`'s retry loop over recorded candidates, in the order given.

    Attempt 1 is `rows[0]`; each REJECT is followed by the next recorded candidate as the retry
    (the N9 candidates differ only in seed at a fixed, swept scale). The loop stops at the first
    candidate that clears every gating gate (`PASS_PENDING_HUMAN`) or after 1 + `cap` attempts
    (`FAILED`, cap exhausted), exactly as the critic would.

    Returns:
        `(attempts, outcome, unexamined)`: one dict per attempt (`row`, `failed`, `verdict`), the
        loop's outcome, and the recorded candidates the loop never reached.
    """
    attempts: list[dict[str, Any]] = []
    outcome = "FAILED (retry cap exhausted)"
    for i, row in enumerate(rows):
        failed = failed_gates(row)
        attempts.append(
            {"row": row, "failed": failed, "verdict": "REJECT" if failed else "PASS_PENDING_HUMAN"}
        )
        if not failed:
            outcome = "PASS_PENDING_HUMAN"
            break
        if i == cap:
            break
    return attempts, outcome, rows[len(attempts) :]


def intended_position(top5: list[dict[str, Any]], style_key: str) -> int | None:
    """1-based position of the intended style in a retrieval top-5, or None if outside it."""
    for pos, entry in enumerate(top5, start=1):
        if entry["style_key"] == style_key:
            return pos
    return None


def format_candidate(row: dict[str, Any]) -> str:
    """One recorded candidate's gate numbers as a compact line (values straight from the row)."""
    answers = row.get("smolvlm_gate3_answers")
    return (
        f"seed {row['seed']}: G1 {'pass' if row['gate1_pass'] else 'FAIL'} "
        f"(CLIP {row['clip_mean_sim']:.3f}/{row['clip_p90_limit']:.3f}, "
        f"DINOv2 {row['dinov2_mean_sim']:.3f}/{row['dinov2_p90_limit']:.3f}); "
        f"G1b {'pass' if row['gate1b_pass'] else 'FAIL'} "
        f"(CLIP {row['clip_max_sim']:.3f}/{row['clip_gate1b_limit']:.3f}, "
        f"DINOv2 {row['dinov2_max_sim']:.3f}/{row['dinov2_gate1b_limit']:.3f}); "
        f"integrity {'pass' if row['integrity_floor_pass'] else 'FAIL'} "
        f"(closest ref {row['floor_max_sim']:.3f} vs global floor {row['global_floor_limit']:.3f}; "
        f"per-style floor advisory {'pass' if row['integrity_style_pass'] else 'fail'}); "
        f"G2 {'pass' if row['gate2_pass'] else 'FAIL'} (SmolVLM fidelity "
        f"{row['smolvlm_fidelity']:.2f}; Florence-2 advisory "
        f"{'pass' if row['gate2_advisory_pass'] else 'fail'}); "
        f"G3 {'pass' if row['gate3_pass'] else 'FAIL'} ({answers})"
    )


def format_qc_attempt(attempt: dict[str, Any], number: int) -> str:
    """Render one replayed critic attempt as a markdown bullet with its verdict and reasons."""
    row = attempt["row"]
    reasons = f" -- failed: {', '.join(attempt['failed'])}" if attempt["failed"] else ""
    return (
        f"- **Attempt {number}** (scale {row['scale']}, seed {row['seed']}) -> "
        f"**{attempt['verdict']}**{reasons}\n  - {format_candidate(row)}"
    )


def render_forecast_table(rows: list[dict[str, Any]]) -> str:
    """Render forecast rows (from a real `forecast_styles` response) as a markdown table.

    Args:
        rows: Row dicts as returned by the `forecast_styles` MCP tool (each carries at least
            `rank`, `style_key`, `predicted_intensity`).

    Returns:
        A markdown table, one row per input row, in the given order.
    """
    header = "| rank | style_key | predicted_intensity |\n|---|---|---|\n"
    body = "\n".join(
        f"| {r['rank']} | {r['style_key']} | {r['predicted_intensity']:.4f} |" for r in rows
    )
    return header + body + "\n"


def compare_live_to_recorded(live: dict[str, Any], row: dict[str, Any]) -> str:
    """Live `score_concept` result vs the recorded table row: per-gate agreement, stated plainly."""
    live_pass = {
        "gate1": live["gate1"]["pass"],
        "gate1b": live["gate1b"]["pass"],
        "integrity": live["integrity"]["pass"],
        "gate2": live["gate2"]["pass"],
        "gate3": live["gate3"]["pass"],
    }
    parts = []
    for name, col in GATING_COLUMNS:
        rec = row[col]
        parts.append(
            f"{name} live {live_pass[name]} / recorded {rec}"
            + ("" if live_pass[name] == rec else " **DIFFERS**")
        )
    agree = all(live_pass[n] == row[c] for n, c in GATING_COLUMNS)
    return (
        ("all five gating gates agree" if agree else "GATE DISAGREEMENT")
        + " ("
        + "; ".join(parts)
        + ")"
    )


def format_live_gates(live: dict[str, Any]) -> str:
    """Bullet summary of one live `score_concept` result (the numbers are copied, not rounded)."""
    g1, g1b, integ = live["gate1"], live["gate1b"], live["integrity"]
    g2, g3 = live["gate2"], live["gate3"]
    judges = "; ".join(
        f"{name} ({j['role']}) {j['fidelity']:.3f} vs threshold {j['threshold']:.3f} -> "
        f"{'pass' if j['pass'] else 'FAIL'}"
        for name, j in g2["judges"].items()
    )
    g1c, g1d = g1["clip"], g1["dinov2"]
    per_style = integ["per_style_floor_advisory"]
    return (
        f"- Gate 1: CLIP {g1c['similarity']:.3f} <= {g1c['limit_p90_of_real_pairs']:.3f}, "
        f"DINOv2 {g1d['similarity']:.3f} <= {g1d['limit_p90_of_real_pairs']:.3f} -> "
        f"{'pass' if g1['pass'] else 'FAIL'}\n"
        f"- Gate 1b: CLIP {g1b['clip']['closest_reference_similarity']:.3f} <= "
        f"{g1b['clip']['limit_p90_of_real_nearest_sibling']:.3f}, DINOv2 "
        f"{g1b['dinov2']['closest_reference_similarity']:.3f} <= "
        f"{g1b['dinov2']['limit_p90_of_real_nearest_sibling']:.3f} -> "
        f"{'pass' if g1b['pass'] else 'FAIL'} (exact-clone control failed as required: "
        f"{g1b['clone_control_failed_as_required']})\n"
        f"- Integrity (GLOBAL floor gates): closest reference "
        f"{integ['closest_reference_dinov2']:.3f} >= {integ['global_floor']:.3f} -> "
        f"{'pass' if integ['pass'] else 'FAIL'}; per-style floor (advisory) "
        f"{per_style['limit']:.3f} -> {'pass' if per_style['pass'] else 'fail'}\n"
        f"- Gate 2: {judges} -> **{'pass' if g2['pass'] else 'FAIL'}**\n"
        f"- Gate 3 (briefed changes visible, SmolVLM): answers {g3['answers']} -> "
        f"**{'pass' if g3['pass'] else 'FAIL'}**\n"
        f"- Human visual check: required = {live['human_visual_check']['required']}\n"
        f"- automated_gates_pass = {live['automated_gates_pass']}; verdict: `{live['verdict']}`\n"
    )


async def _call_tool(
    session: ClientSession,
    allowlists: dict[str, frozenset[str]],
    agent: str,
    tool: str,
    args: dict[str, Any],
    project: Callable[[Any], Any] | None = None,
    note: str = "",
) -> tuple[Any, ToolCallRecord]:
    """Call one MCP tool over a live session as `agent` and capture it as a `ToolCallRecord`.

    Args:
        session: An initialized `ClientSession` connected to the real `nss.mcp_server` subprocess.
        allowlists: Per-agent tool allowlists parsed from `agents/*.md`.
        agent: The sub-agent making this call (for the transcript).
        tool: The MCP tool name.
        args: Arguments to pass to the tool.
        project: Optional transcript-only projection of the response (the parsed result the
            caller gets back is always the full response).
        note: Text appended to "Response" in the transcript when `project` is used.

    Returns:
        `(parsed_result, record)` -- `parsed_result` is the tool's `structured_content`, unwrapped
        from its `{"result": ...}` envelope when the tool's return type is a list (MCP's schema
        convention for non-object return types); `record` is the call's request/response.

    Raises:
        PermissionError: `tool` is not in `agent`'s allowlist in `agents/<agent>.md`.
        RuntimeError: the tool call itself reported an error (`CallToolResult.is_error`).
    """
    if tool not in allowlists[agent]:
        raise PermissionError(
            f"agent {agent!r} may not call {tool!r}: its allowlist in agents/{agent}.md is "
            f"{sorted(allowlists[agent])}"
        )
    result = await session.call_tool(tool, args)
    if result.is_error:
        raise RuntimeError(f"MCP tool {tool!r} call failed for agent {agent!r}: {result.content}")
    structured = result.structured_content
    parsed = structured["result"] if set(structured) == {"result"} else structured
    record = ToolCallRecord(
        agent=agent,
        tool=tool,
        request=args,
        response_json=json.dumps(
            shorten_long_strings(project(parsed) if project else structured), indent=2
        ),
        note=note if project else "",
    )
    return parsed, record


def _scored_rows() -> dict[str, list[dict[str, Any]]]:
    """Recorded N9 candidates at each style's selected scale, per style, in seed order."""
    df = pl.read_csv(SCORED_PATH)
    out: dict[str, list[dict[str, Any]]] = {}
    for style_id, path in ALL_SELECTED.items():
        scale = df.filter(pl.col("image_path") == str(path))["scale"][0]
        rows = df.filter((pl.col("style_id") == style_id) & (pl.col("scale") == scale))
        out[style_id] = rows.sort("seed").to_dicts()
    return out


def _row_for(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    return next(r for r in rows if r["seed"] == seed)


def _image_path(style_id: str, seed: int) -> str:
    """Path of the N9 candidate of `style_id` with `seed`, at the same scale as the shipped one."""
    shipped = ALL_SELECTED[style_id]
    return (shipped.parent / re.sub(r"seed\d+", f"seed{seed}", shipped.name)).as_posix()


def _shipped_seed(style_id: str) -> int:
    return int(re.search(r"seed(\d+)", ALL_SELECTED[style_id].stem).group(1))  # type: ignore[union-attr]


def _brief_lines(style_id: str) -> tuple[list[str], str]:
    """The briefed changes (human design decisions) and the natural-language prompt of a style."""
    changes = n9_generate.CHANGES[style_id]["applied_changes"]
    return list(changes), n9_generate.natural_prompt(style_id, changes, None)


async def run_demo() -> str:
    """Run the full orchestrator -> sub-agent demo against the real MCP server; build the text.

    Returns:
        The complete transcript as markdown text (also written to `TRANSCRIPT_PATH` by `main`).
    """
    allowlists = {
        p.stem: parse_allowlist(p.read_text(encoding="utf-8")) for p in AGENTS_DIR.glob("*.md")
    }
    scored = _scored_rows()
    selection = {r["style_id"]: r for r in pl.read_csv(SELECTION_PATH).to_dicts()}
    closed_recorded = {r["style_id"]: r for r in pl.read_csv(CLOSED_LOOP_PATH).to_dicts()}
    sections: list[str] = []
    live_calls: list[str] = []

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "nss.mcp_server"],
        cwd=str(WORK_DIR),
        # CPU only: an emptied CUDA_VISIBLE_DEVICES makes torch.cuda.is_available() False, so the
        # CLIP/DINOv2/SmolVLM/Florence-2 calls run on CPU (greedy decoding: same answers as GPU
        # for the calls compared below). Offline: model weights must already be cached.
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"},
    )

    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        async def call(
            agent: str, tool: str, args: dict[str, Any], **view: Any
        ) -> tuple[Any, ToolCallRecord]:
            out = await _call_tool(session, allowlists, agent, tool, args, **view)
            live_calls.append(f"`{tool}` ({agent})")
            return out

        # --- Step 1: orchestrator -> forecaster ---
        sections.append(
            "## Step 1 -- orchestrator delegates to `forecaster`\n\n"
            "Per `agents/orchestrator.md` step 1: get the current pre-computed forecast. "
            "`forecaster`'s allowlist (`agents/forecaster.md`) permits `forecast_styles`. The "
            "emerging table has 10 rows, so the whole table is read.\n"
        )
        emerging, emerging_call = await call(
            "forecaster",
            "forecast_styles",
            {
                "origin_date": "2020-09-21",
                "horizon_weeks": 13,
                "table": "emerging",
                "top_n": EMERGING_TABLE_TOP_N,
            },
            project=project_forecast_rows,
            note=" (projected to rank, style_key, predicted_intensity, growth_ratio and the "
            "three guard flags; the tool's JSON also carries SHAP drivers per row)",
        )
        sections.append(format_tool_call(emerging_call))
        by_key = {r["style_key"]: r for r in emerging}
        chosen = [by_key[s] for s in final_registry.STYLE_ORDER]
        sections.append(
            "**forecaster's result** -- the three current final styles (selected from this table "
            "by `nss.models.reselect_final_three`, log in `reports/tables/final_three_selection_"
            "log.csv`: category exclusions and no shared colour or product type) sit at emerging "
            "ranks 2, 3 and 5:\n\n" + render_forecast_table(chosen)
        )
        summer = pl.read_csv(SUMMER_FORECAST_PATH).filter(pl.col("style_key") == SUMMER)
        if summer.height:
            s = summer.to_dicts()[0]
            sections.append(
                "**[REPLAY] forecaster, summer style.** `forecast_styles` serves only the "
                "2020-09-21 tables, so the summer concept's forecast (origin 2020-06-01, trained "
                "under a 13-week embargo) is read from `reports/tables/seasonal_summer_forecast."
                f"csv`: `{SUMMER}` predicted {s['predicted_intensity']:.2f}, realised "
                f"{s['realised_intensity']:.2f}, rank {s['predicted_rank_among_eligible']} of "
                f"{s['n_eligible_styles']} eligible.\n"
            )

        # --- Step 2: orchestrator -> data-analyst (optional supporting question) ---
        sections.append(
            "\n## Step 2 -- orchestrator delegates to `data-analyst` (optional step)\n\n"
            "Supporting question for the dress: what did raw unit volume do into the forecast "
            "origin? Two `query_transactions` calls (13 weeks ending at the origin, and the 13 "
            "weeks before).\n"
        )
        recent, recent_call = await call(
            "data-analyst",
            "query_transactions",
            {"style_key": DRESS, "date_from": "2020-06-23", "date_to": "2020-09-21"},
        )
        prior, prior_call = await call(
            "data-analyst",
            "query_transactions",
            {"style_key": DRESS, "date_from": "2020-03-24", "date_to": "2020-06-22"},
        )
        sections.append(format_tool_call(recent_call))
        sections.append(format_tool_call(prior_call))
        sections.append(
            f"**data-analyst's answer**: {recent['units']} units in the 13 weeks to the origin vs "
            f"{prior['units']} in the 13 weeks before ({recent['units'] / prior['units']:.2f}x). "
            "This is raw units. The forecast's `growth_ratio` "
            f"({by_key[DRESS]['growth_ratio']:.2f} for this style) is over predicted "
            "intensity (units per product per week), so the two measure different things and a "
            "raw-units fall over a seasonal window does not contradict it. A descriptive fact, "
            "not a causal claim.\n"
        )

        # --- Step 3: orchestrator -> style-profiler ---
        sections.append(
            "\n## Step 3 -- orchestrator delegates to `style-profiler` (once per style)\n\n"
            "Per `agents/style-profiler.md`: fetch the structured profile via `get_style_profile` "
            "(its only allowed tool). The design brief itself is a human design decision recorded "
            "in code (`nss.generate.n9_generate.CHANGES`: concrete, visually checkable changes "
            "with the colour anchor kept), not something an LLM invents here; it is **[REPLAY]** "
            "from that table and the N9 prompt builder (`natural_prompt`), not recomputed.\n"
        )
        for style_id in ALL_SELECTED:
            _profile, _profile_call = await call(
                "style-profiler", "get_style_profile", {"style_key": style_id}
            )
            # The profile JSON is long (13 weekly rows); the call is real, its response is
            # summarised beneath it rather than pasted four times.
            traj = _profile["trajectory"]
            drivers = ", ".join(d["feature"] for d in _profile["shap_drivers"]["drivers"][:3])
            sections.append(
                f"**[LIVE] style-profiler** calls `get_style_profile` for `{style_id}`; response "
                f"(summarised): trajectory available = {traj['available']}"
                + (
                    f", {traj['n_weeks_total']} weeks seen, recent mean units "
                    f"{traj['recent_mean_units']:.1f}"
                    if traj["available"]
                    else ""
                )
                + f"; SHAP source `{_profile['shap_drivers']['source']}` "
                f"(style-specific: {_profile['shap_drivers']['style_specific']}), top drivers: "
                f"{drivers or 'none'}.\n"
            )
            changes, prompt = _brief_lines(style_id)
            sections.append(
                f"**style-profiler's brief for `{style_id}`** (**[REPLAY]**): changes "
                f"{changes}; prompt: `{prompt}`\n"
            )

        # --- Step 4: orchestrator -> concept-designer (replay only: no GPU) ---
        sections.append(
            "\n## Step 4 -- orchestrator delegates to `concept-designer` (once per style)\n\n"
            "**[REPLAY] -- no `generate_concept` call, no GPU.** `concept-designer`'s only tool is "
            "`generate_concept` (SDXL + IP-Adapter, GPU). N9 already generated 8 seeds (42-49) per "
            "style at a per-style IP-Adapter scale chosen by a sweep (0.35 for all four), recorded "
            "in `reports/tables/n9_candidates_scored.csv` with sidecar JSONs. The candidates below "
            "are those recorded images; the run makes no claim to have generated anything. "
            "Note that N9 generated the 8 seeds as a batch, not in reaction to critic rejections; "
            "Step 5 replays the critic loop over them in seed order.\n"
        )
        for style_id, rows in scored.items():
            sections.append(
                f"- `{style_id}`: {len(rows)} recorded candidates at scale {rows[0]['scale']}, "
                f"seeds {[r['seed'] for r in rows]}; shipped = seed {_shipped_seed(style_id)}.\n"
            )

        # --- Step 5: orchestrator -> critic ---
        sections.append(
            "\n## Step 5 -- orchestrator delegates to `critic` (once per style, with retries)\n\n"
            "Per `agents/critic.md`: every automatic gate runs inside one `score_concept` call "
            "(Gate 1, Gate 1b with clone control, GLOBAL integrity floor with the per-style floor "
            "advisory, Gate 2 with SmolVLM gating and Florence-2 advisory, Gate 3); the human "
            "check is separate and never automated. The critic owns accept/reject and the retry "
            "cap (1 original + 2 retries).\n\n"
            "### 5a. Retry loop, REPLAYED over the recorded candidates\n\n"
            "Each verdict below is derived from the recorded gating columns (`failed_gates`); "
            "the retry parameter is the next recorded seed. Attempts 1-3 are the critic's budget.\n"
        )
        replays = {}
        for style_id, rows in scored.items():
            attempts, outcome, unexamined = critic_replay(rows)
            replays[style_id] = (attempts, outcome, unexamined)
            sections.append(f"**critic, `{style_id}` -- {outcome}**\n")
            for n, attempt in enumerate(attempts, start=1):
                sections.append(format_qc_attempt(attempt, n) + "\n")
            shipped = _row_for(rows, _shipped_seed(style_id))
            beyond = [r["seed"] for r in unexamined if not failed_gates(r)]
            shipped_failed = failed_gates(shipped)
            beyond_text = ", ".join(f"seed {x}" for x in beyond) or "none"
            if outcome.startswith("FAILED"):
                verdict_text = (
                    f"it fails the gating gate(s): {', '.join(shipped_failed)}"
                    if shipped_failed
                    else "it clears every gating gate"
                )
                extra = (
                    "Under the critic's cap alone the orchestrator would report this style as "
                    "FAILED with this history. The recorded batch continues past the cap; "
                    f"candidates beyond it that clear every gating gate: {beyond_text}. The "
                    f"shipped image is seed {shipped['seed']}; on its recorded row {verdict_text}."
                    + (
                        " (N9 selected best-of-8, so it went past the cap.)"
                        if not shipped_failed
                        else ""
                    )
                    + "\n"
                )
            else:
                extra = (
                    f"Shipped image is seed {shipped['seed']} (selected by "
                    "`h4_deliverables`: passes every automatic gate, then most briefed changes "
                    "visible, then highest fidelity, then a human look); it also clears every "
                    "gating gate on its recorded row.\n"
                )
            sections.append(extra)

        sections.append(
            "\n### 5b. The same gates, LIVE through `score_concept`\n\n"
            "The critic calls `score_concept` with `include_fidelity=true` and the brief's "
            "`applied_changes`. Real calls over the protocol, CPU only. The rejected candidates "
            "are re-scored live too, so the rejection is a live verdict, not only a table read. "
            "Responses are shown in full for the first two calls (long Florence-2 captions "
            "shortened) and as gate summaries for the rest; each is compared to the recorded "
            "columns.\n"
        )
        live_results: dict[tuple[str, int], dict[str, Any]] = {}
        plan = [
            (DRESS, 42, "REJECTED candidate (the first attempt of the dress replay)", True),
            (DRESS, _shipped_seed(DRESS), "the retry that was shipped", True),
            (SWEATER, 44, "REJECTED candidate (third attempt of the sweater replay)", False),
            (SWEATER, _shipped_seed(SWEATER), "shipped", False),
            (TOP, _shipped_seed(TOP), "shipped (recorded verdict FAIL)", False),
            (SUMMER, _shipped_seed(SUMMER), "shipped (recorded verdict FAIL)", False),
            (
                TOP,
                47,
                "DISCREPANCY CHECK: a recorded candidate that clears every gating gate",
                False,
            ),
        ]
        for style_id, seed, note, full_json in plan:
            changes, _prompt = _brief_lines(style_id)
            live, live_call = await call(
                "critic",
                "score_concept",
                {
                    "concept_path": _image_path(style_id, seed),
                    "style_key": style_id,
                    "include_fidelity": True,
                    "changes": changes,
                },
            )
            live_results[(style_id, seed)] = live
            name = final_registry.PLAIN_NAMES[style_id]
            sections.append(f"\n#### {name}, seed {seed} -- {note}\n")
            if full_json:
                sections.append(format_tool_call(live_call))
            else:
                sections.append(
                    f"**[LIVE] critic** calls `score_concept` with `concept_path="
                    f"{_image_path(style_id, seed)}`, `include_fidelity=true` (gate summary of the "
                    "real response):\n\n"
                )
            sections.append(format_live_gates(live))
            sections.append(
                "Live vs recorded: "
                + compare_live_to_recorded(live, _row_for(scored[style_id], seed))
                + "\n"
            )

        # --- Step 6: human check ---
        sections.append(
            "\n## Step 6 -- the human visual check\n\n"
            "**[HUMAN (recorded)] -- not performed by this run.** The critic can only forward "
            "`PASS_PENDING_HUMAN`; no agent decides shippability. What a person saw when they "
            "looked at the shipped images is recorded in `nss.generate.h4_deliverables."
            "HUMAN_CHECK` and is reproduced here verbatim. The committed copies are in "
            "`reports/concepts/`.\n"
        )
        for style_id in ALL_SELECTED:
            sections.append(
                f"- **{final_registry.PLAIN_NAMES[style_id]}** (seed {_shipped_seed(style_id)}): "
                f"every briefed change visible = {HUMAN_BRIEF_MET[style_id]}. "
                f"{HUMAN_CHECK[style_id]}\n"
            )

        # --- Step 7: closed loop ---
        sections.append(
            "\n## Step 7 -- orchestrator delegates the closed loop to `forecaster`\n\n"
            "`forecast_concept` (added to `agents/forecaster.md`'s allowlist in U5; it was in no "
            "agent's allowlist before) matches each shipped image to the nearest real catalogue "
            "style by CLIP + DINOv2 retrieval and looks up that style's forecast. **Prototype:** "
            "near-ties dominate, so the top-5 and the intended style's position are reported, and "
            "the result is about the archetype the picture reads as, not a demand forecast for "
            "the new design.\n"
        )
        closed_summary: dict[str, dict[str, Any]] = {}
        for style_id, path in ALL_SELECTED.items():
            args: dict[str, Any] = {"concept_path": path.as_posix()}
            if style_id == SUMMER:
                args["origin"] = "2020-06-01"  # the summer concept lives at its own origin
            fc, fc_call = await call(
                "forecaster",
                "forecast_concept",
                args,
                project=lambda r, sid=style_id: project_closed_loop(r, sid),
                note=" (projected: top-5 flattened to one line each, `judges` and "
                "`unavailable_judges` omitted; every other field verbatim)",
            )
            closed_summary[style_id] = fc
            pos = intended_position(fc["top5"], style_id)
            rec = closed_recorded.get(style_id, {})
            where = f"{pos}{'st' if pos == 1 else 'nd' if pos == 2 else 'rd' if pos == 3 else 'th'}"
            sections.append(format_tool_call(fc_call))
            name = final_registry.PLAIN_NAMES[style_id]
            recorded_top1 = rec.get("mapped_style_key")
            same = "matches" if recorded_top1 == fc["style_key"] else "DIFFERS FROM"
            same_rank = "matches" if rec.get("rank") == fc["rank"] else "DIFFERS FROM"
            sections.append(
                f"**Reading for {name}**: top-1 `{fc['style_key']}` "
                f"(similarity {fc['similarity']:.3f}), forecast "
                f"{fc['forecast_units_per_product_per_week']:.1f} units/product/week, rank "
                f"{fc['rank']} of {fc['n_styles']}, confidence {fc['confidence']}. Intended style "
                f"{'is ' + where + ' of the top 5' if pos else 'is OUTSIDE the top 5'}. Recorded "
                f"table (`q2_concept_forecast_retrieval.csv`): top-1 `{recorded_top1}`, rank "
                f"{rec.get('rank')}, confidence {rec.get('confidence')} -> live {same} the "
                f"recorded top-1 and {same_rank} the recorded rank.\n"
            )

        # --- Step 8: aggregation ---
        sections.append(
            "\n## Step 8 -- orchestrator aggregates the final result\n\n"
            "Per `agents/orchestrator.md`: each concept is reported with its per-gate verdict; a "
            "failing concept is a failed item, not fatal to the request. Verdicts below are the "
            "LIVE `score_concept` results on the shipped images, beside the recorded "
            "`final_selection_h4.csv` verdict.\n\n"
            "| Concept | live automatic gates | failed gates (live) | recorded (h4) | "
            "human: briefed changes visible |\n|---|---|---|---|---|"
        )
        n_pass = 0
        for style_id in ALL_SELECTED:
            live = live_results[(style_id, _shipped_seed(style_id))]
            failed = [
                n
                for n in ("gate1", "gate1b", "integrity", "gate2", "gate3")
                if live[n]["pass"] is not True
            ]
            verdict = "PASS_PENDING_HUMAN" if live["automated_gates_pass"] else "FAIL"
            n_pass += verdict == "PASS_PENDING_HUMAN"
            sections.append(
                f"| {final_registry.PLAIN_NAMES[style_id]} | {verdict} | "
                f"{', '.join(failed) or 'none'} | {selection[style_id]['automatic_gates']} | "
                f"{HUMAN_BRIEF_MET[style_id]} |"
            )
        sections.append(
            f"\n**Final outcome: {n_pass} of {len(ALL_SELECTED)} concepts clear every automatic "
            "gate and are forwarded for the human check** (which is recorded above, not "
            "re-performed). The others are reported with their failing gates.\n"
        )
        top_live = live_results[(TOP, 47)]
        if top_live["automated_gates_pass"]:
            sections.append(
                "\n**Unresolved discrepancy, flagged rather than absorbed.** The white top is "
                "reported as failing by mechanism (integrity floor and Gate 3), and the shipped "
                f"image (seed {_shipped_seed(TOP)}) does fail both. But the recorded candidate "
                "seed 47 (and 48) at the same scale clears every gating gate on "
                "`n9_candidates_scored.csv`, and the live `score_concept` call above confirms it "
                "for seed 47. `h4_deliverables.SELECTED` pins seed 42 for the white top, so "
                "the written selection rule (passes every automatic gate first) does not "
                "produce the shipped image. Whether seed 47 passes the human check has not been "
                "looked at by a person in this run. The owner of the write-up should decide "
                "whether the white top's failure is a property of the style or of seed choice.\n"
            )

    header = (
        "# Agent Run Transcript -- next-season-styles (task U5)\n\n"
        "> **Current.** Re-run in U5 against the CURRENT gates: Gate 1 (within-style p90), Gate 1b "
        "(closest reference, clone-validated), the GLOBAL integrity floor (per-style floor "
        "advisory), Gate 2 (SmolVLM gates, Florence-2 advisory), Gate 3 (briefed changes visible) "
        "and the human visual check, on the current final concepts (beige melange sweater, red "
        "dress, white jersey top, summer orange bikini top). It replaces the D4 transcript, whose "
        "banner said the agent layer still ran the old margin-band scoring: that was true of "
        "`score_concept` and `agents/critic.md` until this change, and both were updated in the "
        "same PR (the tool now runs the shipped gates; see the commit log).\n>\n"
        "> **What is live and what is replayed.** LIVE (real MCP tool calls over stdio, CPU only, "
        f"{len(live_calls)} calls): "
        + ", ".join(sorted(set(live_calls)))
        + ". REPLAYED from recorded tables: concept generation (no GPU, no `generate_concept` "
        "call), the design briefs, the summer forecast, and the critic's retry sequence "
        "(derived from `n9_candidates_scored.csv`; the same images are also re-scored live and "
        "compared). HUMAN (recorded, not performed here): the visual check.\n\n"
        f"## Request\n\n> {USER_REQUEST}\n\n"
        "## Run notes\n\n"
        "- MCP server: `python -m nss.mcp_server` over stdio, spawned by this driver "
        "(`scripts/agent_demo.py`) with `CUDA_VISIBLE_DEVICES` emptied, so CLIP, DINOv2, SmolVLM "
        "and Florence-2 ran on CPU.\n"
        "- MCP client: the official `mcp` Python SDK `ClientSession`. Every tool call is checked "
        "against the calling agent's allowlist parsed from `agents/<agent>.md` before it is sent.\n"
        "- The rejected candidate images are N9 outputs under `data/generated/n9/`, which is not "
        "in git: the live rejection calls are reproducible only with the local data tree; the "
        "shipped images are committed in `reports/concepts/`.\n"
    )
    return header + "\n" + "\n".join(sections)


def main() -> None:
    """Run the demo (from `WORK_DIR`) and write `reports/agent_run_transcript.md`."""
    os.chdir(WORK_DIR)
    transcript = asyncio.run(run_demo())
    TRANSCRIPT_PATH.write_text(transcript, encoding="utf-8")
    print(f"Wrote transcript to {TRANSCRIPT_PATH}")


if __name__ == "__main__":
    main()
