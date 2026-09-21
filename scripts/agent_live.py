"""H3 4a: the agent workflow with every step live, including generation.

Rule pre-registered in `reports/v3/PREREGISTRATION.md` (H3 / 4a, commit 14a0ae7). Runs the
orchestrator's flow for the three autumn/winter styles through real MCP calls over stdio, the
per-agent allowlists from `agents/*.md` enforced by the driver: forecaster `forecast_styles`,
style-profiler `get_style_profile`, concept-designer `generate_concept` (GPU server), critic
`score_concept` (CPU server), the critic's retry loop (1 original + 2 retries, the pre-registered
adjusted-parameter rule), forecaster `forecast_concept` on a PASS_PENDING_HUMAN concept.

Step labels: LIVE (a real MCP call), INPUT (a fixed human-authored input), LOCAL (an in-process
pure function), NOT PERFORMED (the human visual check). Nothing is replayed. Writes
`reports/agent_run_transcript_live.md` and `reports/tables/v3_agent_live_attempts.csv`.

    uv run --no-sync python scripts/agent_live.py
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from collections import Counter
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import polars as pl
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_demo import (  # noqa: E402
    ToolCallRecord,
    _call_tool,
    format_live_gates,
    format_tool_call,
    parse_allowlist,
    project_closed_loop,
    project_forecast_rows,
    render_forecast_table,
)

from nss.generate import agent_eval as ae  # noqa: E402
from nss.generate import concept_generation, final_registry  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT = ROOT / "reports" / "agent_run_transcript_live.md"
ATTEMPTS = ROOT / "reports" / "tables" / "v3_agent_live_attempts.csv"
ARCHIVE = ROOT / "data" / "generated" / "agent_live"
STYLES = final_registry.STYLE_ORDER  # sweater, dress, top; the summer style has no live forecast
ORIGIN = "2020-09-21"
START_SCALE, START_SEED = 0.35, 42
LABELS = ("LIVE", "INPUT", "LOCAL", "NOT PERFORMED")


class Transcript:
    """Ordered steps, each with a mandatory label; renders the markdown and the label counts."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.counts: Counter[str] = Counter()

    def heading(self, text: str) -> None:
        """A section heading (not a step)."""
        self.parts.append(f"\n{text}\n")

    def step(self, label: str, text: str, seconds: float | None = None) -> None:
        """One labelled step; an unknown label is an error (the lint the rule requires)."""
        if label not in LABELS:
            raise ValueError(f"unlabelled or unknown step label {label!r}")
        self.counts[label] += 1
        took = f" ({seconds:.1f} s)" if seconds is not None else ""
        self.parts.append(f"**[{label}]{took}** {text}\n")

    def render(self) -> str:
        """The transcript body followed by the count of steps by label."""
        table = "| Label | Steps |\n|---|---|\n" + "\n".join(
            f"| {k} | {self.counts[k]} |" for k in LABELS
        )
        return "\n".join(self.parts) + "\n\n## Steps by label\n\n" + table + "\n"


def _server(gpu: bool) -> StdioServerParameters:
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    if not gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""  # CPU only, as in the demo and the 4b eval
    return StdioServerParameters(
        command=sys.executable, args=["-m", "nss.mcp_server"], cwd=str(ROOT), env=env
    )


def _archive(image: str, style: str, attempt: int, scale: float, seed: int) -> str:
    """Copy a generated image and its sidecar to a unique name (the tool's own name is not)."""
    src = ROOT / image
    slug = style.split(" || ")[1].lower().replace(" ", "-")
    dst = ARCHIVE / f"{slug}_a{attempt}_s{scale:.2f}_seed{seed}.png"
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    shutil.copyfile(src.with_suffix(".json"), dst.with_suffix(".json"))
    return dst.relative_to(ROOT).as_posix()


async def _timed(coro: Any) -> tuple[Any, float]:
    t0 = time.time()
    out = await coro
    return out, time.time() - t0


async def run() -> None:
    """The live workflow; writes the transcript and the attempts table."""
    allow = {p.stem: parse_allowlist(p.read_text("utf-8")) for p in (ROOT / "agents").glob("*.md")}
    refs_all = concept_generation.load_refs()
    tr = Transcript()
    attempts: list[dict[str, Any]] = []
    t_start = time.time()
    async with AsyncExitStack() as stack:
        cpu = await stack.enter_async_context(stdio_client(_server(gpu=False)))
        gen = await stack.enter_async_context(stdio_client(_server(gpu=True)))
        s_cpu = await stack.enter_async_context(ClientSession(*cpu))
        s_gen = await stack.enter_async_context(ClientSession(*gen))
        await s_cpu.initialize()
        await s_gen.initialize()

        async def call(
            session: ClientSession, agent: str, tool: str, args: dict[str, Any], **view: Any
        ) -> tuple[Any, ToolCallRecord, float]:
            (parsed, rec), secs = await _timed(
                _call_tool(session, allow, agent, tool, args, **view)
            )
            return parsed, rec, secs

        tr.heading("## Step 1 -- orchestrator delegates to `forecaster`")
        emerging, rec, secs = await call(
            s_cpu,
            "forecaster",
            "forecast_styles",
            {"origin_date": ORIGIN, "horizon_weeks": 13, "table": "emerging", "top_n": 10},
            project=project_forecast_rows,
            note=" (projected to rank, style, intensity, growth ratio and guard flags)",
        )
        tr.step("LIVE", format_tool_call(rec), secs)
        by_key = {r["style_key"]: r for r in emerging}
        tr.step(
            "LOCAL",
            "orchestrator reads the forecaster's table and selects the three styles named in "
            "`final_registry.STYLE_ORDER` (the submitted three; selection rule "
            "`nss.models.reselect_final_three`):\n\n"
            + render_forecast_table([by_key[s] for s in STYLES]),
        )

        tr.heading(
            "## Step 2 -- `data-analyst`: not invoked (optional in `orchestrator.md`, not asked)"
        )

        outcomes: dict[str, str] = {}
        for style in STYLES:
            tr.heading(f"## Style `{style}`")
            profile, _rec, secs = await call(
                s_cpu, "style-profiler", "get_style_profile", {"style_key": style}
            )
            traj = profile["trajectory"]
            tr.step(
                "LIVE",
                f"style-profiler calls `get_style_profile` for `{style}`; response summary: "
                f"trajectory available = {traj['available']}, SHAP source "
                f"`{profile['shap_drivers']['source']}`.",
                secs,
            )
            changes = list(concept_generation.CHANGES[style]["applied_changes"])
            tr.step(
                "INPUT",
                f"design brief (human-authored, `concept_generation.CHANGES`): {changes}",
            )
            prompt = concept_generation.natural_prompt(style, changes, None)
            refs = [p.as_posix() for p in refs_all[style][: concept_generation.N_REFS]]
            tr.step(
                "LOCAL", f"`natural_prompt` builds the prompt: `{prompt}`; {len(refs)} references."
            )

            scale, seed = START_SCALE, START_SEED
            outcome = "FAILED (retry cap exhausted)"
            for attempt in range(1, ae.RETRY_CAP + 2):
                gen_args = {
                    "prompt": prompt,
                    "reference_images": refs,
                    "backend": "local_sdxl",
                    "ip_adapter_scale": scale,
                    "seed": seed,
                    "n": 1,
                }
                paths, _rec, gsecs = await call(
                    s_gen, "concept-designer", "generate_concept", gen_args
                )
                image = paths[0]
                archived = _archive(image, style, attempt, scale, seed)
                tr.step(
                    "LIVE",
                    f"attempt {attempt}: concept-designer calls `generate_concept` "
                    f"(ip_adapter_scale {scale}, seed {seed}, n 1) -> `{image}` (the tool names "
                    "files `seed<seed>_<i>.png` with no style or scale, so the next call with the "
                    f"same seed overwrites it; the driver archived this one at `{archived}`)",
                    gsecs,
                )
                score_args = {
                    "concept_path": image,
                    "style_key": style,
                    "include_fidelity": True,
                    "changes": changes,
                }
                score = None
                for _try in (1, 2):  # critic.md: retry the SCORING call once on an error
                    try:
                        score, _rec, ssecs = await call(
                            s_cpu, "critic", "score_concept", score_args
                        )
                        break
                    except Exception as exc:  # noqa: BLE001
                        tr.step("LIVE", f"critic `score_concept` raised: {exc!r}")
                if score is None:
                    verdict, failed = ae.INCONCLUSIVE, []
                else:
                    passes, clone_ok = ae.passes_from_score(score)
                    verdict, failed = ae.decide(passes, clone_ok), ae.failing(passes)
                    tr.step(
                        "LIVE",
                        f"attempt {attempt}: critic calls `score_concept` (include_fidelity, brief "
                        f"changes):\n\n{format_live_gates(score)}",
                        ssecs,
                    )
                hop = ae.route(verdict, attempt, failed)
                tr.step(
                    "LOCAL",
                    f"critic decision `{verdict}` (failing gates: {failed or 'none'}); router: "
                    f"{hop.outcome}"
                    + (f" -> `{hop.next_agent}`, adjust `{hop.adjust}`" if hop.next_agent else ""),
                )
                attempts.append(
                    {
                        "style": style,
                        "attempt": attempt,
                        "ip_adapter_scale": scale,
                        "seed": seed,
                        "image": archived,
                        "verdict": verdict,
                        "failing": ",".join(failed),
                        "next": hop.outcome,
                        "generate_s": round(gsecs, 1),
                    }
                )
                if verdict == ae.PASS:
                    fc, rec, fsecs = await call(
                        s_cpu, "forecaster", "forecast_concept", {"concept_path": image},
                        project=lambda r, s=style: project_closed_loop(r, s),
                        note=" (top-5 as one line each; `>>` marks the intended style)",
                    )  # fmt: skip
                    tr.step("LIVE", format_tool_call(rec), fsecs)
                    outcome = "PASS_PENDING_HUMAN"
                    break
                if hop.outcome != "RETRY":
                    outcome = (
                        f"{hop.outcome}"
                        if verdict == ae.INCONCLUSIVE
                        else "FAILED (retry cap exhausted)"
                    )
                    break
                scale, seed = ae.next_attempt(failed, scale, seed)
            outcomes[style] = outcome
            last = attempts[-1]["image"]
            tr.step(
                "NOT PERFORMED",
                f"human visual check for `{style}` (outcome so far: {outcome}). A person must open "
                f"`{last}`; this driver cannot perform it and records no result for it.",
            )
    total = time.time() - t_start
    body = (
        "# Agent workflow, fully live\n\n"
        "Every tool call below is a real MCP call over stdio (`ClientSession.call_tool` against a "
        "live `nss.mcp_server` subprocess; generation on the GPU server, everything else on a CPU "
        "server), with the calling agent's allowlist from `agents/*.md` enforced by the driver. "
        "Nothing is replayed. `INPUT` = a fixed human-authored input; `LOCAL` = an in-process pure "
        "function (prompt builder, the critic's decision rule, the router); `NOT PERFORMED` = the "
        "human visual check, which is a person's judgment and is not done here. Rule committed "
        "before the run: `reports/v3/PREREGISTRATION.md`, H3 / 4a.\n\n"
        "**This run is not evidence about the submitted concepts.** The MCP `generate_concept` "
        "tool calls `backends.generate_concept`, which exposes no negative prompt and no concat "
        "reference mode, and does not use the weighted compel prompt; the deliverables were made "
        "by `levers.generate_variant` (concat mode, negative prompt, weighted prompt). The images "
        "here are a different, weaker configuration, and none is a candidate for the submission.\n"
        f"\nWall time for the whole run: {total:.0f} s. Cost: $0 (local).\n"
        + tr.render()
        + "\n## Outcomes\n\n"
        + "\n".join(f"- `{s}`: {o}" for s, o in outcomes.items())
        + "\n"
    )
    TRANSCRIPT.write_text(body, encoding="utf-8")
    pl.DataFrame(attempts).write_csv(ATTEMPTS)
    print(json.dumps({"outcomes": outcomes, "steps": dict(tr.counts), "seconds": round(total)}))


if __name__ == "__main__":
    asyncio.run(run())
