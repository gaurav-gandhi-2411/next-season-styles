# ruff: noqa: E501  -- long lines are the fixed prompt text and a printout
"""K2: LLM critic+orchestrator judgment on cases the written rule does not resolve.

Rule and rubrics pre-registered in `reports/v3/PREREGISTRATION.md` (section K2, commit f14d7a2);
fixtures in `evals/fixtures/silent_rule/`. The model is shown the spec (critic.md and
orchestrator.md as they stood at 896b10d, before the K3/K5 edits, so the rule is silent) and one
scenario, and answers with JSON. Arm A: spec only. Arm B: spec plus the fixed project-evidence note.
Three runs per case and arm.

    uv run --no-sync python scripts/agent_llm_silent.py run
    uv run --no-sync python scripts/agent_llm_silent.py show
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_llm import parse_frontmatter  # noqa: E402
from agent_llm_eval import call_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "evals" / "fixtures" / "silent_rule"
RUNS = ROOT / "reports" / "tables" / "v3_silent_rule_runs.jsonl"
SPEC_COMMIT = "896b10d"
N_RUNS = 3
EVIDENCE = """Project evidence (from the experiments on this pipeline):
- With the production prompt and 8 concatenated references, briefed changes appear at ip_adapter_scale 0.25 to 0.35 and are weaker at 0.45. At 0.6 to 0.7 the references dominate and the changes disappear. With the older prompt, scales 0.15 and 0.25 collapsed into fabric swatches. The swept window is 0.15 to 0.45; the per-style choice is 0.35.
- Gate 3's local reader says yes too easily (specificity 0.58 against human labels).
- SmolVLM's Gate 2 reading of the bikini top is identical (fidelity 0.283) on all 24 seeds; Florence-2 reads its pattern correctly."""
INSTRUCTION = """You are being evaluated on ONE decision in a running pipeline. You are the critic
and the orchestrator for it: decide the verdict and what happens next, as your role definitions
say, using your own judgment wherever they are silent. You have no tools; do not ask questions.
Reply with exactly one JSON object and nothing else:
{"verdict": "REJECT" | "PASS_PENDING_HUMAN" | "INCONCLUSIVE" | "FAILED",
 "outcome": "RETRY" | "FORWARD" | "FAILED" | "ESCALATE",
 "next_agent": "concept-designer" | "forecaster" | "critic" | null,
 "adjust": {"param": "ip_adapter_scale" | "seed" | null, "direction": "up" | "down" | "keep" | null, "value": <number or null>},
 "escalate_to_human": true | false,
 "flags_for_human": ["<short notes for the human reviewer>"],
 "reason": "<two to four sentences of reasoning>"}"""


def spec_text() -> str:
    """critic.md and orchestrator.md bodies at the pre-K3 commit."""
    parts = []
    for name in ("critic", "orchestrator"):
        raw = subprocess.run(  # noqa: S603, S607 -- fixed argv
            ["git", "show", f"{SPEC_COMMIT}:agents/{name}.md"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=ROOT,
            check=True,
        ).stdout
        parts.append(parse_frontmatter(raw)[1])
    return "\n\n=====\n\n".join(parts)


def fixtures() -> list[dict[str, Any]]:
    """The ten cases, in id order."""
    return [json.loads(p.read_text("utf-8")) for p in sorted(FIXTURES.glob("S*.json"))]


def user_prompt(case: dict[str, Any]) -> str:
    """What the model sees for one case."""
    return (
        f"{case['scenario']}\n\nscore_concept result:\n"
        f"{json.dumps(case['tool_output'], indent=1)}"
    )


def run_all() -> None:
    """Every missing (case, arm, run); appended to the runs file as they finish."""
    base = spec_text()
    systems = {
        "A": f"{base}\n\n=====\n\n{INSTRUCTION}",
        "B": f"{base}\n\n=====\n\n{EVIDENCE}\n\n{INSTRUCTION}",
    }
    done = (
        {json.loads(x)["key"] for x in RUNS.read_text("utf-8").splitlines() if x}
        if RUNS.exists()
        else set()
    )
    todo = [
        (f"{c['id']}|{arm}|{r}", c, arm, r)
        for c in fixtures()
        for arm in "AB"
        for r in range(N_RUNS)
        if f"{c['id']}|{arm}|{r}" not in done
    ]
    print(f"{len(todo)} calls", flush=True)

    def one(t: tuple[str, dict[str, Any], str, int]) -> dict[str, Any]:
        key, c, arm, r = t
        return {
            "key": key,
            "case": c["id"],
            "arm": arm,
            "run": r,
            **call_model(systems[arm], user_prompt(c)),
        }

    with ThreadPoolExecutor(4) as pool, RUNS.open("a", encoding="utf-8") as fh:
        for rec in pool.map(one, todo):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()


def show() -> None:
    """Compact printout of every answer, grouped for grading."""
    recs = [json.loads(x) for x in RUNS.read_text("utf-8").splitlines() if x]
    for c in fixtures():
        print(f"\n##### {c['id']} {c['title']}")
        for arm in "AB":
            for r in sorted(
                (x for x in recs if x["case"] == c["id"] and x["arm"] == arm),
                key=lambda x: x["run"],
            ):
                rep = r.get("reply") or {"ERROR": r.get("error")}
                adj = rep.get("adjust") or {}
                print(
                    f"[{arm}{r['run']}] {rep.get('verdict')} {rep.get('outcome')} -> {rep.get('next_agent')} "
                    f"| adj {adj.get('param')} {adj.get('direction')} {adj.get('value')} "
                    f"| human {rep.get('escalate_to_human')} | flags {rep.get('flags_for_human')}"
                )
                print(f"     why: {rep.get('reason')}")


if __name__ == "__main__":
    {"run": run_all, "show": show}[sys.argv[1]]()
