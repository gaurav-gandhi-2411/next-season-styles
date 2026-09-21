"""L5: does the LLM still raise the scale for a Gate 2-only miss once the K3 rule is in critic.md?

Rule pre-registered in `reports/v3/PREREGISTRATION.md` (section L5, commit c5799c3). S09 only, three
runs, Sonnet, the same call shape as K2, system prompt = the CURRENT `agents/critic.md` and
`agents/orchestrator.md` (K3 in context) and no evidence note. The habit persists iff at least one
run changes `ip_adapter_scale`.

    uv run --no-sync python scripts/agent_llm_s09_rerun.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_llm import AGENTS_DIR, parse_frontmatter  # noqa: E402
from agent_llm_eval import call_model  # noqa: E402
from agent_llm_silent import INSTRUCTION, fixtures, user_prompt  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "reports" / "tables" / "v3_s09_rerun_runs.jsonl"


def main() -> None:
    """Three S09 runs with the current spec; print each decision."""
    body = "\n\n=====\n\n".join(
        parse_frontmatter((AGENTS_DIR / f"{n}.md").read_text(encoding="utf-8"))[1]
        for n in ("critic", "orchestrator")
    )
    system = f"{body}\n\n=====\n\n{INSTRUCTION}"
    case = next(c for c in fixtures() if c["id"] == "S09")
    with OUT.open("w", encoding="utf-8") as fh:
        for run in range(3):
            res = call_model(system, user_prompt(case))
            fh.write(json.dumps({"case": "S09", "run": run, **res}) + "\n")
            rep = res.get("reply") or {}
            print(
                run,
                rep.get("verdict"),
                rep.get("outcome"),
                rep.get("adjust"),
                rep.get("escalate_to_human"),
                flush=True,
            )


if __name__ == "__main__":
    main()
