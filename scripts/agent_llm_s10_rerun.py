"""A3: does the LLM now terminate S10 with REJECT under the terminal-human-REJECT rule?

Rule pre-registered in `reports/v3/PREREGISTRATION.md` (section O). S10 only, three runs, Sonnet,
the same call shape as K2/L5, system prompt = the CURRENT `agents/critic.md` and
`agents/orchestrator.md` (the A3 terminal-human-REJECT rule in context) and no evidence note.
Required: all three runs return REJECT, with no FORWARD and no escalation.

    uv run --no-sync python scripts/agent_llm_s10_rerun.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_llm import AGENTS_DIR, parse_frontmatter  # noqa: E402
from agent_llm_eval import call_model  # noqa: E402
from agent_llm_silent import INSTRUCTION, fixtures, user_prompt  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "reports" / "tables" / "v3_s10_rerun_runs.jsonl"


def main() -> None:
    """Three S10 runs with the current spec; print each decision."""
    body = "\n\n=====\n\n".join(
        parse_frontmatter((AGENTS_DIR / f"{n}.md").read_text(encoding="utf-8"))[1]
        for n in ("critic", "orchestrator")
    )
    system = f"{body}\n\n=====\n\n{INSTRUCTION}"
    case = next(c for c in fixtures() if c["id"] == "S10")
    with OUT.open("w", encoding="utf-8") as fh:
        for run in range(3):
            res = call_model(system, user_prompt(case))
            fh.write(json.dumps({"case": "S10", "run": run, **res}) + "\n")
            rep = res.get("reply") or {}
            print(
                run,
                rep.get("verdict"),
                rep.get("outcome"),
                rep.get("next_agent"),
                rep.get("escalate_to_human"),
                flush=True,
            )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
