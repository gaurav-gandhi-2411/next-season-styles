"""J7: which gate rejected each human-approved concept the critic rejected, and why (report only).

Rule pre-registered in `reports/v3/PREREGISTRATION.md` (section J). Reads the cached live
`score_concept` results of the two submitted concepts chosen without a gate pass (white jersey
top, bikini top) and, for context, the recorded candidate tables. It changes no threshold, judge
or gate.

    uv run --no-sync python scripts/critic_strictness.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from nss.generate.final_selection_figures import HUMAN_CHECK

TABLES = Path("reports/tables")
MARGINAL = 0.10  # pre-registered: within 10% (relative) of the threshold


def margin(value: float, threshold: float) -> dict[str, float | bool]:
    """Absolute and relative distance of a measured value from its threshold, and `marginal`."""
    rel = abs(value - threshold) / threshold
    return {
        "value": value,
        "threshold": threshold,
        "abs": value - threshold,
        "rel": rel,
        "marginal": rel <= MARGINAL,
    }


def diagnose(name: str, style: str, score: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per gate that failed for a scored concept."""
    rows: list[dict[str, Any]] = []
    human = HUMAN_CHECK[style]
    integ = score["integrity"]
    if integ["pass"] is False:
        m = margin(integ["closest_reference_dinov2"], integ["global_floor"])
        rows.append(
            {
                "concept": name,
                "gate": "integrity",
                **m,
                "evidence": "closest-reference DINOv2 similarity vs the global floor",
                "human_record": human,
            }
        )
    g2 = score["gate2"]
    if g2["pass"] is False:
        j = g2["judges"]["smolvlm"]
        rows.append(
            {
                "concept": name,
                "gate": "gate2",
                **margin(j["fidelity"], j["threshold"]),
                "evidence": f"SmolVLM reading {j['extraction']}",
                "human_record": human,
            }
        )
    g3 = score["gate3"]
    if g3["pass"] is False:
        answers = g3["answers"]
        rows.append(
            {
                "concept": name,
                "gate": "gate3",
                "value": answers.count("Y"),
                "threshold": len(answers) / 2,
                "abs": answers.count("Y") - len(answers) / 2,
                "rel": float("nan"),
                "marginal": False,
                "evidence": f"changes {g3['changes']} answered {answers} (Y=visible)",
                "human_record": human,
            }
        )
    return rows


def main() -> None:
    """Write `v3_critic_strictness.csv` and the candidate-level context table."""
    cache = {
        r["name"]: r
        for r in (
            json.loads(x)
            for x in (TABLES / "v3_agent_eval_scores.jsonl").read_text("utf-8").splitlines()
            if x
        )
    }
    rows = []
    for name in ("submitted_Top", "submitted_Bikini top"):
        rows += diagnose(name, cache[name]["style"], cache[name]["score"])
    pl.DataFrame(rows).write_csv(TABLES / "v3_critic_strictness.csv")
    cand = pl.read_csv(TABLES / "candidates_scored.csv")
    ctx = []
    for style in cand["style_id"].unique(maintain_order=True):
        s = cand.filter(pl.col("style_id") == style)
        ctx.append(
            {
                "style": style.split(" || ")[1],
                "candidates": s.height,
                "integrity_fails": int((~s["integrity_floor_pass"]).sum()),
                "gate2_fails": int((~s["gate2_pass"]).sum()),
                "gate3_fails": int((~s["gate3_pass"]).sum()),
                "gate3_YN": int((s["smolvlm_gate3_answers"] == "YN").sum()),
                "smolvlm_fidelity_min": float(s["smolvlm_fidelity"].min()),
                "smolvlm_fidelity_max": float(s["smolvlm_fidelity"].max()),
                "all_gates_pass": int(
                    s.select(
                        pl.all_horizontal(
                            "gate1_pass",
                            "gate1b_pass",
                            "integrity_floor_pass",
                            "gate2_pass",
                            "gate3_pass",
                        )
                    )
                    .to_series()
                    .sum()
                ),
            }
        )
    pl.DataFrame(ctx).write_csv(TABLES / "v3_critic_strictness_context.csv")
    for r in rows:
        print({k: r[k] for k in ("concept", "gate", "value", "threshold", "rel", "marginal")})


if __name__ == "__main__":
    main()
