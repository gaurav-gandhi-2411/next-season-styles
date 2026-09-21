"""K4/K5: what the Gate 2 identity constraints and an advisory Gate 1 change, offline.

Rule pre-registered in `reports/v3/PREREGISTRATION.md` (section K, commit f14d7a2). Re-derives
Gate 2 for every stored case from the judge's STORED reading (no judge is re-run): the new Gate 2 is
the old pass AND product type OK AND colour OK under `nss.generate.identity_match`. Also counts the
verdicts that change when Gate 1 stops gating (expected: none).

    uv run --no-sync python scripts/gate2_identity_rescore.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_eval import build_cases, read_cache, stored_wrong_style  # noqa: E402

from nss.generate import agent_eval as ae  # noqa: E402
from nss.generate import critic_rule, identity_match  # noqa: E402
from nss.generate.concept_qc_pipeline import parse_style_attributes  # noqa: E402

TABLES = Path("reports/tables")


def _identity(style: str, extraction: dict[str, str]) -> tuple[bool, bool]:
    truth = parse_style_attributes(style)
    return (
        identity_match.product_type_ok(extraction.get("product_type", ""), truth["product_type"]),
        identity_match.colour_ok(extraction.get("colour_family", ""), truth["colour_family"]),
    )


def _rows() -> list[dict[str, Any]]:
    """Every stored case with its style, SmolVLM reading and old Gate 2 result."""
    cache = read_cache()
    rows: list[dict[str, Any]] = []
    base, _d, _p = build_cases()
    wrong = {c["name"]: c for c in stored_wrong_style()}
    for c in base:
        if c.name in wrong:
            r = wrong[c.name]["row"]
            style, reading, old = (
                r["style_id"],
                json.loads(r["smolvlm_extraction"]),
                c.passes["gate2"],
            )
        else:
            rec = cache[c.name]
            style = rec["style"]
            smol = rec["score"]["gate2"]["judges"]["smolvlm"]
            reading, old = json.loads(smol["extraction"]), c.passes["gate2"]
        rows.append(
            {
                "name": c.name,
                "group": c.stratum,
                "style": style,
                "reading": reading,
                "old_gate2": old,
                "passes": c.passes,
                "clone_ok": c.clone_ok,
            }
        )
    for line in (TABLES / "v3_hard_negatives_scores.jsonl").read_text("utf-8").splitlines():
        r = json.loads(line)
        passes, clone_ok = ae.passes_from_score(r["score"])
        smol = r["score"]["gate2"]["judges"]["smolvlm"]
        rows.append(
            {
                "name": r["name"],
                "group": r["cls"],
                "style": r["style"],
                "reading": json.loads(smol["extraction"]),
                "old_gate2": passes["gate2"],
                "passes": passes,
                "clone_ok": clone_ok,
            }
        )
    return rows


def main() -> None:
    """Write the per-case table and print the effect summary."""
    out = []
    changes = []
    gate1_changes = 0
    for r in _rows():
        prod_ok, col_ok = _identity(r["style"], r["reading"])
        new_g2 = bool(r["old_gate2"]) and prod_ok and col_ok
        passes_new = {**r["passes"], "gate2": new_g2}
        old_decision = "REJECT" if r["passes"].get("gate1") is False else None
        v_old = old_decision or critic_rule.decide(r["passes"], r["clone_ok"])
        v_new = critic_rule.decide(passes_new, r["clone_ok"])
        gate1_changes += r["passes"].get("gate1") is False
        why = []
        if r["old_gate2"] and not prod_ok:
            why.append("product_type")
        if r["old_gate2"] and not col_ok:
            why.append("colour")
        out.append(
            {
                "name": r["name"],
                "group": r["group"],
                "style": r["style"].split(" || ")[1],
                "product_reading": r["reading"].get("product_type"),
                "colour_reading": r["reading"].get("colour_family"),
                "old_gate2": r["old_gate2"],
                "product_ok": prod_ok,
                "colour_ok": col_ok,
                "new_gate2": new_g2,
                "gate2_changed": bool(r["old_gate2"]) != new_g2,
                "changed_by": "+".join(why),
                "verdict_old": v_old,
                "verdict_new": v_new,
            }
        )
        if v_old != v_new:
            changes.append((r["name"], v_old, v_new))
    df = pl.DataFrame(out)
    df.write_csv(TABLES / "v3_gate2_identity_rescore.csv")
    cand = pl.read_csv(TABLES / "candidates_scored.csv")
    crows = []
    for c in cand.to_dicts():
        prod_ok, col_ok = _identity(c["style_id"], json.loads(c["smolvlm_extraction"]))
        crows.append(
            {
                "style": c["style_id"].split(" || ")[1],
                "image": Path(c["image_path"]).name,
                "scale": c["scale"],
                "product_reading": json.loads(c["smolvlm_extraction"])["product_type"],
                "colour_reading": json.loads(c["smolvlm_extraction"])["colour_family"],
                "old_gate2": c["gate2_pass"],
                "product_ok": prod_ok,
                "colour_ok": col_ok,
                "new_gate2": bool(c["gate2_pass"]) and prod_ok and col_ok,
            }
        )
    pl.DataFrame(crows).write_csv(TABLES / "v3_gate2_identity_rescore_candidates.csv")
    print(
        json.dumps(
            {
                "cases": df.height,
                "gate2_changed": int(df["gate2_changed"].sum()),
                "verdict_changes_k4_plus_k5": len(changes),
                "gate1_failures_in_cases": int(gate1_changes),
            }
        )
    )
    for ch in changes:
        print("verdict change:", ch)


if __name__ == "__main__":
    main()
