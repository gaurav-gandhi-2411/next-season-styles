"""K3: which recorded retry decisions the new scale-direction rule would have changed.

Replays `agent_eval.next_attempt` (the K3 rule now written into `agents/critic.md`) over three
records of retry decisions: the scripted live run (`v3_agent_live_attempts.csv`, old rule: seed for
an integrity-only miss, otherwise scale down by 0.10 to a floor of 0.15), the fourth LLM run
(`v3_agent_llm_stream_run.jsonl`, where the LLM critics chose), and the J2 `adjust` reference over
the 115 H3 cases (old rule as above).

    uv run --no-sync python scripts/scale_rule_replay.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import polars as pl

from nss.generate import agent_eval as ae

TABLES = Path("reports/tables")


def scripted_run() -> list[dict[str, Any]]:
    """Transitions of the scripted live run."""
    df = pl.read_csv(TABLES / "v3_agent_live_attempts.csv")
    rows = []
    for style in df["style"].unique(maintain_order=True):
        s = df.filter(pl.col("style") == style).sort("attempt").to_dicts()
        for a, b in zip(s, s[1:], strict=False):
            failed = [g for g in (a["failing"] or "").split(",") if g and g != "gate1"]
            prev = (
                [g for g in (s[s.index(a) - 1]["failing"] or "").split(",") if g]
                if a["attempt"] > 1
                else []
            )
            old = (b["ip_adapter_scale"], b["seed"])
            rows.append(
                {
                    "source": "scripted live run",
                    "style": style.split(" || ")[1],
                    "from_attempt": a["attempt"],
                    "failing": ",".join(failed),
                    "used": (a["ip_adapter_scale"], a["seed"]),
                    "chosen": old,
                    "new_rule": ae.next_attempt(failed, a["ip_adapter_scale"], a["seed"], prev),
                }
            )
    return rows


def llm_run() -> list[dict[str, Any]]:
    """Transitions of the LLM-orchestrated run, from its event stream."""
    path = TABLES / "v3_agent_llm_stream_run.jsonl"
    call_id: dict[str, str] = {}
    per_style: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text("utf-8").splitlines():
        e = json.loads(line)
        if e.get("type") == "assistant":
            for b in e["message"]["content"]:
                if b["type"] == "tool_use" and b["name"].endswith("score_concept"):
                    call_id[b["id"]] = b["input"]["concept_path"]
        elif e.get("type") == "user" and isinstance(e.get("message", {}).get("content"), list):
            for b in e["message"]["content"]:
                if b.get("type") == "tool_result" and b.get("tool_use_id") in call_id:
                    txt = (
                        b["content"]
                        if isinstance(b["content"], str)
                        else " ".join(x.get("text", "") for x in b["content"])
                    )
                    v = re.search(r'"verdict":\s*"([^"]+)"', txt)
                    p = call_id[b["tool_use_id"]]
                    m = re.search(r"/([a-z\-_]+)/s([0-9.]+)_seed(\d+)\.png", p.replace("\\", "/"))
                    if not (v and m):
                        continue
                    failed = re.findall(r"failed ([a-z0-9, ]+)", v.group(1))
                    fl = [g.strip() for g in failed[0].split(",")] if failed else []
                    per_style.setdefault(m.group(1), []).append(
                        {
                            "scale": float(m.group(2)),
                            "seed": int(m.group(3)),
                            "failing": [g for g in fl if g != "gate1"],
                            "verdict": v.group(1),
                        }
                    )
    rows = []
    for style, seq in per_style.items():
        for i, (a, b) in enumerate(zip(seq, seq[1:], strict=False)):
            prev = seq[i - 1]["failing"] if i > 0 else []
            rows.append(
                {
                    "source": "LLM run (attempt 4)",
                    "style": style.split("_")[1],
                    "from_attempt": i + 1,
                    "failing": ",".join(a["failing"]),
                    "used": (a["scale"], a["seed"]),
                    "chosen": (b["scale"], b["seed"]),
                    "new_rule": ae.next_attempt(a["failing"], a["scale"], a["seed"], prev),
                }
            )
    return rows


def main() -> None:
    """Write the replay table and print the counts."""
    rows = [*scripted_run(), *llm_run()]
    for r in rows:
        r["changed"] = tuple(r["chosen"]) != tuple(r["new_rule"])
        r["used"], r["chosen"], r["new_rule"] = (str(r[k]) for k in ("used", "chosen", "new_rule"))
    df = pl.DataFrame(rows)
    df.write_csv(TABLES / "v3_scale_rule_replay.csv")
    print(
        df.select(
            "source", "style", "from_attempt", "failing", "used", "chosen", "new_rule", "changed"
        )
    )
    cases = pl.read_csv(TABLES / "v3_agent_eval_cases.csv").filter(pl.col("outcome") == "RETRY")
    changed = 0
    for r in cases.to_dicts():
        failed = [g for g in (r["failing"] or "").split(",") if g and g != "gate1"]
        old = "seed" if failed == ["integrity"] else "ip_adapter_scale"
        new = ae.route(ae.REJECT, 1, failed).adjust
        changed += old != new
    print(f"J2 adjust reference: {changed} of {cases.height} RETRY cases change parameter")


if __name__ == "__main__":
    main()
