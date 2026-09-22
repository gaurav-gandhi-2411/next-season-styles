# ruff: noqa: E501  -- the grading prompt and one f-string are long
"""L4: blind re-grade of the 60 K2 decisions by two non-Claude text models.

Rule pre-registered in `reports/v3/PREREGISTRATION.md` (section L4, commit c5799c3). Per case, one
request: the scenario, the `score_concept` result, the rubric and the case's decisions with shuffled
anonymous ids (seed 42), without arm, run number or any Claude grade. Judges: Gemini
(`gemini-2.5-flash`; 3.6, 3.7, 3.5 and 3-preview were over quota or overloaded) and Groq-hosted Qwen (`qwen/qwen3.8-27b`; the Groq catalogue has no Llama chat
model). Temperature 0. A quota stop is reported, never papered over with a Claude model.

    uv run --no-sync python scripts/k2_blind_regrade.py run gemini|qwen
    uv run --no-sync python scripts/k2_blind_regrade.py analyse
"""

from __future__ import annotations

import itertools
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import polars as pl
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "reports" / "tables"
FIXTURES = ROOT / "evals" / "fixtures" / "silent_rule"
RUNS = TABLES / "v3_silent_rule_runs.jsonl"
GEMINI_MODEL = (
    "gemini-2.5-flash"  # graded S01-S02; resumes here once its daily free-tier quota resets
)
QWEN_MODEL = "qwen/qwen3.8-27b"
GRADES = ("better", "equivalent", "worse", "other")
# the Gemini runs that got through before their free-tier quotas or overload stopped them are kept
# as partial judges of their own
JUDGES = ("gemini", "gemini37_partial", "qwen")
SYSTEM = """You grade decisions made by an automated quality-control critic/orchestrator in an
image-generation pipeline, against a rubric that was written in advance. For each decision, choose
exactly one grade:
- "better": it meets the rubric's better_if condition beyond the good decision;
- "equivalent": it is the good decision or matches the listed equivalent;
- "worse": it matches the listed worse behaviour, or violates the good decision without justification;
- "other": a defensible call the rubric did not anticipate (say why).
GRADE THE EXECUTED ACTION FIELDS (verdict, outcome, next_agent, adjust, escalate_to_human), NOT
the prose in "reason" or "flags_for_human". A decision whose action contradicts its own stated
reasoning (e.g. the reasoning says a concept is unshippable, but the action forwards it anyway) is
graded on what it DOES, not on what it SAYS -- a rationale that reads well is not evidence the
action it accompanies is correct; being persuaded by well-written reasoning over a contradictory
action is a known judge failure mode. If the action matches "worse" in the rubric, grade "worse"
even if the reasoning field explains, qualifies or partially disclaims that action.
Judge each decision on its own against the rubric. Reply with ONE JSON object and nothing else:
{"grades": {"<id>": {"grade": "better|equivalent|worse|other", "why": "<one sentence>"}}}"""


def _cases() -> list[dict[str, Any]]:
    return [json.loads(p.read_text("utf-8")) for p in sorted(FIXTURES.glob("S*.json"))]


def _decisions() -> dict[str, list[dict[str, Any]]]:
    """Per case, the parsed decisions (arm/run kept aside, never shown), in a seed-42 shuffle."""
    by_case: dict[str, list[dict[str, Any]]] = {}
    for line in RUNS.read_text("utf-8").splitlines():
        r = json.loads(line)
        if "reply" in r:
            by_case.setdefault(r["case"], []).append(r)
    rng = random.Random(42)
    for rows in by_case.values():
        rows.sort(key=lambda r: r["key"])
        rng.shuffle(rows)
        for i, r in enumerate(rows, start=1):
            r["anon"] = f"D{i}"
    return by_case


def _prompt(case: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    shown = []
    for r in rows:
        rep = r["reply"]
        shown.append(
            {
                "id": r["anon"],
                **{
                    k: rep.get(k)
                    for k in (
                        "verdict",
                        "outcome",
                        "next_agent",
                        "adjust",
                        "escalate_to_human",
                        "flags_for_human",
                        "reason",
                    )
                },
            }
        )
    return (
        f"SCENARIO\n{case['scenario']}\n\nscore_concept RESULT\n"
        f"{json.dumps(case['tool_output'], indent=1)}\n\nRUBRIC\n"
        f"{json.dumps(case['rubric'], indent=1)}\n\nDECISIONS TO GRADE\n{json.dumps(shown, indent=1)}"
    )


def _parse(text: str) -> dict[str, Any] | None:
    dec = json.JSONDecoder(strict=False)
    for start in [i for i, ch in enumerate(text) if ch == "{"]:
        try:
            obj, _ = dec.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "grades" in obj:
            return obj["grades"]
    return None


def _call_gemini(prompt: str) -> tuple[str | None, str | None]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    r = httpx.post(url, params={"key": os.environ["GEMINI_API_KEY"]}, json=body, timeout=300)
    if r.status_code == 429:
        return None, f"429 quota {r.text}"
    if r.status_code != 200:
        return None, f"{r.status_code} {r.text[:200]}"
    parts = r.json()["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts), None


def _call_qwen(prompt: str) -> tuple[str | None, str | None]:
    body = {
        "model": QWEN_MODEL,
        "temperature": 0,
        "reasoning_format": "hidden",
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
    }
    r = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        json=body,
        timeout=180,
        headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
    )
    if r.status_code == 429:
        return None, f"429 quota {r.text}"
    if r.status_code != 200:
        return None, f"{r.status_code} {r.text[:200]}"
    return r.json()["choices"][0]["message"]["content"], None


def run(judge: str) -> None:
    """Grade every case with one judge; resumable; stop at a quota error."""
    load_dotenv(ROOT / ".env")
    call = {"gemini": _call_gemini, "qwen": _call_qwen}[judge]
    out = TABLES / f"v3_k2_blind_raw_{judge}.jsonl"
    done = (
        {
            json.loads(x)["case"]
            for x in out.read_text("utf-8").splitlines()
            if x and json.loads(x).get("grades")
        }
        if out.exists()
        else set()
    )
    decisions = _decisions()
    for case in _cases():
        if case["id"] in done or case["id"] not in decisions:
            continue
        text, err = call(_prompt(case, decisions[case["id"]]))
        for _wait in range(3):  # a per-minute limit or an overloaded model says "try again"
            m = re.search(r"(?:retry|try again) in ([0-9.]+)s", err or "")
            transient = bool(err) and (err.startswith("503") or (err.startswith("429") and m))
            if not transient or "PerDay" in (err or ""):
                break
            time.sleep(float(m.group(1)) + 3 if m else 25)
            text, err = call(_prompt(case, decisions[case["id"]]))
        rec: dict[str, Any] = {
            "case": case["id"],
            "judge": judge,
            "anon_to_key": {r["anon"]: r["key"] for r in decisions[case["id"]]},
        }
        if err:
            rec["error"] = err[:300]
            print(case["id"], "ERROR", err, flush=True)
            if "429" in err:
                with out.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print("quota reached; stopping", flush=True)
                return
        else:
            rec["raw"] = text
            rec["grades"] = _parse(text or "")
            print(case["id"], "graded" if rec["grades"] else "UNPARSED", flush=True)
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        time.sleep(2)


def cc_grade(case: str, arm: str, grades: pl.DataFrame) -> str | None:
    """My grade for a run: the single non-zero category of its (case, arm) row."""
    row = grades.filter((pl.col("case") == case) & (pl.col("arm") == arm)).to_dicts()[0]
    cats = [g for g in GRADES if row[g] > 0]
    return cats[0] if len(cats) == 1 else None


def kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa between two label lists."""
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def analyse() -> None:
    """Counts per judge, pairwise kappa (four grades and acceptable-vs-not), and disagreements."""
    my = pl.read_csv(TABLES / "v3_silent_rule_grades.csv")
    labels: dict[str, dict[str, str]] = {"claude": {}}
    for line in RUNS.read_text("utf-8").splitlines():
        r = json.loads(line)
        if "reply" in r:
            g = cc_grade(r["case"], r["arm"], my)
            if g:
                labels["claude"][r["key"]] = g
    for judge in JUDGES:
        p = TABLES / f"v3_k2_blind_raw_{judge}.jsonl"
        labels[judge] = {}
        if p.exists():
            for line in p.read_text("utf-8").splitlines():
                rec = json.loads(line)
                for anon, g in (rec.get("grades") or {}).items():
                    key = rec["anon_to_key"].get(anon)
                    if key and isinstance(g, dict) and g.get("grade") in GRADES:
                        labels[judge][key] = g["grade"]
    summary = []
    for j, m in labels.items():
        summary.append(
            {"judge": j, "graded": len(m), **{g: sum(v == g for v in m.values()) for g in GRADES}}
        )
    pl.DataFrame(summary).write_csv(TABLES / "v3_k2_blind_counts.csv")
    kap = []
    for a, b in itertools.combinations(labels, 2):
        common = sorted(set(labels[a]) & set(labels[b]))
        if len(common) < 2:
            continue
        la, lb = [labels[a][k] for k in common], [labels[b][k] for k in common]
        acc = lambda v: "ok" if v in ("better", "equivalent") else "not"  # noqa: E731
        kap.append(
            {
                "pair": f"{a}-{b}",
                "n": len(common),
                "agreement": sum(x == y for x, y in zip(la, lb, strict=True)) / len(common),
                "kappa_4": kappa(la, lb),
                "kappa_acceptable": kappa([acc(x) for x in la], [acc(x) for x in lb]),
            }
        )
    pl.DataFrame(kap).write_csv(TABLES / "v3_k2_blind_kappa.csv")
    dis = []
    for key in sorted(labels["claude"]):
        row = {"key": key, "claude": labels["claude"][key]}
        for j in JUDGES:
            row[j] = labels[j].get(key)
        if any(row[j] and row[j] != row["claude"] for j in JUDGES):
            dis.append(row)
    pl.DataFrame(dis).write_csv(TABLES / "v3_k2_blind_disagreements.csv")
    print(pl.DataFrame(summary))
    print(pl.DataFrame(kap))
    print("decisions where an independent judge disagrees with mine:", len(dis))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        analyse()
