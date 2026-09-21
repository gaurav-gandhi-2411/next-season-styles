"""J2: does an LLM critic+orchestrator route the 115 H3 cases the way the deterministic rule does?

Rule pre-registered in `reports/v3/PREREGISTRATION.md` (section J, commit bcd1357). Each case is
shown to a Claude model (headless Claude Code, Claude Max login, `ANTHROPIC_API_KEY` removed, no
tools, no MCP, no settings) as the `score_concept` result of a critic sub-agent; the model plays
the critic and the orchestrator for that one decision and answers with JSON. Three independent
runs per case. The reference is `critic_rule.decide` + `agent_eval.route` (the J3 rule).

    uv run --no-sync python scripts/agent_llm_eval.py run      # LLM calls (resumable)
    uv run --no-sync python scripts/agent_llm_eval.py analyse  # tables from the saved runs
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_eval import build_cases  # noqa: E402
from agent_llm import AGENTS_DIR, clean_env, parse_frontmatter  # noqa: E402

from nss.generate import agent_eval as ae  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "reports" / "tables"
RUNS = TABLES / "v3_llm_routing_runs.jsonl"
MODEL = "sonnet"
N_RUNS = 3
WORKERS = 4
CAP_ATTEMPT = 3
INSTRUCTION = """You are being evaluated on ONE routing decision. Play both roles for it: first the
critic (decide the verdict from the score_concept result below, under critic.md), then the
orchestrator (decide the next step, under orchestrator.md). You have no tools; do not ask
questions. Reply with exactly one JSON object and nothing else:
{"verdict": "REJECT" | "PASS_PENDING_HUMAN" | "INCONCLUSIVE",
 "outcome": "RETRY" | "FORWARD" | "FAILED" | "ESCALATE_INCONCLUSIVE",
 "next_agent": "concept-designer" | "forecaster" | null,
 "adjust": "seed" | "ip_adapter_scale" | null,
 "reason": "<one or two sentences>"}
`adjust` is the one parameter the critic sends back to concept-designer on a RETRY, else null."""


def system_prompt() -> str:
    """The bodies of critic.md and orchestrator.md plus the task instruction."""
    bodies = [
        parse_frontmatter((AGENTS_DIR / f"{n}.md").read_text(encoding="utf-8"))[1]
        for n in ("critic", "orchestrator")
    ]
    return "\n\n=====\n\n".join([*bodies, INSTRUCTION])


def case_view(case: ae.Case) -> dict[str, Any]:
    """The `score_concept` result as the model sees it (pass flags only, same schema for all)."""
    p = case.passes

    def gate(name: str) -> dict[str, Any]:
        out: dict[str, Any] = {"pass": p[name]}
        if p[name] is None:
            out["status"] = "not_run"
        return out

    g1b = {**gate("gate1b"), "clone_control_failed_as_required": case.clone_ok}
    return {
        "gate1": gate("gate1"),
        "gate1b": g1b,
        "integrity": gate("integrity"),
        "gate2": gate("gate2"),
        "gate3": gate("gate3"),
        "human_visual_check": {"required": True, "status": "not automated"},
    }


def user_prompt(case: ae.Case, attempt: int) -> str:
    """One case: the tool result, the attempt number and the history."""
    hist = (
        "This is the first attempt (retry count 0)."
        if attempt == 1
        else f"This is attempt {attempt} of 3 (retry count {attempt - 1}); attempts "
        f"1..{attempt - 1} of this concept request were each REJECTed."
    )
    return (
        f"Concept request: `{case.name}`. {hist}\n\n"
        f"score_concept result (include_fidelity=true):\n{json.dumps(case_view(case), indent=1)}"
    )


def reference(case: ae.Case, attempt: int) -> dict[str, Any]:
    """The deterministic rule's answer for a case at an attempt."""
    v = ae.decide(case.passes, case.clone_ok)
    r = ae.route(v, attempt, ae.failing(case.passes, clone_ok=case.clone_ok))
    return {"verdict": v, "outcome": r.outcome, "next_agent": r.next_agent, "adjust": r.adjust}


def call_model(system: str, user: str) -> dict[str, Any]:
    """One headless `claude -p` call; returns the parsed JSON reply or an error record."""
    cmd = [
        "claude",
        "-p",
        user,
        "--system-prompt",
        system,
        "--tools",
        "",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--model",
        MODEL,
        "--max-turns",
        "1",
        "--output-format",
        "json",
        "--no-session-persistence",
    ]
    proc = subprocess.run(  # noqa: S603 -- fixed argv, no shell
        cmd, capture_output=True, text=True, env=clean_env(), cwd=ROOT, check=False, timeout=300
    )
    try:
        env = json.loads(proc.stdout)
        text = env.get("result", "")
    except json.JSONDecodeError:
        return {"error": f"envelope: {proc.stdout[:200]!r} {proc.stderr[:200]!r}"}
    start = text.find("{")
    if start < 0:
        return {"error": f"no json: {text[:200]!r}", "raw": text}
    try:
        # strict=False: free-text fields may contain raw newlines or tabs; raw_decode stops at the
        # end of the first object, so trailing prose or a code fence is harmless
        reply, _end = json.JSONDecoder(strict=False).raw_decode(text[start:])
    except json.JSONDecodeError:
        return {"error": f"bad json: {text[:200]!r}", "raw": text}
    return {"reply": reply}


def tasks() -> list[tuple[str, int, int, ae.Case]]:
    """(key, attempt, run, case): the 115 cases at attempt 1, the 19 live ones also at attempt 3."""
    cases, _detail, _parity = build_cases()
    out = []
    for c in cases:
        for run in range(N_RUNS):
            out.append((f"{c.name}|1|{run}", 1, run, c))
            if not c.name.startswith("wrong_"):
                out.append((f"{c.name}|{CAP_ATTEMPT}|{run}", CAP_ATTEMPT, run, c))
    return out


def done_keys() -> set[str]:
    """Keys already saved with a reply or a definitive error."""
    if not RUNS.exists():
        return set()
    return {json.loads(x)["key"] for x in RUNS.read_text("utf-8").splitlines() if x.strip()}


def run_all() -> None:
    """Run every missing (case, attempt, run) through the model; append each result."""
    system = system_prompt()
    todo = [t for t in tasks() if t[0] not in done_keys()]
    print(f"{len(todo)} calls to make", flush=True)

    def one(t: tuple[str, int, int, ae.Case]) -> dict[str, Any]:
        key, attempt, run, case = t
        try:
            res = call_model(system, user_prompt(case, attempt))
        except subprocess.TimeoutExpired:
            res = {"error": "timeout"}
        return {
            "key": key,
            "case": case.name,
            "stratum": case.stratum,
            "attempt": attempt,
            "run": run,
            **res,
        }

    with ThreadPoolExecutor(WORKERS) as pool, RUNS.open("a", encoding="utf-8") as fh:
        for n, rec in enumerate(pool.map(one, todo), start=1):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if n % 20 == 0:
                print(f"{n}/{len(todo)}", flush=True)


def analyse() -> None:
    """Agreement with the rule and consistency across the 3 runs, per stratum."""
    cases, _d, _p = build_cases()
    by_name = {c.name: c for c in cases}
    recs = [json.loads(x) for x in RUNS.read_text("utf-8").splitlines() if x.strip()]
    rows = []
    for r in recs:
        c = by_name[r["case"]]
        ref = reference(c, r["attempt"])
        rep = r.get("reply") or {}
        got = {k: rep.get(k) for k in ("verdict", "outcome", "next_agent", "adjust")}
        rows.append(
            {
                "case": r["case"],
                "stratum": r["stratum"],
                "attempt": r["attempt"],
                "run": r["run"],
                "parsed": "reply" in r,
                **{f"ref_{k}": v for k, v in ref.items()},
                **{f"llm_{k}": v for k, v in got.items()},
                "verdict_ok": got["verdict"] == ref["verdict"],
                "outcome_ok": got["outcome"] == ref["outcome"],
                "hop_ok": got["next_agent"] == ref["next_agent"],
                "adjust_ok": got["adjust"] == ref["adjust"],
                "reason": rep.get("reason", r.get("error")),
            }
        )
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_csv(TABLES / "v3_llm_routing_runs.csv")
    per_case = []
    for (case, attempt), g in df.group_by(["case", "attempt"], maintain_order=True):
        sigs = [
            (a, b, c_, d)
            for a, b, c_, d in zip(
                g["llm_verdict"],
                g["llm_outcome"],
                g["llm_next_agent"],
                g["llm_adjust"],
                strict=True,
            )
        ]
        top = Counter(sigs).most_common(1)[0]
        ref0 = (
            g["ref_verdict"][0],
            g["ref_outcome"][0],
            g["ref_next_agent"][0],
            g["ref_adjust"][0],
        )
        per_case.append(
            {
                "case": case,
                "stratum": g["stratum"][0],
                "attempt": attempt,
                "n_runs": g.height,
                "consistent": len(set(sigs)) == 1,
                "majority": "|".join(map(str, top[0])),
                "majority_count": top[1],
                "majority_matches_rule": top[0] == ref0,
                "all_runs_match_rule": all(s == ref0 for s in sigs),
                "verdict_outcome_consistent": len({(s[0], s[1]) for s in sigs}) == 1,
                "verdict_outcome_all_match": all(s[:2] == ref0[:2] for s in sigs),
            }
        )
    pc = pl.DataFrame(per_case, infer_schema_length=None)
    pc.write_csv(TABLES / "v3_llm_routing_cases.csv")
    summ = []
    for attempt in (1, CAP_ATTEMPT):
        for stratum in ["ALL", *sorted(set(df["stratum"]))]:
            d = df.filter(pl.col("attempt") == attempt)
            p = pc.filter(pl.col("attempt") == attempt)
            if stratum != "ALL":
                d, p = (
                    d.filter(pl.col("stratum") == stratum),
                    p.filter(pl.col("stratum") == stratum),
                )
            if d.is_empty():
                continue
            n, k_case = d.height, p.height
            row = {
                "attempt": attempt,
                "stratum": stratum,
                "cases": k_case,
                "runs": n,
                "unparsed": int((~d["parsed"]).sum()),
            }
            for m in ("verdict_ok", "outcome_ok", "hop_ok"):
                k = int(d[m].sum())
                lo, hi = ae.wilson(k, n)
                row |= {m: k / n, m + "_lo": lo, m + "_hi": hi}
            rd = d.filter(pl.col("ref_adjust").is_not_null())
            row["adjust_ok_runs"] = int(rd["adjust_ok"].sum())
            row["adjust_runs"] = rd.height
            row["consistent_cases"] = int(p["consistent"].sum())
            row["verdict_outcome_consistent_cases"] = int(p["verdict_outcome_consistent"].sum())
            row["majority_matches_rule_cases"] = int(p["majority_matches_rule"].sum())
            row["all_runs_match_rule_cases"] = int(p["all_runs_match_rule"].sum())
            row["verdict_outcome_all_match_cases"] = int(p["verdict_outcome_all_match"].sum())
            summ.append(row)
    pl.DataFrame(summ, infer_schema_length=None).write_csv(TABLES / "v3_llm_routing_summary.csv")
    bad = df.filter(~(pl.col("verdict_ok") & pl.col("outcome_ok") & pl.col("hop_ok")))
    print(
        f"runs {df.height}, verdict/outcome/hop disagreements {bad.height}, "
        f"unparsed {int((~df['parsed']).sum())}"
    )


if __name__ == "__main__":
    {"run": run_all, "analyse": analyse}[sys.argv[1]]()
