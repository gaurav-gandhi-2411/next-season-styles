"""H3 4b/4c: critic decisions, routing and gate ablation over labelled cases.

Rule pre-registered in `reports/v3/PREREGISTRATION.md` (H3, commit 14a0ae7), committed before this
ran. Each case is scored once through the real MCP `score_concept` tool (CPU, over stdio, the
critic's allowlist enforced by the driver), cached in `reports/tables/v3_agent_eval_scores.jsonl`;
the 96 wrong-style rows use their stored gate columns plus a live parity sample of 6.

    uv run --no-sync python scripts/agent_eval.py score     # live scoring (slow, resumable)
    uv run --no-sync python scripts/agent_eval.py analyse   # tables + summary from the cache
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import polars as pl
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_demo import _call_tool, critic_replay, parse_allowlist  # noqa: E402

from nss.generate import agent_eval as ae  # noqa: E402
from nss.generate import concept_generation, final_concepts, final_registry, qc_gates  # noqa: E402
from nss.generate.final_selection_figures import ALL_SELECTED  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "reports" / "tables"
CACHE = TABLES / "v3_agent_eval_scores.jsonl"
CLONE_DIR = Path("data/generated/agent_eval")
UNDERWEAR = "Ladieswear || Underwear bottom || Under-, Nightwear || Red || Solid"
SWEATER = final_registry.SWEATER
DRESS = final_registry.DRESS
H3 = "data/generated/final_concepts_h3/ladieswear_underwear-bottom_under--nightwear_red_solid"
N1L = "data/generated/n1_levers/ladieswear_sweater_knitwear_beige_melange"
# integrity_labels.json names these M3 images under reports/concepts/, which no longer exist there;
# the same bytes (sha256 checked against commit 41f302c) are in data/generated/final_concepts_m3/.
M3A1_SWEATER = (
    "data/generated/final_concepts_m3/attempt1/ladieswear_sweater_knitwear_beige_melange_seed43.png"
)
M3A2_DRESS = (
    "data/generated/final_concepts_m3/attempt2/ladieswear_dress_dresses-ladies_red_solid_seed45.png"
)
PARITY_N, PARITY_SEED = 6, 42
# The pre-registered routing table: verdict -> (next agent, tool), written before `route()` was.
EXPECTED_HOP = {
    ae.PASS: ("forecaster", "forecast_concept"),
    ae.INCONCLUSIVE: (None, None),
}
STRATA_NEG = ("N1_malformed", "N2_swatch", "N3_clone")
NEG_SETS = {"hard(N1-N3)": STRATA_NEG, "all(N1-N4)": (*STRATA_NEG, "N4_wrong_style")}
POS_SETS = {"P1": ("P1_approved",), "P1+P2": ("P1_approved", "P2_coherent")}


def _changes(style: str) -> list[str]:
    return list(concept_generation.CHANGES[style]["applied_changes"])


def live_cases() -> list[dict[str, Any]]:
    """The cases scored live: N1, N2, N3, P1, P2 (N4 is stored, plus a live parity sample)."""
    cases: list[dict[str, Any]] = []

    def add(name: str, stratum: str, truth: bool, path: str, style: str, changes: Any) -> None:
        cases.append(
            {
                "name": name,
                "stratum": stratum,
                "truth": truth,
                "path": Path(path).as_posix(),
                "style": style,
                "changes": changes,
            }
        )

    for seed in (42, 44, 45):
        add(f"underwear_seed{seed}", "N1_malformed", False, f"{H3}_seed{seed}.png", UNDERWEAR, None)
    for f in ("concat-0.15_s43.png", "concat-style-only-1.0_s42.png"):
        add(f"swatch_{f[:-4]}", "N2_swatch", False, f"{N1L}/{f}", SWEATER, _changes(SWEATER))
    CLONE_DIR.mkdir(parents=True, exist_ok=True)
    for style in ALL_SELECTED:
        src = qc_gates.reference_paths_for_style(style)[0]
        dst = CLONE_DIR / f"clone_{final_concepts._slugify(style)}{src.suffix}"
        shutil.copyfile(src, dst)
        add(f"clone_{style.split(' || ')[1]}", "N3_clone", False, str(dst), style, _changes(style))
    for style, path in ALL_SELECTED.items():
        add(
            f"submitted_{style.split(' || ')[1]}",
            "P1_approved",
            True,
            str(path),
            style,
            _changes(style),
        )
    add("h3_underwear_seed43", "P2_coherent", True, f"{H3}_seed43.png", UNDERWEAR, None)
    for f in ("concat-0.25-nat_s43.png", "concat-0.35-nat_s43.png", "single-0.35-nat_s43.png"):
        add(
            f"clean_sweater_{f[:-4]}", "P2_coherent", True, f"{N1L}/{f}", SWEATER, _changes(SWEATER)
        )
    add(
        "m3_sweater",
        "P2_coherent",
        True,
        M3A1_SWEATER,
        SWEATER,
        _changes(SWEATER),
    )
    add(
        "m3_dress",
        "P2_coherent",
        True,
        M3A2_DRESS,
        DRESS,
        _changes(DRESS),
    )
    return cases


def stored_wrong_style() -> list[dict[str, Any]]:
    """The 96 wrong-style rows with their stored gate columns."""
    rows = pl.read_csv(TABLES / "v3_yield_floor_scored.csv").to_dicts()
    return [
        {
            "name": f"wrong_{i:02d}",
            "stratum": "N4_wrong_style",
            "truth": False,
            "path": Path(r["image_path"]).as_posix(),
            "style": r["style_id"],
            "changes": _changes(r["style_id"]),
            "row": r,
        }
        for i, r in enumerate(rows)
    ]


def parity_indices() -> list[int]:
    """The pre-registered live parity sample of the 96 stored rows."""
    return sorted(random.Random(PARITY_SEED).sample(range(96), PARITY_N))


def read_cache() -> dict[str, dict[str, Any]]:
    """Scored cases so far, by name."""
    if not CACHE.exists():
        return {}
    return {
        r["name"]: r for r in (json.loads(x) for x in CACHE.read_text("utf-8").splitlines() if x)
    }


async def score_live() -> None:
    """Score every live case (and the parity sample) through the MCP server; resumable."""
    allow = {p.stem: parse_allowlist(p.read_text("utf-8")) for p in (ROOT / "agents").glob("*.md")}
    wrong = stored_wrong_style()
    todo = [*live_cases(), *(wrong[i] for i in parity_indices())]
    done = read_cache()
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "nss.mcp_server"],
        cwd=str(ROOT),
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"},
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        for case in todo:
            if case["name"] in done:
                continue
            args: dict[str, Any] = {
                "concept_path": case["path"],
                "style_key": case["style"],
                "include_fidelity": True,
            }
            if case["changes"] is not None:
                args["changes"] = case["changes"]
            t0 = time.time()
            score, _ = await _call_tool(session, allow, "critic", "score_concept", args)
            rec = {
                "name": case["name"],
                "path": case["path"],
                "style": case["style"],
                "seconds": round(time.time() - t0, 1),
                "score": score,
            }
            with CACHE.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"{case['name']}: {rec['seconds']}s {score['verdict'][:60]}", flush=True)


def build_cases() -> tuple[list[ae.Case], list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluation cases with gate results, the per-case detail rows and the parity comparison."""
    cache = read_cache()
    parity = set(parity_indices())
    cases: list[ae.Case] = []
    detail: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    wrong = stored_wrong_style()
    for i, c in enumerate(wrong):
        passes, clone_ok = ae.passes_from_row(c["row"])
        source = "stored"
        if i in parity:
            live_p, live_clone = ae.passes_from_score(cache[c["name"]]["score"])
            parity_rows.append(
                {
                    "name": c["name"],
                    "agree": live_p == passes and live_clone == clone_ok,
                    **{f"stored_{g}": passes[g] for g in ae.GATES},
                    **{f"live_{g}": live_p[g] for g in ae.GATES},
                }
            )
            passes, clone_ok, source = live_p, live_clone, "live (parity sample)"
        cases.append(ae.Case(c["name"], c["stratum"], False, passes, clone_ok))
        detail.append({"name": c["name"], "source": source})
    for c in live_cases():
        passes, clone_ok = ae.passes_from_score(cache[c["name"]]["score"])
        cases.append(ae.Case(c["name"], c["stratum"], c["truth"], passes, clone_ok))
        detail.append({"name": c["name"], "source": "live"})
    return cases, detail, parity_rows


def routing_checks(cases: list[ae.Case]) -> list[dict[str, Any]]:
    """Checks (a)-(d) of the pre-registered routing rule."""
    out: list[dict[str, Any]] = []
    # (a) route() against the pre-registered table, for every case's actual verdict
    bad = []
    for c in cases:
        v = ae.decide(c.passes, c.clone_ok)
        r = ae.route(v, 1, ae.failing(c.passes))
        if v == ae.REJECT:
            ok = (r.next_agent, r.tool) == ("concept-designer", "generate_concept")
            ok = ok and r.adjust == (
                "seed" if ae.failing(c.passes) == ["integrity"] else "ip_adapter_scale"
            )
        else:
            ok = (r.next_agent, r.tool) == EXPECTED_HOP[v]
        if not ok:
            bad.append(c.name)
    out.append(
        {
            "check": "a: route() == pre-registered hop, every case",
            "n": len(cases),
            "failures": len(bad),
            "detail": ";".join(bad),
        }
    )
    # (b) the retry cap on the three N1 images in seed order, read literally (INCONCLUSIVE when a
    # gate is null) and with a definite failure deciding (`reject_first`)
    n1 = [c for c in cases if c.stratum == "N1_malformed"]
    for reject_first in (False, True):
        hops, rows, wrong = [], [], 0
        for attempt, c in enumerate(n1, start=1):
            v = ae.decide(c.passes, c.clone_ok, reject_first=reject_first)
            r = ae.route(v, attempt, ae.failing(c.passes))
            hops.append(f"{attempt}:{v}->{r.outcome}")
            rows.append({col: c.passes[g] for g, col in ae._ROW_COLUMNS.items()})
            if v == ae.REJECT:
                wrong += r.outcome != ("RETRY" if attempt <= ae.RETRY_CAP else "FAILED")
            if v != ae.REJECT or r.outcome == "FAILED":
                break
        replay = critic_replay(rows)[1]
        out.append(
            {
                "check": "b: retry cap over N1 seeds 42,44,45 "
                + ("(reject_first)" if reject_first else "(literal critic.md)"),
                "n": len(n1),
                "failures": wrong,
                "detail": f"{' | '.join(hops)}; driver critic_replay: {replay}",
            }
        )

    # (c) fail-closed unit cases with a stubbed score_concept
    def stub_ok(**over: Any) -> dict[str, Any]:
        s: dict[str, Any] = {g: {"pass": True} for g in ae.GATES}
        s["gate1b"]["clone_control_failed_as_required"] = True
        s.update(over)
        return s

    def raises() -> dict[str, Any]:
        raise RuntimeError("stub")

    n = {"k": 0}

    def once() -> dict[str, Any]:
        n["k"] += 1
        if n["k"] == 1:
            raise RuntimeError("stub")
        return stub_ok()

    unit = {
        "raises twice": (ae.critic_verdict(raises), (ae.INCONCLUSIVE, [], 2)),
        "raises once, recovers": (ae.critic_verdict(once), (ae.PASS, [], 2)),
        "gate2 not_run": (
            ae.critic_verdict(lambda: stub_ok(gate2={"pass": None})),
            (ae.INCONCLUSIVE, [], 1),
        ),
        "gate3 pass null": (
            ae.critic_verdict(lambda: stub_ok(gate3={"pass": None})),
            (ae.INCONCLUSIVE, [], 1),
        ),
        "clone control not validated": (
            ae.critic_verdict(
                lambda: stub_ok(gate1b={"pass": False, "clone_control_failed_as_required": False})
            ),
            (ae.INCONCLUSIVE, [], 1),
        ),
    }
    for name, (got, want) in unit.items():
        out.append(
            {
                "check": f"c: {name}",
                "n": 1,
                "failures": int((got[0], got[2]) != (want[0], want[2])),  # verdict, scoring calls
                "detail": f"got {got} want verdict/calls {want[0]}/{want[2]}",
            }
        )
    # (d) allowlists from agents/*.md
    allow = {p.stem: parse_allowlist(p.read_text("utf-8")) for p in (ROOT / "agents").glob("*.md")}
    for agent, tool, must in (
        ("concept-designer", "generate_concept", True),
        ("forecaster", "forecast_concept", True),
        ("critic", "generate_concept", False),
        ("critic", "forecast_concept", False),
        ("critic", "forecast_styles", False),
    ):
        ok = (tool in allow[agent]) == must
        out.append(
            {
                "check": f"d: {agent} {'lists' if must else 'omits'} {tool}",
                "n": 1,
                "failures": int(not ok),
                "detail": str(sorted(allow[agent])),
            }
        )
    return out


def metrics_rows(cases: list[ae.Case]) -> list[dict[str, Any]]:
    """Precision/recall per positive x negative set, with floors and with each gate removed."""
    rows = []
    for pname, pstrata in POS_SETS.items():
        for nname, nstrata in NEG_SETS.items():
            sub = [c for c in cases if c.stratum in (*pstrata, *nstrata)]
            npos = sum(c.truth for c in sub)
            prev = npos / len(sub)
            for removed in (None, *ae.GATES):
                mask = frozenset() if removed is None else frozenset({removed})
                conf = ae.confusion(sub, mask)
                pr = ae.precision_recall(conf)
                rows.append(
                    {
                        "positives": pname,
                        "negatives": nname,
                        "gate_removed": removed or "none",
                        "n_pos": npos,
                        "n_neg": len(sub) - npos,
                        **{k: v for k, v in pr.items() if not k.endswith("_ci")},
                        "precision_lo": pr["precision_ci"][0],
                        "precision_hi": pr["precision_ci"][1],
                        "recall_lo": pr["recall_ci"][0],
                        "recall_hi": pr["recall_ci"][1],
                        "floor_precision_prevalence": prev,
                        "floor_recall_random": prev,
                        "inconclusive": conf["inconclusive_pos"] + conf["inconclusive_neg"],
                    }
                )
    return rows


def main(mode: str) -> None:
    """`score` (live, resumable) or `analyse` (tables from the cache)."""
    if mode == "score":
        asyncio.run(score_live())
        return
    cases, detail, parity = build_cases()
    routing = routing_checks(cases)
    metrics = metrics_rows(cases)
    abl = ae.ablate(cases)
    case_rows = []
    for c, d in zip(cases, detail, strict=True):
        v = ae.decide(c.passes, c.clone_ok)
        r = ae.route(v, 1, ae.failing(c.passes))
        case_rows.append(
            {
                "name": c.name,
                "stratum": c.stratum,
                "truth_accept": c.truth,
                "source": d["source"],
                **{g: c.passes[g] for g in ae.GATES},
                "clone_control_ok": c.clone_ok,
                "verdict": v,
                "failing": ",".join(ae.failing(c.passes)),
                "next_agent": r.next_agent,
                "adjust": r.adjust,
                "outcome": r.outcome,
                "verdict_reject_first": ae.decide(c.passes, c.clone_ok, reject_first=True),
            }
        )
    for name, rows in (
        ("cases", case_rows),
        ("metrics", metrics),
        ("ablation", abl),
        ("routing", routing),
        ("parity", parity),
    ):
        pl.DataFrame(rows, infer_schema_length=None).write_csv(TABLES / f"v3_agent_eval_{name}.csv")
    print(
        json.dumps(
            {
                "parity_disagreements": sum(not p["agree"] for p in parity),
                "routing_failures": sum(r["failures"] for r in routing),
            }
        )
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "analyse")
