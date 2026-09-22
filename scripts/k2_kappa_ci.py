"""N5 addendum: bootstrap 95% CIs for the N2/M4 pairwise kappa figures, and Qwen's dissenting
cases with each grader's reasoning.

Not a re-score: reuses the same `labels` dict k2_blind_regrade.py `analyse()` computes (imported,
not recomputed by a different method). LLM consensus, not human ground truth -- see N2/M4.

    uv run --no-sync python scripts/k2_kappa_ci.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k2_blind_regrade import GRADES, JUDGES, RUNS, TABLES, cc_grade, kappa  # noqa: E402

FULL_JUDGES = ("claude", "gemini", "qwen")
N_BOOT = 2000
SEED = 42


def _acc(v: str) -> str:
    return "ok" if v in ("better", "equivalent") else "not"


def _labels_and_reasons() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    my = pl.read_csv(TABLES / "v3_silent_rule_grades.csv")
    my_notes = {(r["case"], r["arm"]): r["note"] for r in my.to_dicts()}
    labels: dict[str, dict[str, str]] = {"claude": {}}
    reasons: dict[str, dict[str, str]] = {"claude": {}}
    for line in RUNS.read_text("utf-8").splitlines():
        r = json.loads(line)
        if "reply" in r:
            g = cc_grade(r["case"], r["arm"], my)
            if g:
                labels["claude"][r["key"]] = g
                reasons["claude"][r["key"]] = my_notes.get((r["case"], r["arm"]), "")
    for judge in JUDGES:
        p = TABLES / f"v3_k2_blind_raw_{judge}.jsonl"
        labels[judge] = {}
        reasons[judge] = {}
        if p.exists():
            for line in p.read_text("utf-8").splitlines():
                rec = json.loads(line)
                for anon, g in (rec.get("grades") or {}).items():
                    key = rec["anon_to_key"].get(anon)
                    if key and isinstance(g, dict) and g.get("grade") in GRADES:
                        labels[judge][key] = g["grade"]
                        reasons[judge][key] = g.get("why", "")
    return labels, reasons


def _bootstrap_kappa_ci(la: list[str], lb: list[str]) -> tuple[float, float, float]:
    """Point kappa and a percentile bootstrap 95% CI, resampling matched (la[i], lb[i]) pairs."""
    n = len(la)
    point = kappa(la, lb)
    rng = random.Random(SEED)
    boots = []
    for _ in range(N_BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(kappa([la[i] for i in idx], [lb[i] for i in idx]))
    boots.sort()
    lo = boots[int(0.025 * N_BOOT)]
    hi = boots[min(int(0.975 * N_BOOT), N_BOOT - 1)]
    return point, lo, hi


def _clopper_pearson(x: int, n: int) -> tuple[float, float]:
    """Exact 95% CI on a raw proportion `x / n` (Clopper-Pearson, inverting the binomial)."""
    lo = float(stats.beta.ppf(0.025, x, n - x + 1)) if x > 0 else 0.0
    hi = float(stats.beta.ppf(0.975, x + 1, n - x)) if x < n else 1.0
    return lo, hi


def _pair_ci(a: str, b: str, labels: dict[str, dict[str, str]]) -> dict[str, object]:
    """A1: kappa's percentile bootstrap CI collapses to a point when raw agreement is 100% (no
    disagreement in the sample to resample), which reports certainty the estimate does not have.
    For a 100%-agreement pair, report the exact Clopper-Pearson CI on raw agreement instead; the
    bootstrap kappa CI is kept where kappa is not degenerate (agreement < 100%)."""
    common = sorted(set(labels[a]) & set(labels[b]))
    la = [_acc(labels[a][k]) for k in common]
    lb = [_acc(labels[b][k]) for k in common]
    n = len(common)
    n_agree = sum(x == y for x, y in zip(la, lb, strict=True))
    if n_agree == n:
        lo, hi = _clopper_pearson(n_agree, n)
        return {
            "pair": f"{a}-{b}",
            "n": n,
            "method": "clopper_pearson_raw_agreement",
            "point": n_agree / n,
            "ci_lo": lo,
            "ci_hi": hi,
            "note": "kappa=1.0 (59/59 raw agreement); bootstrap kappa CI is degenerate here "
            "(no disagreement to resample), so this is an exact CI on raw agreement, not on kappa",
        }
    point, lo, hi = _bootstrap_kappa_ci(la, lb)
    return {
        "pair": f"{a}-{b}",
        "n": n,
        "method": "bootstrap_kappa",
        "point": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "note": f"{N_BOOT}-resample percentile bootstrap of Cohen's kappa, seed {SEED}",
    }


def main() -> None:
    labels, reasons = _labels_and_reasons()
    pairs = (("claude", "gemini"), ("claude", "qwen"), ("gemini", "qwen"))
    rows = [_pair_ci(a, b, labels) for a, b in pairs]
    ci_df = pl.DataFrame(rows)
    ci_df.write_csv(f"{TABLES}/v3_k2_kappa_ci.csv")
    with pl.Config(tbl_cols=-1, fmt_str_lengths=120):
        print(ci_df)

    # Qwen's dissenting cases: acc(qwen) != acc(claude) AND acc(qwen) != acc(gemini)
    common_cq = sorted(set(labels["claude"]) & set(labels["qwen"]))
    common_gq = sorted(set(labels["gemini"]) & set(labels["qwen"]))
    dissent_cq = {k for k in common_cq if _acc(labels["claude"][k]) != _acc(labels["qwen"][k])}
    dissent_gq = {k for k in common_gq if _acc(labels["gemini"][k]) != _acc(labels["qwen"][k])}
    both = sorted(dissent_cq & dissent_gq)
    print(f"\nQwen dissents from BOTH claude and gemini on {len(both)} case(s): {both}")

    runs_by_key = {}
    for line in RUNS.read_text("utf-8").splitlines():
        r = json.loads(line)
        if "reply" in r:
            runs_by_key[r["key"]] = r["reply"]

    detail_rows = []
    for key in both:
        row = {
            "key": key,
            "claude_grade": labels["claude"][key],
            "claude_reason": reasons["claude"][key],
            "gemini_grade": labels["gemini"].get(key),
            "gemini_reason": reasons["gemini"].get(key, ""),
            "qwen_grade": labels["qwen"][key],
            "qwen_reason": reasons["qwen"][key],
            "orchestrator_reply": json.dumps(runs_by_key.get(key, {})),
        }
        detail_rows.append(row)
        print(f"\n--- {key} ---")
        print("orchestrator reply:", runs_by_key.get(key, {}))
        print("claude:", row["claude_grade"], "|", row["claude_reason"])
        print("gemini:", row["gemini_grade"], "|", row["gemini_reason"])
        print("qwen:  ", row["qwen_grade"], "|", row["qwen_reason"])
    pl.DataFrame(detail_rows).write_csv(f"{TABLES}/v3_k2_qwen_dissent_detail.csv")


if __name__ == "__main__":
    main()
