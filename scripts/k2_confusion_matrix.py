"""R6: confusion matrices and the "not acceptable" base rate behind the N2/M4 kappa figures.

Kappa is unstable when one class is rare (a large chance-agreement term `pe` can make even a
near-perfect raw agreement look artificially high, or conceal how much of it is driven by one
dominant class). This does not recompute anything `k2_blind_regrade.py analyse` already computed
(same `labels` construction, reused verbatim); it just reports what is behind the three pairwise
`kappa_acceptable` numbers on the full-59 graders (claude, gemini, qwen -- `gemini37_partial` is
excluded, it only has 6 graded cases).

    uv run --no-sync python scripts/k2_confusion_matrix.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k2_blind_regrade import GRADES, JUDGES, RUNS, TABLES, cc_grade  # noqa: E402

FULL_JUDGES = ("claude", "gemini", "qwen")


def _acc(v: str) -> str:
    return "ok" if v in ("better", "equivalent") else "not"


def _labels() -> dict[str, dict[str, str]]:
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
    return labels


def main() -> None:
    labels = _labels()
    base_rate_rows = []
    for j in FULL_JUDGES:
        vals = [_acc(v) for v in labels[j].values()]
        base_rate_rows.append(
            {
                "judge": j,
                "n": len(vals),
                "n_not_acceptable": sum(v == "not" for v in vals),
                "base_rate_not_acceptable": sum(v == "not" for v in vals) / len(vals),
            }
        )
    base_rates = pl.DataFrame(base_rate_rows)
    base_rates.write_csv(f"{TABLES}/v3_k2_base_rates.csv")
    print(base_rates)

    conf_rows = []
    for a, b in (("claude", "gemini"), ("claude", "qwen"), ("gemini", "qwen")):
        common = sorted(set(labels[a]) & set(labels[b]))
        la = [_acc(labels[a][k]) for k in common]
        lb = [_acc(labels[b][k]) for k in common]
        for a_val in ("ok", "not"):
            for b_val in ("ok", "not"):
                conf_rows.append(
                    {
                        "pair": f"{a}-{b}",
                        "n": len(common),
                        f"{a}": a_val,
                        f"{b}": b_val,
                        "count": sum(x == a_val and y == b_val for x, y in zip(la, lb, strict=True)),
                    }
                )
    confusion = pl.DataFrame(conf_rows)
    confusion.write_csv(f"{TABLES}/v3_k2_confusion_matrices.csv")
    with pl.Config(tbl_rows=-1):
        print(confusion)


if __name__ == "__main__":
    main()
