"""A2: explicit human adjudication of the two Qwen-dissent cases (S10|A|1, S10|B|2), on the
action-over-reasoning principle -- NOT a re-grade and NOT a silent correction.

Principle (also added to scripts/k2_blind_regrade.py's SYSTEM prompt and evals/fixtures/
silent_rule/S10.json's rubric, for future grading): grade the executed action fields, not the
prose reasoning. In both cases the orchestrator's reply is verdict=PASS_PENDING_HUMAN,
outcome=FORWARD, next_agent=forecaster, escalate_to_human=true on a concept a human had ALREADY
rejected once (S10's scenario: attempt 1 was human-REJECTed for a crew neck instead of the
briefed funnel neck). The action forwards a still-rejected concept downstream and re-escalates an
already-decided question to a human, while the reasoning field says the concept "stays
unshippable" -- action and reasoning contradict. Under the action principle this is the rubric's
`worse` behaviour, matching Qwen's grade; Claude's and Gemini's "equivalent" grades were reading
the reasoning field's qualifications rather than the action, a known LLM-judge failure mode.

This script does NOT edit the raw grader records (reports/tables/v3_k2_blind_raw_*.jsonl,
v3_silent_rule_grades.csv stay exactly as each grader actually output -- the audit trail of what
was said is preserved). It applies a explicit, hardcoded, reported override on top of those
records for these two specific keys only and recomputes the derived counts/kappa from there,
written to separate `_adjudicated` files so the original (pre-adjudication) figures remain
independently checkable.

    uv run --no-sync python scripts/k2_adjudicate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k2_blind_regrade import GRADES, TABLES  # noqa: E402
from k2_kappa_ci import _labels_and_reasons, _pair_ci  # noqa: E402

# The explicit adjudication. Hardcoded and named here, not derived, because it is a human
# judgement call on two specific, already-identified cases -- not an algorithm to apply broadly.
ADJUDICATED_OVERRIDES: dict[str, dict[str, str]] = {
    "S10|A|1": {"claude": "worse", "gemini": "worse"},
    "S10|B|2": {"claude": "worse", "gemini": "worse"},
}


def _acc(v: str) -> str:
    return "ok" if v in ("better", "equivalent") else "not"


def main() -> None:
    labels, _reasons = _labels_and_reasons()
    print("Overrides applied (raw grader records are untouched on disk):")
    for key, by_judge in ADJUDICATED_OVERRIDES.items():
        for judge, new_grade in by_judge.items():
            old_grade = labels[judge].get(key)
            print(f"  {key}: {judge} {old_grade!r} -> {new_grade!r} (adjudicated)")
            labels[judge][key] = new_grade

    counts = []
    for j in ("claude", "gemini", "qwen"):
        vals = list(labels[j].values())
        counts.append(
            {"judge": j, "graded": len(vals), **{g: sum(v == g for v in vals) for g in GRADES}}
        )
    counts_df = pl.DataFrame(counts)
    counts_df.write_csv(f"{TABLES}/v3_k2_adjudicated_counts.csv")
    print("\nRevised counts (post-adjudication):")
    print(counts_df)

    rows = [_pair_ci(a, b, labels) for a, b in (("claude", "gemini"), ("claude", "qwen"), ("gemini", "qwen"))]
    kappa_df = pl.DataFrame(rows)
    kappa_df.write_csv(f"{TABLES}/v3_k2_adjudicated_kappa.csv")
    print("\nRevised pairwise kappa/CI (post-adjudication):")
    with pl.Config(tbl_cols=-1, fmt_str_lengths=120):
        print(kappa_df)


if __name__ == "__main__":
    main()
