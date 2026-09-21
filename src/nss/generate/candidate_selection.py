"""The concept selection rule, the pass definition and pass-rate statistics (fixed in advance).

These definitions are pre-registered in `reports/v3/PREREGISTRATION.md` (Track 1) and committed
BEFORE the rule was applied to any pool. They restate the rule the repo already used
(`final_selection_figures`: "passes every automatic gate -> most briefed changes visible -> highest
fidelity, then confirmed by a human looking at the image"); they do not change it.

PASS. A candidate passes iff EVERY automatic gate passes: Gate 1, Gate 1b, the global integrity
floor (`integrity_floor_pass` in `candidates_scored.csv`, already the global floor), Gate 2 (the
gating panel, SmolVLM) and Gate 3 (every briefed change judged visible by a strict majority; the
panel rule is in `concept_scoring.apply_panel_rule`). Florence-2 is advisory and never gates;
the per-style integrity floor is advisory and never gates. Exactly `final_selection_figures.
selection_rows`' `automatic_gates == "PASS"`.

RANK among passing candidates, in this order:
 1. the number of briefed changes Gate 3 judged visible (count of "Y" in the Gate 3 answers);
 2. the highest SmolVLM attribute fidelity (the gating judge);
 3. the highest Florence-2 fidelity (advisory judge; tie-break only);
 4. the lowest seed (determinism).
The human visual check is a VETO applied afterwards by a person looking at the image. It is never a
ranking input and it is never inferred from a score.

If no candidate passes, `select_candidate` returns None: it never silently promotes a failure.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import polars as pl

GATE_COLUMNS = (
    "gate1_pass",
    "gate1b_pass",
    "integrity_floor_pass",
    "gate2_pass",
    "gate3_pass",
)
WILSON_Z = 1.959964  # two-sided 95%


def passes_all_gates(row: Mapping[str, Any]) -> bool:
    """True iff every automatic gate passed (a missing or null gate is a fail, never a pass)."""
    return all(row.get(col) is True for col in GATE_COLUMNS)


def n_changes_visible(answers: str | None) -> int:
    """How many briefed changes Gate 3 judged visible, from its answer string ("YN" -> 1)."""
    return 0 if not answers else answers.count("Y")


def rank_key(row: Mapping[str, Any]) -> tuple[int, float, float, int]:
    """Sort key (larger is better) for a PASSING candidate; see module docstring."""
    return (
        n_changes_visible(row.get("smolvlm_gate3_answers")),
        float(row.get("smolvlm_fidelity") or 0.0),
        float(row.get("florence2_fidelity") or 0.0),
        -int(row["seed"]),
    )


def select_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The rule's pick among candidates of ONE style, or None when nothing passes."""
    passing = [r for r in rows if passes_all_gates(r)]
    return max(passing, key=rank_key) if passing else None


def wilson_interval(successes: int, n: int) -> tuple[float, float]:
    """Wilson score 95% interval for a proportion (well behaved at 0/n and n/n)."""
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    z2 = WILSON_Z**2
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = WILSON_Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def intervals_disjoint(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """True iff two intervals do not overlap (the pre-registered 'yield changed' criterion)."""
    return a[1] < b[0] or b[1] < a[0]


def pass_rate_table(scored: pl.DataFrame, seed_groups: Mapping[str, range]) -> pl.DataFrame:
    """Per style and seed group: passes, n, rate and Wilson interval.

    Args:
        scored: A `candidates_scored.csv`-shaped frame, already restricted to one scale.
        seed_groups: Name -> seeds, e.g. {"prior_8": range(42, 50), "new_16": range(50, 66)}.
    """
    rows: list[dict[str, Any]] = []
    for style_id in scored["style_id"].unique(maintain_order=True).to_list():
        sub = scored.filter(pl.col("style_id") == style_id).to_dicts()
        for name, seeds in seed_groups.items():
            group = [r for r in sub if int(r["seed"]) in set(seeds)]
            k = sum(passes_all_gates(r) for r in group)
            lo, hi = wilson_interval(k, len(group))
            rows.append(
                {
                    "style_id": style_id,
                    "seed_group": name,
                    "n": len(group),
                    "n_pass": k,
                    "pass_rate": k / len(group) if group else math.nan,
                    "wilson_lo": lo,
                    "wilson_hi": hi,
                }
            )
    return pl.DataFrame(rows)
