"""H3: the critic's decision rule, the orchestrator's routing, and the gate ablation, as code.

The agent layer (`agents/*.md` + `scripts/agent_demo.py`) is role definitions and a scripted
driver: no LLM makes a decision. This module is the executable form of the two rules that matter
for evaluation, written from `agents/critic.md` and `agents/orchestrator.md` and checked against
the expectation table pre-registered in `reports/v3/PREREGISTRATION.md` (H3, commit 14a0ae7):

- `decide` (from `critic_rule`, the single rule shared with `qc_gates` and the drivers): REJECT if
  any gating gate definitely failed, else INCONCLUSIVE if a gate is unmeasured, else
  PASS_PENDING_HUMAN.
- `route`: the next hop for a verdict at a given attempt (1 original + `RETRY_CAP` retries).
- `ablate`: recompute decisions with each gate removed in turn, and count catches per gate.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from nss.generate.critic_rule import GATES, INCONCLUSIVE, PASS, REJECT, Passes, decide
from nss.generate.critic_rule import failing as _failing

RETRY_CAP = 2  # `critic.md`: at most 2 retries, i.e. 3 attempts in total
MIN_SCALE, SCALE_STEP = 0.15, 0.10  # the pre-registered retry rule for ip_adapter_scale
_ROW_COLUMNS = {
    "gate1": "gate1_pass",
    "gate1b": "gate1b_pass",
    "integrity": "integrity_floor_pass",
    "gate2": "gate2_pass",
    "gate3": "gate3_pass",
}


def passes_from_score(score: dict[str, Any]) -> tuple[Passes, bool | None]:
    """Gate pass flags and the clone-control flag from one `score_concept` response.

    A gate that is absent from the response is recorded as `None` (unmeasured), never as a pass.
    """
    passes = {g: (score.get(g) or {}).get("pass") for g in GATES}
    clone_ok = (score.get("gate1b") or {}).get("clone_control_failed_as_required")
    return passes, clone_ok


def passes_from_row(row: dict[str, Any]) -> tuple[Passes, bool | None]:
    """The same from a stored `*_scored.csv` row (`clone_fails_gate1b` = clone-control flag)."""
    return {g: row.get(col) for g, col in _ROW_COLUMNS.items()}, row.get("clone_fails_gate1b")


def failing(
    passes: Passes, mask: frozenset[str] = frozenset(), clone_ok: bool | None = True
) -> list[str]:
    """Gates (in gate order, minus `mask`) that definitely failed (see `critic_rule`)."""
    return _failing(passes, clone_ok, mask)


@dataclass(frozen=True)
class Route:
    """Where a verdict sends the run next, per `orchestrator.md`.

    Attributes:
        next_agent: The sub-agent called next, or `None` when the run stops or escalates.
        tool: The MCP tool that hop needs (`None` when there is no hop).
        adjust: The one parameter the critic changes for a retry (`seed` or `ip_adapter_scale`).
        outcome: `RETRY`, `FORWARD`, `FAILED` or `ESCALATE_INCONCLUSIVE`.
    """

    next_agent: str | None
    tool: str | None
    adjust: str | None
    outcome: str


def route(verdict: str, attempt: int, failed: list[str] | None = None) -> Route:
    """The next hop for `verdict` on the given 1-based `attempt`."""
    if verdict == PASS:
        return Route("forecaster", "forecast_concept", None, "FORWARD")
    if verdict == INCONCLUSIVE:
        return Route(None, None, None, "ESCALATE_INCONCLUSIVE")
    if verdict != REJECT:
        raise ValueError(f"unknown verdict {verdict!r}")
    if attempt > RETRY_CAP:
        return Route(None, None, None, "FAILED")
    adjust = "seed" if failed == ["integrity"] else "ip_adapter_scale"
    return Route("concept-designer", "generate_concept", adjust, "RETRY")


def next_attempt(failed: list[str], scale: float, seed: int) -> tuple[float, int]:
    """The pre-registered retry rule: change only the seed (integrity-only miss) or the scale."""
    if failed == ["integrity"]:
        return scale, seed + 1
    return round(max(MIN_SCALE, scale - SCALE_STEP), 2), seed


def score_with_retry(
    score_fn: Callable[[], dict[str, Any]],
) -> tuple[dict[str, Any] | None, int]:
    """Call the scoring tool; one retry of the SCORING call on an exception (`critic.md`).

    Returns `(score, n_calls)`; `score` is `None` when both calls raised (=> INCONCLUSIVE).
    """
    for n_calls in (1, 2):
        try:
            return score_fn(), n_calls
        except Exception:  # noqa: BLE001  the critic treats any tool error as inconclusive
            continue
    return None, 2


def critic_verdict(score_fn: Callable[[], dict[str, Any]]) -> tuple[str, list[str], int]:
    """`(verdict, failing gates, scoring calls made)` for one attempt, including tool errors."""
    score, n_calls = score_with_retry(score_fn)
    if score is None:
        return INCONCLUSIVE, [], n_calls
    passes, clone_ok = passes_from_score(score)
    return decide(passes, clone_ok), failing(passes, clone_ok=clone_ok), n_calls


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a proportion (`nan, nan` when `n == 0`)."""
    if n == 0:
        return math.nan, math.nan
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True)
class Case:
    """One evaluation case: its stratum, truth (`True` = should be accepted) and gate results."""

    name: str
    stratum: str
    truth: bool
    passes: Passes
    clone_ok: bool | None = True


def confusion(cases: Iterable[Case], mask: frozenset[str] = frozenset()) -> Counter[str]:
    """Counts of `tp`, `fp`, `fn`, `tn` and `inconclusive_*` (INCONCLUSIVE counts as not-accept)."""
    c: Counter[str] = Counter()
    for case in cases:
        v = decide(case.passes, case.clone_ok, mask)
        accepted = v == PASS
        c[("tp" if accepted else "fn") if case.truth else ("fp" if accepted else "tn")] += 1
        if v == INCONCLUSIVE:
            c["inconclusive_pos" if case.truth else "inconclusive_neg"] += 1
    return c


def precision_recall(c: Counter[str]) -> dict[str, Any]:
    """Precision and recall with Wilson intervals from a `confusion` count."""
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": c["tn"],
        "precision": tp / (tp + fp) if tp + fp else math.nan,
        "precision_ci": wilson(tp, tp + fp),
        "recall": tp / (tp + fn) if tp + fn else math.nan,
        "recall_ci": wilson(tp, tp + fn),
    }


def ablate(cases: list[Case]) -> list[dict[str, Any]]:
    """Per gate and stratum: catches, unique catches and unique false rejects.

    A case is *caught* by a gate if the gate is among its failing gates. A catch is *unique* when
    removing the gate would newly let the case reach PASS_PENDING_HUMAN (the pre-registered
    meaning): the gate is the only failing gate AND nothing else blocks a pass (no other null
    gate, clone control validated). `sole_failing` counts only the first condition, so the two
    differ exactly where an unmeasured gate (e.g. Gate 3 with no brief) would still stop the case.
    """
    rows = []
    for gate in GATES:
        for stratum in sorted({c.stratum for c in cases}):
            sub = [c for c in cases if c.stratum == stratum]
            caught = unique = sole = 0
            for c in sub:
                f = failing(c.passes, clone_ok=c.clone_ok)
                caught += gate in f
                sole += f == [gate]
                unique += (
                    decide(c.passes, c.clone_ok) != PASS
                    and decide(c.passes, c.clone_ok, frozenset({gate})) == PASS
                )
            rows.append(
                {
                    "gate": gate,
                    "stratum": stratum,
                    "truth_accept": sub[0].truth,
                    "n": len(sub),
                    "catches": caught,
                    "unique": unique,
                    "sole_failing": sole,
                }
            )
    return rows
