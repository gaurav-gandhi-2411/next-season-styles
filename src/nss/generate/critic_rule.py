"""The critic's accept/reject rule: one definition for the spec, the tool and the drivers.

Before J3 three places answered "a gate failed and another gate was not run" three ways:
`agents/critic.md` (INCONCLUSIVE, escalate), `scripts/agent_demo.critic_replay` (REJECT, retry) and
`qc_gates.verdict_from_gates` (REJECT). They now all call this module.

The rule, in order:

1. A gate is *unmeasured* if its `pass` is `None` / missing, or it is Gate 1b and its exact-clone
   control did not fail as required (the gate is UNVALIDATED and "cannot pass anything", so its own
   `False` is not evidence about the concept either).
2. Any gate that is definitely `False` -> **REJECT**, whatever else is unmeasured. A concept that
   has already failed a gate cannot pass, and a retry can fix the failure.
3. Else any unmeasured gate -> **INCONCLUSIVE** (escalate, never retry the generation: nothing
   failed, so a new image would face the same unmeasured gate; the cause is a missing brief, a tool
   error or a broken control, and a person has to fix it).
4. Else **PASS_PENDING_HUMAN**.
"""

from __future__ import annotations

from typing import Any

GATES = ("gate1", "gate1b", "integrity", "gate2", "gate3")
REJECT, PASS, INCONCLUSIVE = "REJECT", "PASS_PENDING_HUMAN", "INCONCLUSIVE"

Passes = dict[str, bool | None]


def measured(
    passes: Passes, clone_ok: bool | None = True, mask: frozenset[str] = frozenset()
) -> dict[str, bool | None]:
    """Each gate's result with unmeasured gates as `None` (`mask`ed gates are dropped)."""
    out: dict[str, bool | None] = {}
    for gate in GATES:
        if gate in mask:
            continue
        value = passes.get(gate)
        if gate == "gate1b" and clone_ok is not True:
            value = None
        out[gate] = value
    return out


def failing(
    passes: Passes, clone_ok: bool | None = True, mask: frozenset[str] = frozenset()
) -> list[str]:
    """Gates (in gate order) that definitely failed."""
    return [g for g, v in measured(passes, clone_ok, mask).items() if v is False]


def unmeasured(
    passes: Passes, clone_ok: bool | None = True, mask: frozenset[str] = frozenset()
) -> list[str]:
    """Gates (in gate order) that have no usable result."""
    return [g for g, v in measured(passes, clone_ok, mask).items() if v is None]


def decide(passes: Passes, clone_ok: bool | None = True, mask: frozenset[str] = frozenset()) -> str:
    """REJECT, INCONCLUSIVE or PASS_PENDING_HUMAN under the rule above."""
    if failing(passes, clone_ok, mask):
        return REJECT
    if unmeasured(passes, clone_ok, mask):
        return INCONCLUSIVE
    return PASS


def clone_ok_from_gates(gates: dict[str, dict[str, Any]]) -> bool | None:
    """The clone-control flag of a `score_gates` gate dict (`True` when it is not reported)."""
    return gates.get("gate1b", {}).get("clone_control_failed_as_required", True)
