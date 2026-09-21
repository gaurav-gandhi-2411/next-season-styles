from __future__ import annotations

import math
from typing import Any

from nss.generate import agent_eval as ae

ALL_PASS: ae.Passes = dict.fromkeys(ae.GATES, True)


def _score(**overrides: Any) -> dict[str, Any]:
    out: dict[str, Any] = {g: {"pass": True} for g in ae.GATES}
    out["gate1b"]["clone_control_failed_as_required"] = True
    for gate, val in overrides.items():
        out[gate] = val
    return out


def test_decide_pass_reject_and_null() -> None:
    assert ae.decide(ALL_PASS) == ae.PASS
    assert ae.decide({**ALL_PASS, "gate3": False}) == ae.REJECT
    assert ae.decide({**ALL_PASS, "gate2": None}) == ae.INCONCLUSIVE  # never a pass, never a reject


def test_a_definite_failure_rejects_even_if_another_gate_was_not_run() -> None:
    """J3: failed + unmeasured -> REJECT (a retry can fix it); unmeasured alone -> escalate."""
    p = {**ALL_PASS, "gate3": None, "integrity": False}
    assert ae.decide(p) == ae.REJECT
    assert ae.decide({**ALL_PASS, "gate3": None}) == ae.INCONCLUSIVE  # nothing failed: escalate


def test_unvalidated_clone_control_is_inconclusive_not_pass() -> None:
    assert ae.decide(ALL_PASS, clone_ok=False) == ae.INCONCLUSIVE
    assert ae.decide(ALL_PASS, clone_ok=None) == ae.INCONCLUSIVE
    # an unvalidated gate1b is unmeasured, so its own False is not a definite failure either ...
    assert ae.decide({**ALL_PASS, "gate1b": False}, clone_ok=False) == ae.INCONCLUSIVE
    # ... but another definite failure still rejects
    assert ae.decide({**ALL_PASS, "gate1b": False, "gate2": False}, clone_ok=False) == ae.REJECT
    # once gate1b is masked out of the decision, its clone control no longer matters
    assert ae.decide(ALL_PASS, clone_ok=False, mask=frozenset({"gate1b"})) == ae.PASS


def test_mask_removes_a_gate_from_the_decision() -> None:
    p = {**ALL_PASS, "integrity": False}
    assert ae.decide(p) == ae.REJECT
    assert ae.decide(p, mask=frozenset({"integrity"})) == ae.PASS


def test_route_table() -> None:
    assert ae.route(ae.REJECT, 1, ["gate2"]) == ae.Route(
        "concept-designer", "generate_concept", "ip_adapter_scale", "RETRY"
    )
    assert ae.route(ae.REJECT, 2, ["integrity"]).adjust == "seed"
    assert ae.route(ae.REJECT, 3, ["gate2"]).outcome == "FAILED"  # cap: 1 + 2 retries
    assert ae.route(ae.PASS, 1).next_agent == "forecaster"
    assert ae.route(ae.INCONCLUSIVE, 1).next_agent is None


def test_next_attempt_changes_only_one_parameter() -> None:
    assert ae.next_attempt(["integrity"], 0.35, 42) == (0.35, 43)
    assert ae.next_attempt(["gate1", "gate3"], 0.35, 42) == (0.25, 42)
    assert ae.next_attempt(["gate2"], 0.15, 42) == (0.15, 42)  # floor


def test_tool_error_is_retried_once_then_inconclusive() -> None:
    calls = []

    def always_raises() -> dict[str, Any]:
        calls.append(1)
        raise RuntimeError("boom")

    assert ae.critic_verdict(always_raises) == (ae.INCONCLUSIVE, [], 2)
    assert len(calls) == 2


def test_tool_error_that_recovers_gives_the_normal_verdict() -> None:
    state = {"n": 0}

    def flaky() -> dict[str, Any]:
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("transient")
        return _score(gate3={"pass": False})

    assert ae.critic_verdict(flaky) == (ae.REJECT, ["gate3"], 2)


def test_malformed_scores_are_inconclusive() -> None:
    v, _f, _n = ae.critic_verdict(lambda: _score(gate2={"status": "not_run", "pass": None}))
    assert v == ae.INCONCLUSIVE
    v, _f, _n = ae.critic_verdict(lambda: _score(gate3={"pass": None}))
    assert v == ae.INCONCLUSIVE
    bad = _score(gate1b={"pass": False, "clone_control_failed_as_required": False})
    assert ae.critic_verdict(lambda: bad)[0] == ae.INCONCLUSIVE


def test_wilson_bounds() -> None:
    lo, hi = ae.wilson(4, 4)
    assert abs(lo - 0.510) < 0.01 and hi > 0.999999
    assert all(math.isnan(x) for x in ae.wilson(0, 0))


def test_ablation_counts_unique_catches() -> None:
    def case(name: str, stratum: str, truth: bool, **fail: bool) -> ae.Case:
        return ae.Case(name, stratum, truth, {**ALL_PASS, **{g: False for g in fail}})

    cases = [
        case("a", "bad", False, gate2=True, gate3=True),  # two failing gates: no unique catch
        case("b", "bad", False, gate1b=True),  # unique to gate1b
        case("c", "good", True, integrity=True),  # a unique false reject by integrity
    ]
    table = {(r["gate"], r["stratum"]): r for r in ae.ablate(cases)}
    assert table["gate2", "bad"]["catches"] == 1 and table["gate2", "bad"]["unique"] == 0
    assert table["gate1b", "bad"]["unique"] == 1
    assert table["integrity", "good"]["unique"] == 1
    assert table["gate1", "bad"]["catches"] == 0


def test_a_null_gate_means_the_sole_failing_gate_is_not_a_unique_catch() -> None:
    # integrity is the only failing gate, but gate3 was never run, so removing integrity would
    # still not let the case pass: not a unique catch under the pre-registered meaning
    c = ae.Case("x", "bad", False, {**ALL_PASS, "integrity": False, "gate3": None})
    row = next(r for r in ae.ablate([c]) if r["gate"] == "integrity")
    assert row["catches"] == 1 and row["sole_failing"] == 1 and row["unique"] == 0


def test_confusion_and_precision_recall() -> None:
    cases = [
        ae.Case("tp", "p", True, ALL_PASS),
        ae.Case("fn", "p", True, {**ALL_PASS, "gate2": False}),
        ae.Case("fp", "n", False, ALL_PASS),
        ae.Case("tn", "n", False, {**ALL_PASS, "gate1": False}),
    ]
    pr = ae.precision_recall(ae.confusion(cases))
    assert (pr["tp"], pr["fp"], pr["fn"], pr["tn"]) == (1, 1, 1, 1)
    assert pr["precision"] == 0.5 and pr["recall"] == 0.5
