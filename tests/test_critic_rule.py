"""Unit tests for `nss.generate.critic_rule.decide` (A3: a prior human REJECT is terminal)."""

from __future__ import annotations

from nss.generate import critic_rule as cr


def _all_pass() -> dict[str, bool | None]:
    return dict.fromkeys(cr.GATING, True)


def test_prior_human_reject_is_terminal_even_when_every_gate_passes() -> None:
    """S10's shape: every gate passes, but a person already rejected an earlier attempt."""
    assert cr.decide(_all_pass(), prior_human_reject=True) == cr.REJECT


def test_prior_human_reject_beats_an_unmeasured_gate_too() -> None:
    passes = {**_all_pass(), "gate3": None}
    assert cr.decide(passes, prior_human_reject=True) == cr.REJECT


def test_default_is_unchanged_when_there_is_no_prior_human_reject() -> None:
    """The new parameter defaults to False, so every existing call site is unaffected."""
    assert cr.decide(_all_pass()) == cr.PASS
    assert cr.decide(_all_pass(), prior_human_reject=False) == cr.PASS


def test_a_failed_gate_still_rejects_without_any_prior_human_reject() -> None:
    passes = {**_all_pass(), "gate2": False}
    assert cr.decide(passes) == cr.REJECT
