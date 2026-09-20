from __future__ import annotations

import json
from pathlib import Path

import pytest

from nss.generate import gate3


def test_majority_present_requires_strict_majority() -> None:
    assert gate3.majority_present([True, True])
    assert gate3.majority_present([True, True, False])
    assert not gate3.majority_present([True, False])  # 1 of 2 is not a majority
    assert not gate3.majority_present([False, False, True])


def test_majority_present_empty_is_a_fail_not_a_pass() -> None:
    assert not gate3.majority_present([])


def test_change_questions_one_binary_question_per_change() -> None:
    qs = gate3.change_questions(["a high funnel neck collar", "dark brown rib cuffs"])
    assert list(qs) == ["c0", "c1"]
    assert qs["c0"] == "Does this garment have a high funnel neck collar?"


def test_parse_yes_no_is_strict_and_fails_closed() -> None:
    parsed = gate3._parse_yes_no(
        json.dumps({"c0": "Yes", "c1": "no", "c2": "maybe"}), ["c0", "c1", "c2", "c3"]
    )
    assert parsed == {"c0": True, "c1": False, "c2": False, "c3": False}  # missing key -> no


def test_parse_yes_no_empty_response_raises() -> None:
    with pytest.raises(RuntimeError):
        gate3._parse_yes_no("", ["c0"])


def test_gate3_local_uses_majority(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = iter([(True, "Yes"), (False, "No")])
    monkeypatch.setattr(gate3.local_vlm, "ask_yes_no", lambda _img, _q: next(answers))
    out = gate3.gate3_local(Path("x.png"), ["one", "two"])
    assert out["n_present"] == 1
    assert out["pass"] is False  # 1 of 2 briefed changes is not a majority


def test_gate3_api_batches_all_questions_in_one_call(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, str]] = []

    def fake(_img: Path, questions: dict[str, str]) -> dict[str, bool]:
        seen.append(questions)
        return {k: k != "c1" for k in questions}

    monkeypatch.setitem(gate3.API_BATCH_CALLERS, "groq", fake)
    out = gate3.gate3_api("groq", Path("x.png"), ["one", "two", "three"])
    assert len(seen) == 1 and set(seen[0]) == {"c0", "c1", "c2", "integrity"}
    assert out["answers"] == [True, False, True]
    assert out["pass"] is True and out["coherent"] is True
