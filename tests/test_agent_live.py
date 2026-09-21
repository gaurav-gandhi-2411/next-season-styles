"""The 4a transcript lint: a step must carry one of the allowed labels, and REPLAY is not one."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from agent_live import LABELS, Transcript  # noqa: E402 -- see sys.path.insert above


def test_replay_is_not_an_allowed_label() -> None:
    assert "REPLAY" not in LABELS
    with pytest.raises(ValueError, match="unlabelled or unknown"):
        Transcript().step("REPLAY", "read from a table")


def test_counts_by_label_are_rendered() -> None:
    t = Transcript()
    t.step("LIVE", "a call", 1.5)
    t.step("LIVE", "another")
    t.step("NOT PERFORMED", "human check")
    out = t.render()
    assert "| LIVE | 2 |" in out and "| NOT PERFORMED | 1 |" in out and "| INPUT | 0 |" in out
