"""The closed-loop top-5 view must reproduce the recorded retrieval table, not re-derive it."""

from __future__ import annotations

import pytest

from nss.generate import final_registry
from nss.generate.final_selection_figures import CLOSED_LOOP_LABEL, SWEATER, TOP, closed_loop_view

DRESS = final_registry.STYLE_ORDER[1]


def test_sweater_intended_style_is_third_and_0_006_behind() -> None:
    v = closed_loop_view(SWEATER)
    assert v is not None
    assert v["intended_pos"] == 3
    assert v["gap"] == pytest.approx(0.006)
    assert v["summary"] == "Intended style 3rd, 0.006 behind top-1"
    assert [t["pos"] for t in v["rows"]] == [1, 2, 3, 4, 5]
    assert v["rows"][0]["rank"] == 461  # the recorded top-1 rank the old figure showed alone


def test_dress_intended_style_is_first() -> None:
    v = closed_loop_view(DRESS)
    assert v is not None
    assert v["intended_pos"] == 1
    assert v["gap"] is None
    assert v["summary"].startswith("Intended style is 1st")


def test_white_top_intended_style_outside_top5_is_stated_not_hidden() -> None:
    v = closed_loop_view(TOP)
    assert v is not None
    assert v["intended_pos"] is None
    assert v["summary"] == "Intended style is outside the top 5"
    assert not any(t["intended"] for t in v["rows"])


def test_unknown_style_has_no_view() -> None:
    assert closed_loop_view("No || Such || Style || Here || Ever") is None


def test_label_states_the_recorded_numbers() -> None:
    for needle in ("27.5%", "12.5%", "n=40", "p=0.21", "1,980", "Near-ties dominate"):
        assert needle in CLOSED_LOOP_LABEL
