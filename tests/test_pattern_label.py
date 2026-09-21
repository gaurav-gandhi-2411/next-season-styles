from __future__ import annotations

import pytest

from nss.generate.pattern_label import (
    apply_mapping,
    mapped_graphical_score,
    names_visible_pattern,
)

LABEL = "All over pattern"


@pytest.mark.parametrize(
    "answer",
    ["Polka dot.", "Floral.", "Checkered.", "Dots.", "Melange.", "geometric pattern", "Striped"],
)
def test_named_patterns_satisfy_the_label(answer: str) -> None:
    assert names_visible_pattern(answer)
    assert mapped_graphical_score(answer, LABEL) == 1.0


@pytest.mark.parametrize(
    "answer",
    ["Solid.", "Plain.", "solid color", "None.", "ORIGINAL.", "", "Bikini.", "solid, striped"],
)
def test_plain_nonsense_empty_and_ambiguous_answers_fail(answer: str) -> None:
    assert not names_visible_pattern(answer)
    assert mapped_graphical_score(answer, LABEL) == 0.0


def test_a_long_caption_is_left_to_the_legacy_scorer() -> None:
    caption = "The image is of a bikini top with a plain grey background and an orange pattern."
    assert mapped_graphical_score(caption, LABEL) is None


def test_the_rule_is_inactive_for_every_other_label() -> None:
    for truth in ("Solid", "Melange", "Stripe", "Jersey Basic", "Other structure", ""):
        assert mapped_graphical_score("Polka dot.", truth) is None
        assert mapped_graphical_score("Solid.", truth) is None


def test_apply_mapping_changes_only_the_graphical_dimension_and_only_for_the_label() -> None:
    scores = {"product_type": 0.0, "colour_family": 0.85, "graphical_treatment": 0.0}
    extraction = {
        "product_type": "Bikini.",
        "colour_family": "Orange.",
        "graphical_treatment": "Floral.",
    }
    out = apply_mapping(scores, extraction, {"graphical_treatment": LABEL})
    assert out == {"product_type": 0.0, "colour_family": 0.85, "graphical_treatment": 1.0}
    assert apply_mapping(scores, extraction, {"graphical_treatment": "Solid"}) == scores
    assert scores["graphical_treatment"] == 0.0  # the input is not mutated
