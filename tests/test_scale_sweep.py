from __future__ import annotations

import pytest

from nss.generate.scale_sweep import (
    STYLE_KEY,
    build_prompt,
    is_monotonic_increasing,
    style_description,
)


def test_style_description_splits_final_rank_1_style_key() -> None:
    """The chosen B4 style_key parses into its 5 attribute parts in the expected order."""
    assert style_description(STYLE_KEY) == "black solid jersey basic t-shirt"


def test_style_description_rejects_malformed_style_key() -> None:
    """A style_key without exactly 5 ' || '-separated parts fails loudly, not silently."""
    with pytest.raises(ValueError, match="Expected 5"):
        style_description("Ladieswear || T-shirt")


def test_build_prompt_includes_department_and_description() -> None:
    """The generation prompt is deterministically derived from the style_key's attributes."""
    prompt = build_prompt(STYLE_KEY)
    assert prompt == (
        "a new black solid jersey basic t-shirt fashion concept for ladieswear, "
        "product photography, plain background"
    )


def test_is_monotonic_increasing_true_for_nondecreasing_sequence() -> None:
    """A strictly increasing sequence counts as (non-strictly) monotonic."""
    assert is_monotonic_increasing([0.1, 0.2, 0.2, 0.5])


def test_is_monotonic_increasing_false_when_a_later_value_drops() -> None:
    """A single decrease anywhere in the sequence breaks monotonicity."""
    assert not is_monotonic_increasing([0.1, 0.5, 0.3, 0.6])


def test_is_monotonic_increasing_true_for_single_value() -> None:
    """A single-element (or empty) sequence is trivially monotonic."""
    assert is_monotonic_increasing([0.42])
    assert is_monotonic_increasing([])
