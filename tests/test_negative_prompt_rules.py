"""Tests for `nss.generate.negative_prompt_rules` (task F4).

Synthetic style-attribute combos, per the task's own verification requirement -- these predicates
and the rule table must generalize to ANY future style with a matching attribute profile, not just
this project's 3 currently-selected styles.
"""

from __future__ import annotations

from nss.generate.negative_prompt_rules import (
    NEGATIVE_PROMPT_RULES,
    apply_negative_prompt_rules,
    is_knitwear_or_sweater,
    is_solid,
    is_underwear_or_intimate,
)

_BASE_NEGATIVE = "blurry, distorted proportions, low-resolution, watermark"


def _attrs(**overrides: str) -> dict[str, str]:
    base = {
        "garment_category": "Trouser",
        "construction_group": "Woven",
        "colour_name": "Navy",
        "pattern_or_finish": "Stripe",
    }
    base.update(overrides)
    return base


def test_is_solid_matches_case_insensitively() -> None:
    assert is_solid(_attrs(pattern_or_finish="solid")) is True
    assert is_solid(_attrs(pattern_or_finish="Solid")) is True
    assert is_solid(_attrs(pattern_or_finish="Melange")) is False


def test_is_underwear_or_intimate_matches_hm_taxonomy_construction_group() -> None:
    """H&M's real construction-group value for underwear is `Under-, Nightwear` -- must match."""
    assert is_underwear_or_intimate(_attrs(construction_group="Under-, Nightwear")) is True
    assert is_underwear_or_intimate(_attrs(garment_category="Underwear bottom")) is True
    assert is_underwear_or_intimate(_attrs(garment_category="Lingerie set")) is True
    assert is_underwear_or_intimate(_attrs()) is False


def test_is_knitwear_or_sweater_matches_either_attribute() -> None:
    assert is_knitwear_or_sweater(_attrs(construction_group="Knitwear")) is True
    assert is_knitwear_or_sweater(_attrs(garment_category="Sweater")) is True
    assert is_knitwear_or_sweater(_attrs()) is False


def test_apply_negative_prompt_rules_solid_style_excludes_pattern_terms() -> None:
    """Task F4's real finding: a Solid style drifted into a floral-lace pattern because nothing
    excluded pattern/print language -- the rule must add exactly that."""
    negative = apply_negative_prompt_rules(_BASE_NEGATIVE, _attrs(pattern_or_finish="Solid"))
    for term in ("floral", "lace", "pattern", "print", "embroidery"):
        assert term in negative


def test_apply_negative_prompt_rules_underwear_style_excludes_human_terms() -> None:
    negative = apply_negative_prompt_rules(
        _BASE_NEGATIVE, _attrs(construction_group="Under-, Nightwear")
    )
    for term in ("model", "person", "body", "human"):
        assert term in negative


def test_apply_negative_prompt_rules_knitwear_style_excludes_close_up_terms() -> None:
    """Task F4's real finding: a Sweater style's candidates were degenerate fabric-texture
    close-ups; E5 hand-fixed this for ONE style only -- the rule must generalize to any Knitwear
    style, not just that specific style_key."""
    negative = apply_negative_prompt_rules(_BASE_NEGATIVE, _attrs(construction_group="Knitwear"))
    for term in ("close-up", "macro", "fabric swatch", "texture detail", "cropped", "zoomed"):
        assert term in negative


def test_apply_negative_prompt_rules_unrelated_style_adds_nothing() -> None:
    """A style matching none of the 3 rules (e.g. a striped woven trouser) is left unchanged."""
    negative = apply_negative_prompt_rules(_BASE_NEGATIVE, _attrs())
    assert negative == _BASE_NEGATIVE


def test_apply_negative_prompt_rules_multiple_matching_rules_combine() -> None:
    """A style matching MORE THAN ONE rule (e.g. a Solid knitwear sweater) gets every matching
    rule's terms, not just the first match -- never a hardcoded per-style branch."""
    attrs = _attrs(
        garment_category="Sweater", construction_group="Knitwear", pattern_or_finish="Solid"
    )
    negative = apply_negative_prompt_rules(_BASE_NEGATIVE, attrs)
    for term in ("floral", "print", "close-up", "fabric swatch"):
        assert term in negative


def test_apply_negative_prompt_rules_is_idempotent() -> None:
    """Calling twice on the same inputs does not duplicate terms."""
    attrs = _attrs(pattern_or_finish="Solid")
    once = apply_negative_prompt_rules(_BASE_NEGATIVE, attrs)
    twice = apply_negative_prompt_rules(once, attrs)
    assert once == twice
    assert twice.count("floral") == 1


def test_apply_negative_prompt_rules_does_not_duplicate_already_present_terms() -> None:
    """A term already present in the base negative_prompt (e.g. from an existing per-style
    strengthening step) is not duplicated."""
    negative_with_person = _BASE_NEGATIVE + ", person, human"
    result = apply_negative_prompt_rules(
        negative_with_person, _attrs(garment_category="Underwear bottom")
    )
    assert result.count("person") == 1
    assert result.count("human") == 1
    assert "model" in result  # still-missing terms from the same rule ARE added
    assert "body" in result


def test_negative_prompt_rules_table_is_generic_not_keyed_by_style_id() -> None:
    """The rule table's predicates take only a `StyleAttributes` dict -- never a style_id/style_key
    -- so it applies correctly to a future style with a matching attribute profile regardless of
    which specific styles a future retraining run selects."""
    for predicate, terms in NEGATIVE_PROMPT_RULES:
        assert predicate(_attrs(pattern_or_finish="Solid", construction_group="Knitwear")) in (
            True,
            False,
        )
        assert isinstance(terms, tuple)
        assert all(isinstance(t, str) for t in terms)
