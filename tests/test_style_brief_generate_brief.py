"""Tests for the `style-brief` skill's transform (`skills/style-brief/generate_brief.py`).

The skill module lives outside `src/nss/` on purpose (see that module's docstring) so it's loaded
here via an explicit `sys.path` insertion rather than a normal `nss.*` package import -- this
mirrors how `nss.generate.build_design_briefs` loads it at runtime (via `importlib`), just via the
simpler mechanism that's adequate for a test module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SKILL_DIR = Path(__file__).resolve().parents[1] / "skills" / "style-brief"
if str(_SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(_SKILL_DIR))

from generate_brief import (  # noqa: E402 -- import must follow sys.path setup above
    REQUIRED_BRIEF_KEYS,
    generate_design_brief,
    validate_style_profile,
)


def _synthetic_profile(**overrides: object) -> dict[str, object]:
    """A minimal, fully-generic `StyleProfile` fixture -- no H&M-specific data anywhere."""
    profile: dict[str, object] = {
        "style_id": "synthetic-style-001",
        "attributes": {
            "garment_category": "Jacket",
            "construction_group": "Outerwear",
            "colour_name": "Navy",
            "pattern_or_finish": "Quilted",
        },
        "performance_signal": {"predicted_intensity": 42.0, "growth_ratio": 1.5},
        "dominant_mechanism": (
            "seasonal recovery (fourier_sin_1) dominates: SHAP=0.30 vs. lag_1 SHAP=0.05"
        ),
        "top_drivers": [
            {"feature": "fourier_sin_1", "importance": 0.30},
            {"feature": "n_active_articles_level", "importance": 0.20},
        ],
        "reference_image_paths": ["images/synthetic_001.jpg", "images/synthetic_002.jpg"],
        "sensitivity_tag": None,
    }
    profile.update(overrides)
    return profile


def test_generate_design_brief_returns_all_required_keys_with_correct_types() -> None:
    """A synthetic, non-H&M input produces a brief matching the full JSON schema's required
    keys and types (rule: schema-transform function, output validated against the schema)."""
    brief = generate_design_brief(_synthetic_profile())

    assert set(REQUIRED_BRIEF_KEYS).issubset(brief.keys())
    assert isinstance(brief["style_id"], str)
    for text_key in (
        "silhouette",
        "fabric_and_hand",
        "colour_direction",
        "detail_and_graphic_treatment",
        "rendered_prompt",
        "negative_prompt",
    ):
        assert isinstance(brief[text_key], str)
        assert brief[text_key]  # non-empty
    for list_key in ("preserve", "change"):
        assert isinstance(brief[list_key], list)
        assert len(brief[list_key]) > 0
        assert all(isinstance(item, str) for item in brief[list_key])


def test_generate_design_brief_preserve_contains_all_four_identity_attributes() -> None:
    """`preserve` must name every identity attribute the skill was told made the style a winner."""
    brief = generate_design_brief(_synthetic_profile())
    preserve_text = " ".join(brief["preserve"])

    assert "Jacket" in preserve_text
    assert "Outerwear" in preserve_text
    assert "Navy" in preserve_text
    assert "Quilted" in preserve_text


def test_generate_design_brief_seasonal_mechanism_preserves_seasonal_cue_not_combination() -> None:
    """A seasonally-dominated `dominant_mechanism` (no 'persist'/'lag' keyword) preserves the
    seasonal cue, not the "winning combination as a whole" persistence framing."""
    brief = generate_design_brief(_synthetic_profile())

    preserve_text = " ".join(brief["preserve"])
    assert "seasonal" in preserve_text.lower()
    assert "winning combination as a whole" not in preserve_text


def test_generate_design_brief_persistence_mechanism_preserves_combination_as_a_whole() -> None:
    """A persistence-dominated `dominant_mechanism` (contains 'lag_1'/'persist') preserves the
    combination as a whole, matching the real underwear-style finding this skill must not distort
    into a false seasonal narrative."""
    profile = _synthetic_profile(
        dominant_mechanism=(
            "persistence (lag_1) dominates: SHAP=0.49 (also the single largest driver overall)"
        )
    )

    brief = generate_design_brief(profile)

    preserve_text = " ".join(brief["preserve"])
    assert "winning combination as a whole" in preserve_text


def test_generate_design_brief_rendered_prompt_has_no_human_model_language() -> None:
    """`rendered_prompt`/`negative_prompt` stay product/garment-focused for every style, by
    default (see SKILL.md Design notes) -- not just for sensitive categories."""
    brief = generate_design_brief(_synthetic_profile(sensitivity_tag="intimate_apparel"))

    for banned_term in ("model wearing", "worn by", " on a woman", " on a man"):
        assert banned_term not in brief["rendered_prompt"].lower()
    assert "worn by a human model" in brief["negative_prompt"]


def test_generate_design_brief_rendered_prompt_leads_with_attribute_clause() -> None:
    """`rendered_prompt` must start with `attribute_clause` (task F4: SDXL/CLIP truncation always
    drops the TAIL of a too-long prompt, so the 4 defining attributes must lead, not trail)."""
    brief = generate_design_brief(_synthetic_profile())

    assert brief["rendered_prompt"].startswith(brief["attribute_clause"])
    assert "Jacket" in brief["attribute_clause"]
    assert "navy" in brief["attribute_clause"].lower()
    assert "quilted" in brief["attribute_clause"].lower()


def test_generate_design_brief_novelty_clauses_come_after_attribute_clause() -> None:
    """Novelty/change-axis content must appear AFTER the defining-attribute clause, never before
    -- the exact ordering defect a real generation run found (task F4)."""
    brief = generate_design_brief(_synthetic_profile())
    prompt = brief["rendered_prompt"]
    attribute_end = prompt.index(brief["attribute_clause"]) + len(brief["attribute_clause"])

    assert len(brief["novelty_clauses"]) == len(brief["change"])
    for clause in brief["novelty_clauses"]:
        assert prompt.index(clause) >= attribute_end


def test_generate_design_brief_attribute_clause_assembles_novelty_and_descriptive_clauses() -> None:
    """`rendered_prompt` is exactly `attribute_clause` + `novelty_clauses` + `descriptive_clause`
    joined -- so a downstream token-budget-fitting caller can reconstruct any subset."""
    brief = generate_design_brief(_synthetic_profile())

    reassembled = " ".join(
        [brief["attribute_clause"], *brief["novelty_clauses"], brief["descriptive_clause"]]
    )
    assert reassembled == brief["rendered_prompt"]


def test_generate_design_brief_unknown_colour_and_pattern_fall_back_to_generic_defaults() -> None:
    """An attribute value with no dedicated hint (e.g. an unusual colour name) still produces a
    valid, non-empty brief via the generic default templates -- portability requirement."""
    profile = _synthetic_profile(
        attributes={
            "garment_category": "Widget",
            "construction_group": "Injection-molded plastic",
            "colour_name": "Chartreuse",
            "pattern_or_finish": "Marbled",
        }
    )

    brief = generate_design_brief(profile)

    assert "Chartreuse" in brief["colour_direction"]
    assert "tonal neighbour within the same colour family" in brief["colour_direction"]
    assert "Marbled" in brief["detail_and_graphic_treatment"]


def test_validate_style_profile_raises_on_missing_top_level_key() -> None:
    profile = _synthetic_profile()
    del profile["dominant_mechanism"]

    with pytest.raises(ValueError, match="dominant_mechanism"):
        validate_style_profile(profile)


def test_validate_style_profile_raises_on_missing_attribute_key() -> None:
    profile = _synthetic_profile()
    del profile["attributes"]["colour_name"]  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="colour_name"):
        validate_style_profile(profile)


def test_generate_design_brief_raises_on_invalid_profile() -> None:
    with pytest.raises(ValueError):
        generate_design_brief({"style_id": "incomplete"})
