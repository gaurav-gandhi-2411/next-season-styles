"""Attribute-value-keyed negative-prompt rule table (task F4).

TWO REAL, DOCUMENTED DEFECTS THIS MODULE REPLACES WITH A GENERIC MECHANISM:

1. A Solid-pattern style's generated concept drifted into a floral-lace pattern and lost its
   defining "Solid" surface-treatment attribute -- nothing in that style's negative prompt told
   the model to avoid pattern/print language, because no code anywhere in this project ever
   derived negative-prompt terms FROM a style's `pattern_or_finish` value.
2. `nss.generate.final_concepts`'s Sweater-style candidates were degenerate fabric-texture
   close-up shots (see that module's docstring, "SWEATER MECHANISM"); the fix (excluding
   `close-up, macro, fabric swatch, texture detail, cropped, zoomed`) was hand-typed directly into
   `reports/tables/design_briefs.json` for that ONE style by task E5 -- it is NOT derived from any
   rule, so it would NOT automatically apply to a different knitwear/sweater style a future
   retraining run (Track G) might select instead.

This module replaces both single-style, ad-hoc fixes with a declarative rule table keyed OFF
STYLE ATTRIBUTE VALUES -- never a hardcoded `style_id`/`style_key` string -- so a future style with
a matching attribute profile gets the same protection automatically, with no per-style edit
needed. Dataset-agnostic by construction: every predicate reads only the generic `StyleAttributes`
keys `skills/style-brief/generate_brief.py` already defines (`garment_category`,
`construction_group`, `colour_name`, `pattern_or_finish`) -- never an H&M column name directly --
so this module would work unchanged for any calling code that maps its own taxonomy onto that same
generic schema (mirrors the skill/calling-code split `skills/style-brief/SKILL.md` documents).
"""

from __future__ import annotations

from collections.abc import Callable

StyleAttributes = dict[str, str]
"""Matches `skills/style-brief/generate_brief.py`'s `StyleAttributes` keys: `garment_category`,
`construction_group`, `colour_name`, `pattern_or_finish`."""

NegativePromptRule = tuple[Callable[[StyleAttributes], bool], tuple[str, ...]]


def is_solid(attrs: StyleAttributes) -> bool:
    """True iff the style's surface treatment is exactly "Solid" (case-insensitive, whitespace-
    tolerant)."""
    return attrs["pattern_or_finish"].strip().lower() == "solid"


def is_underwear_or_intimate(attrs: StyleAttributes) -> bool:
    """True iff the garment category or construction family names an underwear/intimates family.

    Same keyword set as `nss.generate.build_design_briefs.infer_sensitivity_tag` (H&M's
    `Under-, Nightwear` construction-group value is the real taxonomy string this needs to match)
    -- kept as a free function here (not imported from that module) so this module has zero
    dependency on H&M-specific calling code, per this module's own dataset-agnostic design intent.

    Args:
        attrs: The style's `StyleAttributes`-shaped dict.

    Returns:
        `True` if either field contains an underwear/intimates keyword.
    """
    haystack = f"{attrs['garment_category']} {attrs['construction_group']}".lower()
    return any(keyword in haystack for keyword in ("underwear", "night", "lingerie", "intimate"))


def is_knitwear_or_sweater(attrs: StyleAttributes) -> bool:
    """True iff the garment category or construction family names a knitwear/sweater family.

    Args:
        attrs: The style's `StyleAttributes`-shaped dict.

    Returns:
        `True` if either field contains a knitwear/sweater keyword.
    """
    haystack = f"{attrs['garment_category']} {attrs['construction_group']}".lower()
    return any(keyword in haystack for keyword in ("knitwear", "sweater"))


# Declarative rule table (task F4 requirement 3): each entry is (predicate, terms-to-exclude).
# Keyed ENTIRELY off attribute VALUES, checked in this fixed order -- a style can match more than
# one rule (e.g. a Solid knitwear sweater matches both rule 1 and rule 3 simultaneously; both sets
# of terms are applied, see `apply_negative_prompt_rules`).
NEGATIVE_PROMPT_RULES: tuple[NegativePromptRule, ...] = (
    (is_solid, ("floral", "lace", "pattern", "print", "embroidery")),
    (is_underwear_or_intimate, ("model", "person", "body", "human")),
    (
        is_knitwear_or_sweater,
        ("close-up", "macro", "fabric swatch", "texture detail", "cropped", "zoomed"),
    ),
)


def apply_negative_prompt_rules(
    negative_prompt: str,
    attrs: StyleAttributes,
    rules: tuple[NegativePromptRule, ...] = NEGATIVE_PROMPT_RULES,
) -> str:
    """Append every matching rule's exclusion terms to `negative_prompt` (idempotent, missing-only).

    Args:
        negative_prompt: The style's base negative prompt.
        attrs: The style's `StyleAttributes`-shaped dict.
        rules: The rule table to apply (`NEGATIVE_PROMPT_RULES` by default).

    Returns:
        `negative_prompt` with every matching rule's terms appended, deduplicated case-
        insensitively against terms already present and against each other -- calling this twice
        in a row on the same inputs is a no-op the second time.
    """
    negative_lower = negative_prompt.lower()
    to_add: list[str] = []
    for predicate, terms in rules:
        if not predicate(attrs):
            continue
        for term in terms:
            if term not in negative_lower and term not in to_add:
                to_add.append(term)
    if not to_add:
        return negative_prompt
    return negative_prompt.rstrip().rstrip(",") + ", " + ", ".join(to_add)
