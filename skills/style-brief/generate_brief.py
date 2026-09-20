"""Style-profile -> structured design-brief transform (the "style-brief" skill).

Generic, dataset-agnostic transform: given a style profile matching the abstract schema
documented in `SKILL.md` (garment attributes, a historical/predicted performance signal, a
dominant-mechanism narrative, ranked feature drivers, and reference image paths), produce a
structured JSON design brief suitable for driving a text-to-image generation prompt.

This module intentionally knows NOTHING about any specific product taxonomy's column names (no
`index_group_name`, `perceived_colour_master_name`, `graphical_appearance_name`, etc.) -- that
mapping lives entirely in the calling code that adapts a specific dataset onto the generic
`StyleProfile` schema below. See `SKILL.md` for the full schema and two fully worked examples.
"""

from __future__ import annotations

from typing import Any, TypedDict


class StyleDriver(TypedDict):
    """One ranked feature driving a style's predicted/historical performance."""

    feature: str
    importance: float


class PerformanceSignal(TypedDict):
    """Historical/predicted performance signal for a style."""

    predicted_intensity: float
    growth_ratio: float | None


class StyleAttributes(TypedDict):
    """Generic garment-identity attributes. See `SKILL.md` for the abstract schema."""

    garment_category: str
    construction_group: str
    colour_name: str
    pattern_or_finish: str


class StyleProfile(TypedDict, total=False):
    """Generic input to `generate_design_brief`. See `SKILL.md` for the full schema."""

    style_id: str
    attributes: StyleAttributes
    performance_signal: PerformanceSignal
    dominant_mechanism: str
    top_drivers: list[StyleDriver]
    reference_image_paths: list[str]
    sensitivity_tag: str | None


REQUIRED_BRIEF_KEYS: tuple[str, ...] = (
    "style_id",
    "silhouette",
    "fabric_and_hand",
    "colour_direction",
    "detail_and_graphic_treatment",
    "preserve",
    "change",
    "rendered_prompt",
    "negative_prompt",
    "attribute_clause",
    "novelty_clauses",
    "descriptive_clause",
)

_REQUIRED_PROFILE_KEYS: tuple[str, ...] = (
    "style_id",
    "attributes",
    "performance_signal",
    "dominant_mechanism",
    "top_drivers",
    "reference_image_paths",
)
_REQUIRED_ATTRIBUTE_KEYS: tuple[str, ...] = (
    "garment_category",
    "construction_group",
    "colour_name",
    "pattern_or_finish",
)

# Generic textile-family hints, keyed by a lowercase substring matched against
# `construction_group` + `garment_category`. Domain-general fashion vocabulary, not tied to any
# one dataset's taxonomy -- reusable by any style-profile input that mentions these families.
_FABRIC_HAND_HINTS: list[tuple[str, str]] = [
    ("jersey", "a soft, stretch single-knit jersey hand with a relaxed, body-skimming drape"),
    ("knit", "an engineered knit structure with stretch recovery and a soft, insulating hand"),
    ("denim", "a structured woven denim hand with body and minimal stretch"),
    (
        "night",
        "a lightweight, skin-friendly hand prioritising comfort and breathability over structure",
    ),
    (
        "under",
        "a lightweight, stretch-recovery hand prioritising comfort, fit, and breathability",
    ),
    ("woven", "a structured woven hand with a cleaner, lower-stretch drape"),
]
_DEFAULT_FABRIC_HAND_HINT = "a hand and drape consistent with the stated construction family"

_SILHOUETTE_HINTS: list[tuple[str, str]] = [
    ("t-shirt", "relaxed, straight-body silhouette with a simple crew or scoop neckline"),
    ("sweater", "semi-fitted through the body with ribbed hem and cuff finishing"),
    (
        "underwear bottom",
        "brief/hipster-style silhouette, low- to mid-rise, following the body's natural line "
        "without structural embellishment",
    ),
    ("shirt", "structured through the shoulders with a defined collar and placket"),
    ("dress", "fitted through the bodice with a defined waist seam"),
    ("trouser", "straight through the leg with a defined waistband"),
]
_DEFAULT_SILHOUETTE_HINT = "true-to-category proportions, cut for everyday wearability"

_PATTERN_HINTS: dict[str, str] = {
    "solid": (
        "clean solid-ground treatment with no print or graphic; surface interest, if any, comes "
        "from construction details (ribbing, seaming, trims) rather than graphics"
    ),
    "melange": "melange heathered yarn-dye effect for textural depth without a literal print",
    "stripe": "a directional stripe treatment as the defining surface pattern",
}
_DEFAULT_PATTERN_HINT_TEMPLATE = (
    "{pattern} surface treatment, kept as the reference point for any derived concept"
)

_ADJACENT_COLOUR: dict[str, str] = {
    "black": "charcoal or ink navy",
    "white": "off-white or ecru",
    "beige": "warm oatmeal or camel",
    "red": "burgundy or brick",
    "blue": "denim or slate blue",
    "green": "olive or sage",
    "grey": "stone or heather grey",
    "pink": "dusty rose or blush",
    "yellow": "mustard or ochre",
    "brown": "chestnut or tan",
}
_DEFAULT_ADJACENT_COLOUR = "a tonal neighbour within the same colour family"

# Kept category-agnostic and product/garment-focused by default -- a design brief describes the
# garment, not a photoshoot direction, so no style-specific branching is needed to stay tasteful
# for sensitive categories (see SKILL.md's underwear worked example).
_DEFAULT_NEGATIVE_PROMPT = (
    "blurry, distorted proportions, extra limbs, warped seams, low-resolution, watermark, text "
    "overlay, logo, duplicate garments, mismatched colourway, worn by a human model, face, skin, "
    "lifestyle photography"
)


def _match_hint(haystack_parts: list[str], hints: list[tuple[str, str]], default: str) -> str:
    """Return the first hint whose keyword substring matches the joined, lower-cased haystack."""
    haystack = " ".join(haystack_parts).lower()
    for keyword, hint in hints:
        if keyword in haystack:
            return hint
    return default


def _mechanism_is_persistence_led(dominant_mechanism: str) -> bool:
    """Whether `dominant_mechanism`'s narrative leads with persistence rather than seasonality.

    Deliberately checks for the *earlier-occurring* of "persist"/"season" (not e.g. a literal
    `lag_1` feature-name substring, which would false-positive whenever a persistence feature is
    merely mentioned as the LOSING driver in an otherwise seasonal narrative -- concretely: the
    string "seasonal recovery dominates over persistence: ... vs. lag_1 SHAP=..." must classify
    as seasonal-led, and only the ordering of "persist" vs. "season" is robust to that). A
    narrative naming the dominant mechanism first (e.g. "persistence dominates: ...", "seasonal
    recovery dominates over persistence: ...") is the convention this relies on; see SKILL.md's
    `dominant_mechanism` field description.

    Args:
        dominant_mechanism: Free-text mechanism narrative.

    Returns:
        `True` if persistence is the leading/dominant concept (or neither concept is mentioned,
        the conservative default), `False` if seasonality leads.
    """
    text = dominant_mechanism.lower()
    persist_index = text.find("persist")
    season_index = text.find("season")
    if persist_index == -1 and season_index == -1:
        return True  # neither concept named -- default to preserving the combination as a whole
    if persist_index == -1:
        return False
    if season_index == -1:
        return True
    return persist_index < season_index


def validate_style_profile(profile: dict[str, Any]) -> None:
    """Raise `ValueError` if `profile` is missing any required generic-schema key.

    Args:
        profile: Candidate input, expected to match the `StyleProfile` schema documented in
            `SKILL.md`.

    Raises:
        ValueError: naming the first missing top-level or `attributes` sub-key found.
    """
    missing_top = [k for k in _REQUIRED_PROFILE_KEYS if k not in profile]
    if missing_top:
        raise ValueError(f"style profile missing required keys: {missing_top}")
    missing_attrs = [k for k in _REQUIRED_ATTRIBUTE_KEYS if k not in profile["attributes"]]
    if missing_attrs:
        raise ValueError(f"style profile attributes missing required keys: {missing_attrs}")


def generate_design_brief(profile: dict[str, Any]) -> dict[str, Any]:
    """Transform a generic style profile into a structured design brief.

    See `SKILL.md` for the full input/output JSON schema and two fully worked examples. This
    function knows nothing about any specific dataset's column names -- callers adapt their own
    product taxonomy onto the generic `StyleProfile` schema before calling this.

    Args:
        profile: A `StyleProfile`-shaped dict (see this module's `StyleProfile` TypedDict).

    Returns:
        A dict matching `REQUIRED_BRIEF_KEYS`: `style_id`, `silhouette`, `fabric_and_hand`,
        `colour_direction`, `detail_and_graphic_treatment`, `preserve` (list[str]), `change`
        (list[str]), `rendered_prompt` (str), `negative_prompt` (str).

    Raises:
        ValueError: if `profile` is missing required schema keys.
    """
    validate_style_profile(profile)
    attrs = profile["attributes"]
    garment_category = attrs["garment_category"]
    construction_group = attrs["construction_group"]
    colour_name = attrs["colour_name"]
    pattern_or_finish = attrs["pattern_or_finish"]
    dominant_mechanism = profile["dominant_mechanism"]

    silhouette_hint = _match_hint([garment_category], _SILHOUETTE_HINTS, _DEFAULT_SILHOUETTE_HINT)
    silhouette = (
        f"{construction_group} {garment_category.lower()}: {silhouette_hint}. No directional fit "
        "change is signalled by the source data -- silhouette is a `preserve`, not a `change`, "
        "axis."
    )

    fabric_hint = _match_hint(
        [construction_group, garment_category], _FABRIC_HAND_HINTS, _DEFAULT_FABRIC_HAND_HINT
    )
    fabric_and_hand = f"{construction_group} fabrication with {fabric_hint}."

    adjacent_colour = _ADJACENT_COLOUR.get(colour_name.lower(), _DEFAULT_ADJACENT_COLOUR)
    colour_direction = (
        f"{colour_name} as the anchor colour (the attribute that won); {adjacent_colour} as an "
        "optional adjacent/complementary accent, not a replacement for the anchor."
    )

    pattern_hint = _PATTERN_HINTS.get(
        pattern_or_finish.lower(),
        _DEFAULT_PATTERN_HINT_TEMPLATE.format(pattern=pattern_or_finish),
    )
    detail_and_graphic_treatment = pattern_hint

    preserve = [
        f"garment category: {garment_category}",
        f"construction/fabrication family: {construction_group}",
        f"anchor colour: {colour_name}",
        f"surface treatment: {pattern_or_finish}",
        f"overall silhouette proportions ({silhouette_hint})",
    ]
    if _mechanism_is_persistence_led(dominant_mechanism):
        preserve.append(
            "the winning combination as a whole -- performance is driven by persistence of "
            "demand for this exact combination, not a seasonal/calendar cue, so the concept must "
            "read as recognisably the same style rather than a reinterpretation of a seasonal "
            "theme"
        )
    else:
        preserve.append(
            "the seasonal/occasion cue implied by the driver signal, since it materially "
            "contributes to this style's performance"
        )

    change = [
        "one subtle graphic motif, print placement, or embroidery accent not present in the "
        "source style (skip this axis if the source is already a strong graphic/print treatment)",
        "a trim or construction detail (topstitch colour, binding, rib width, hardware finish)",
        "a small proportion tweak within the category's normal range (hem length, cuff width, "
        "rise)",
    ]

    # ATTRIBUTE-FIRST ORDERING (fixing a real defect found by hand): SDXL's CLIP text
    # encoders truncate at 77 tokens, and truncation always drops the TAIL of the prompt. Earlier
    # versions of this template put the four defining attributes (`garment_category`,
    # `construction_group`, `colour_name`, `pattern_or_finish` -- exactly what a copy-check/VLM-
    # fidelity gate scores) deep inside a verbose silhouette/fabric preamble, with novelty content
    # trailing at the very end -- the part most likely to be silently dropped. `attribute_clause`
    # now leads every rendered prompt (short, ~15-20 tokens) so the defining attributes survive
    # even if a downstream token-budget fit (see `nss.generate.prompt_budget`, the actual SDXL-
    # facing enforcement -- this skill stays generation-backend-agnostic and does no token
    # counting itself) has to drop lower-priority clauses. `novelty_clauses`/`descriptive_clause`
    # are exposed separately (not just folded into `rendered_prompt`) so that downstream budget-
    # fitting code can drop individual clauses by priority instead of truncating a flat string.
    attribute_clause = (
        f"{garment_category}, {construction_group.lower()} construction, "
        f"{colour_name.lower()} {pattern_or_finish.lower()}."
    )
    novelty_clauses = [f"Novel accent: {item}." for item in change]
    descriptive_clause = (
        f"Silhouette: {silhouette_hint}. Fabric: {fabric_hint}. Colour accent option: "
        f"{adjacent_colour}. Surface treatment: {pattern_hint}. Product photography of the "
        "garment itself, clean studio background, even lighting, no styling props."
    )
    rendered_prompt = " ".join([attribute_clause, *novelty_clauses, descriptive_clause])

    return {
        "style_id": profile["style_id"],
        "silhouette": silhouette,
        "fabric_and_hand": fabric_and_hand,
        "colour_direction": colour_direction,
        "detail_and_graphic_treatment": detail_and_graphic_treatment,
        "preserve": preserve,
        "change": change,
        "rendered_prompt": rendered_prompt,
        "negative_prompt": _DEFAULT_NEGATIVE_PROMPT,
        "attribute_clause": attribute_clause,
        "novelty_clauses": novelty_clauses,
        "descriptive_clause": descriptive_clause,
    }
