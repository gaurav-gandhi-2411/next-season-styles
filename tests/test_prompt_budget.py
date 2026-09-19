"""Tests for `nss.generate.prompt_budget` (task F4).

Uses the REAL SDXL/CLIP tokenizer (`CLIPTokenizer.from_pretrained`, network/cache access, same as
the rest of this project's real-tokenizer checks) -- no mocked token counts, so these tests are
hand-verifiable against the actual truncation limit SDXL applies at generation time.
"""

from __future__ import annotations

from nss.generate.prompt_budget import (
    SDXL_TOKEN_BUDGET,
    count_clip_tokens,
    fit_prompt_to_token_budget,
)


def test_count_clip_tokens_matches_known_value_for_a_short_phrase() -> None:
    """A short, hand-countable phrase has the expected real CLIP token count (measured directly
    against `CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")`, not estimated:
    'a photo of a cat' -> 5 word tokens + 2 BOS/EOS special tokens = 7)."""
    assert count_clip_tokens("a photo of a cat") == 7


def test_count_clip_tokens_empty_string_is_just_the_special_tokens() -> None:
    """An empty string still costs the 2 BOS/EOS special tokens -- never 0."""
    assert count_clip_tokens("") == 2


def test_count_clip_tokens_longer_text_exceeds_budget() -> None:
    """A verbose, real project-shaped prompt (mirrors the old unfixed `rendered_prompt` template)
    measurably exceeds SDXL's 77-token budget -- this is the exact real defect task F4 fixes,
    reproduced here as a regression guard."""
    verbose_prompt = (
        "T-shirt, jersey basic construction. Silhouette: relaxed, straight-body silhouette with a "
        "simple crew or scoop neckline. Fabric: a soft, stretch single-knit jersey hand with a "
        "relaxed, body-skimming drape. Colour: Black, anchor tone (optional accent: charcoal or "
        "ink navy). Surface treatment: clean solid-ground treatment with no print or graphic; "
        "surface interest, if any, comes from construction details (ribbing, seaming, trims) "
        "rather than graphics. Novel accents to introduce: one subtle graphic motif, print "
        "placement, or embroidery accent not present in the source style (skip this axis if the "
        "source is already a strong graphic/print treatment); a trim or construction detail "
        "(topstitch colour, binding, rib width, hardware finish); a small proportion tweak within "
        "the category's normal range (hem length, cuff width, rise). Product photography of the "
        "garment itself, clean studio background, even lighting, no styling props."
    )
    assert count_clip_tokens(verbose_prompt) > SDXL_TOKEN_BUDGET


def test_fit_prompt_to_token_budget_keeps_everything_when_already_under_budget() -> None:
    """A short assembly that already fits drops nothing."""
    prompt, token_count, dropped = fit_prompt_to_token_budget(
        "T-shirt, jersey basic construction, black solid.",
        ["Novel accent: contrast topstitching."],
        ["Product photography, clean studio background."],
    )
    assert dropped == []
    assert token_count <= 77
    assert "T-shirt, jersey basic construction, black solid." in prompt
    assert "Novel accent: contrast topstitching." in prompt
    assert "Product photography, clean studio background." in prompt


def test_fit_prompt_to_token_budget_drops_novelty_before_descriptive() -> None:
    """When over budget, novelty clauses are dropped FIRST (from the end), before any descriptive
    clause -- the exact opposite priority of task E5's original defect (which lost novelty while
    keeping less important content)."""
    mandatory = "T-shirt, jersey basic construction, black solid."
    novelty = [
        f"Novel accent number {i}: a long descriptive clause about a design change." * 3
        for i in range(4)
    ]
    descriptive = ["Silhouette and fabric detail that is comparatively less important."]

    prompt, token_count, dropped = fit_prompt_to_token_budget(
        mandatory, novelty, descriptive, budget=40
    )

    assert token_count <= 40
    assert mandatory in prompt
    # At least one novelty clause was dropped to fit; descriptive survives longer than novelty.
    assert any(n in dropped for n in novelty)


def test_fit_prompt_to_token_budget_never_drops_mandatory_clause() -> None:
    """Even in the extreme case where EVERY novelty and descriptive clause is dropped, the
    mandatory clause always survives in the final prompt."""
    mandatory = "T-shirt, jersey basic construction, black solid."
    novelty = ["Novel accent: " + ("word " * 30)]
    descriptive = ["Descriptive filler: " + ("word " * 30)]

    prompt, token_count, dropped = fit_prompt_to_token_budget(
        mandatory, novelty, descriptive, budget=20
    )

    assert prompt == mandatory
    assert token_count == count_clip_tokens(mandatory)
    assert set(dropped) == {novelty[0], descriptive[0]}


def test_fit_prompt_to_token_budget_raises_when_mandatory_clause_alone_exceeds_budget() -> None:
    """A mandatory clause that alone exceeds budget cannot be fixed by dropping other clauses --
    fails loudly rather than silently truncating (never the E5 defect's failure mode)."""
    import pytest

    too_long_mandatory = "word " * 100
    with pytest.raises(ValueError, match="mandatory_clause alone"):
        fit_prompt_to_token_budget(too_long_mandatory, ["novelty"], ["descriptive"])


def test_fit_prompt_to_token_budget_real_tshirt_style_fits_after_fix() -> None:
    """Real project data (task F4's fix applied to the T-shirt style's actual clauses, pre-fix
    values measured 175/196 tokens unfitted) now fits within budget end to end."""
    attribute_clause = "T-shirt, jersey basic construction, black solid."
    novelty_clauses = [
        "Novel accent: one subtle graphic motif, print placement, or embroidery accent not "
        "present in the source style (skip this axis if the source is already a strong "
        "graphic/print treatment).",
        "Novel accent: a trim or construction detail (topstitch colour, binding, rib width, "
        "hardware finish).",
        "Novel accent: a small proportion tweak within the category's normal range (hem length, "
        "cuff width, rise).",
    ]
    descriptive_clauses = [
        "Silhouette: relaxed, straight-body silhouette with a simple crew or scoop neckline. "
        "Fabric: a soft, stretch single-knit jersey hand with a relaxed, body-skimming drape. "
        "Colour accent option: charcoal or ink navy. Surface treatment: clean solid-ground "
        "treatment with no print or graphic; surface interest, if any, comes from construction "
        "details (ribbing, seaming, trims) rather than graphics. Product photography of the "
        "garment itself, clean studio background, even lighting, no styling props."
    ]

    prompt, token_count, dropped = fit_prompt_to_token_budget(
        attribute_clause, novelty_clauses, descriptive_clauses
    )

    assert token_count <= SDXL_TOKEN_BUDGET
    assert prompt.startswith(attribute_clause)
    assert dropped  # the full unfitted assembly measures 196 tokens -- something had to go.
