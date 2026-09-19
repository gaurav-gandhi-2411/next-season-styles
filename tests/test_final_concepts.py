from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from nss.generate import final_concepts
from nss.generate.final_concepts import (
    UNDERWEAR_STYLE_KEY,
    _distance_to_band,
    _slugify,
    build_generation_spec,
    gemini_prompt_text,
    generate_gemini_candidate,
    load_design_briefs,
    load_final_three_references,
    select_best_candidate,
    strengthen_underwear_prompts,
)


def test_slugify_is_filesystem_safe_and_lowercase() -> None:
    """No spaces, commas, or uppercase survive -- Windows-safe path segments."""
    slug = _slugify("Ladieswear || T-shirt || Jersey Basic || Black || Solid")
    assert slug == "ladieswear_t-shirt_jersey-basic_black_solid"
    assert " " not in slug
    assert "," not in slug


def test_slugify_handles_underwear_style_key_with_embedded_comma() -> None:
    """The underwear style_key's `Under-, Nightwear` segment (embedded comma+space) still slugs to
    a valid, comma-free path segment."""
    slug = _slugify(UNDERWEAR_STYLE_KEY)
    assert "," not in slug
    assert " " not in slug
    assert slug.startswith("ladieswear_underwear-bottom_")


def test_distance_to_band_zero_when_inside() -> None:
    """A value inside [lower, upper] has distance 0."""
    assert _distance_to_band(0.05, 0.0325, 0.0975) == pytest.approx(0.0)


def test_distance_to_band_below_lower() -> None:
    """A value below the band returns the positive gap to `lower`."""
    assert _distance_to_band(0.01, 0.0325, 0.0975) == pytest.approx(0.0225)


def test_distance_to_band_above_upper() -> None:
    """A value above the band returns the positive gap to `upper`."""
    assert _distance_to_band(0.20, 0.0325, 0.0975) == pytest.approx(0.1025)


def test_strengthen_underwear_prompts_adds_missing_negative_terms() -> None:
    """Bare 'person'/'body'/'nude' terms absent from the design brief's negative_prompt are
    appended; terms already present are not duplicated."""
    rendered = "Underwear bottom, product photography."
    negative = "blurry, worn by a human model, face, skin, lifestyle photography"

    prompt, negative_prompt = strengthen_underwear_prompts(rendered, negative)

    assert "no human model" in prompt.lower()
    assert "flat-lay or mannequin" in prompt.lower()
    assert "person" in negative_prompt
    assert "nude" in negative_prompt
    assert "nudity" in negative_prompt
    # Already-present terms are not duplicated.
    assert negative_prompt.count("face") == 1
    assert negative_prompt.count("skin") == 1


def test_strengthen_underwear_prompts_is_idempotent() -> None:
    """Calling twice on already-strengthened strings is a no-op (no duplicate suffix/terms)."""
    rendered = "Underwear bottom, product photography."
    negative = "blurry, worn by a human model, face, skin, lifestyle photography"

    once = strengthen_underwear_prompts(rendered, negative)
    twice = strengthen_underwear_prompts(*once)

    assert once == twice


def test_build_generation_spec_strengthens_only_underwear_style() -> None:
    """Non-underwear styles pass through their brief's prompt/negative_prompt unchanged."""
    brief = {
        "rendered_prompt": "A sweater concept.",
        "negative_prompt": "blurry, low-resolution",
    }
    prompt, negative_prompt = build_generation_spec(
        "Ladieswear || Sweater || Knitwear || Beige || Melange", brief
    )
    assert prompt == brief["rendered_prompt"]
    assert negative_prompt == brief["negative_prompt"]


def test_build_generation_spec_strengthens_underwear_style() -> None:
    """The underwear style_key triggers `strengthen_underwear_prompts`."""
    brief = {
        "rendered_prompt": "An underwear bottom concept.",
        "negative_prompt": "blurry, worn by a human model",
    }
    prompt, negative_prompt = build_generation_spec(UNDERWEAR_STYLE_KEY, brief)
    assert "flat-lay or mannequin" in prompt.lower()
    assert "person" in negative_prompt


def test_gemini_prompt_text_passes_through_for_non_underwear_styles() -> None:
    """Non-underwear styles' Gemini prompt text is the prompt, unmodified."""
    text = gemini_prompt_text("Ladieswear || Sweater || Knitwear || Beige || Melange", "P", "N")
    assert text == "P"


def test_gemini_prompt_text_folds_negative_prompt_in_for_underwear() -> None:
    """The underwear style folds the negative_prompt into the text prompt (Gemini has no
    negative_prompt slot -- see `nss.generate.backends.generate_concept`)."""
    text = gemini_prompt_text(UNDERWEAR_STYLE_KEY, "P", "no humans, no models")
    assert text.startswith("P")
    assert "do not include a human model" in text.lower()
    assert "no humans, no models" in text


def test_select_best_candidate_prefers_in_band_clip_margin() -> None:
    """A candidate whose CLIP margin falls inside the band beats one outside it, regardless of
    DINOv2 band status (DINOv2 is a reported secondary signal only -- see module docstring)."""
    candidates = [
        {"seed": 42, "clip_margin": 0.15, "dino_margin": 0.60, "image_path": "a.png"},
        {"seed": 43, "clip_margin": 0.05, "dino_margin": 0.70, "image_path": "b.png"},
    ]
    result = select_best_candidate(candidates, clip_band=(0.0325, 0.0975), dino_band=(0.14, 0.43))

    assert result["selected"]["seed"] == 43
    assert result["selected"]["clip_in_band"] is True
    assert result["selected"]["dino_in_band"] is False  # DINOv2 miss reported, not suppressed.
    assert [d["seed"] for d in result["discarded"]] == [42]


def test_select_best_candidate_falls_back_to_closest_to_band_when_none_in_band() -> None:
    """When no candidate is CLIP in-band, the one numerically closest to the band wins -- selection
    is never blocked by a strict DINOv2 requirement (C3's finding: DINOv2 may never be in-band)."""
    candidates = [
        {"seed": 42, "clip_margin": 0.20, "dino_margin": 0.70, "image_path": "a.png"},
        {"seed": 43, "clip_margin": 0.11, "dino_margin": 0.75, "image_path": "b.png"},
    ]
    result = select_best_candidate(candidates, clip_band=(0.0325, 0.0975), dino_band=(0.14, 0.43))

    assert result["selected"]["seed"] == 43  # 0.11 is closer to 0.0975 than 0.20 is.
    assert result["selected"]["clip_in_band"] is False
    assert result["selected"]["dino_in_band"] is False


def test_select_best_candidate_ties_broken_by_seed() -> None:
    """Equal clip_distance_to_band and clip_margin: the lower seed wins, deterministically."""
    candidates = [
        {"seed": 45, "clip_margin": 0.05, "dino_margin": 0.20, "image_path": "a.png"},
        {"seed": 42, "clip_margin": 0.05, "dino_margin": 0.20, "image_path": "b.png"},
    ]
    result = select_best_candidate(candidates, clip_band=(0.0325, 0.0975), dino_band=(0.14, 0.43))

    assert result["selected"]["seed"] == 42


def test_select_best_candidate_rejects_empty_list() -> None:
    """An empty candidate list is a programmer error, not a silent no-op."""
    with pytest.raises(ValueError, match="non-empty"):
        select_best_candidate([], clip_band=(0.0, 1.0), dino_band=(0.0, 1.0))


def test_select_best_candidate_disqualified_seed_never_selected_even_with_best_margin() -> None:
    """A manual visual-QC veto (`disqualified_seeds`) always outranks margin score -- a
    disqualified candidate with the best CLIP margin still loses to a non-disqualified one."""
    candidates = [
        {"seed": 42, "clip_margin": 0.05, "dino_margin": 0.20, "image_path": "a.png"},  # in band
        {"seed": 45, "clip_margin": 0.20, "dino_margin": 0.70, "image_path": "b.png"},  # far off
    ]
    result = select_best_candidate(
        candidates,
        clip_band=(0.0325, 0.0975),
        dino_band=(0.14, 0.43),
        disqualified_seeds=frozenset({42}),
    )

    assert result["selected"]["seed"] == 45
    assert result["selected"]["visual_qc_disqualified"] is False
    disqualified_in_discarded = [d for d in result["discarded"] if d["seed"] == 42]
    assert len(disqualified_in_discarded) == 1
    assert disqualified_in_discarded[0]["visual_qc_disqualified"] is True


def test_select_best_candidate_raises_when_every_candidate_disqualified() -> None:
    """If a manual visual QC veto disqualifies every candidate, this is a hard stop -- never a
    silent fallback to a disqualified image."""
    candidates = [
        {"seed": 42, "clip_margin": 0.05, "dino_margin": 0.20, "image_path": "a.png"},
        {"seed": 43, "clip_margin": 0.06, "dino_margin": 0.21, "image_path": "b.png"},
    ]
    with pytest.raises(ValueError, match="visual_qc_disqualified"):
        select_best_candidate(
            candidates,
            clip_band=(0.0325, 0.0975),
            dino_band=(0.14, 0.43),
            disqualified_seeds=frozenset({42, 43}),
        )


def test_generate_gemini_candidate_returns_none_on_quota_exhausted_api_error(
    tmp_path: Path,
) -> None:
    """A Gemini `APIError` (e.g. 429 RESOURCE_EXHAUSTED quota) is caught and converted to a None
    return, not raised -- one style's quota failure must not crash the rest of C6's Gemini
    appendix."""
    from google.genai import errors as genai_errors

    api_error = genai_errors.ClientError(429, {"error": {"message": "quota exceeded"}})
    with patch.object(final_concepts.backends, "generate_concept", side_effect=api_error):
        result = generate_gemini_candidate(
            "Style A", "prompt text", [Path("ref.jpg")], output_dir=tmp_path
        )

    assert result is None


def test_load_design_briefs_keys_by_style_id(tmp_path: Path) -> None:
    """`load_design_briefs` returns a mapping keyed by each brief's `style_id`."""
    path = tmp_path / "design_briefs.json"
    path.write_text(
        json.dumps(
            [
                {"style_id": "A", "rendered_prompt": "pa", "negative_prompt": "na"},
                {"style_id": "B", "rendered_prompt": "pb", "negative_prompt": "nb"},
            ]
        ),
        encoding="utf-8",
    )

    briefs = load_design_briefs(path)

    assert set(briefs) == {"A", "B"}
    assert briefs["A"]["rendered_prompt"] == "pa"


def test_load_final_three_references_groups_by_style_and_skips_fetch_failures(
    tmp_path: Path,
) -> None:
    """Only `final_rank_*` rows with `fetch_success == True` and an on-disk file are included;
    `control`-role rows and failed fetches are excluded."""
    img_a = tmp_path / "a.jpg"
    img_b = tmp_path / "b.jpg"
    img_a.write_bytes(b"x")
    img_b.write_bytes(b"x")
    missing_img = tmp_path / "missing.jpg"  # never written -- simulates a failed fetch.

    manifest_path = tmp_path / "exemplar_images_final_three.csv"
    manifest_path.write_text(
        "role,style_key,article_id,local_image_path,fetch_success\n"
        f"final_rank_1,Style A,1,{img_a},true\n"
        f"final_rank_1,Style A,2,{missing_img},false\n"
        f"final_rank_2,Style B,3,{img_b},true\n"
        f"control,Style C,4,{img_a},true\n",
        encoding="utf-8",
    )

    grouped = load_final_three_references(manifest_path)

    assert set(grouped) == {"Style A", "Style B"}
    assert grouped["Style A"] == [img_a]
    assert grouped["Style B"] == [img_b]


def test_generate_gemini_candidate_returns_none_on_missing_api_key(tmp_path: Path) -> None:
    """A missing-GEMINI_API_KEY RuntimeError is caught and converted to a None return, not raised
    -- this sub-task must not block the rest of C6."""
    with patch.object(
        final_concepts.backends,
        "generate_concept",
        side_effect=RuntimeError("GEMINI_API_KEY is not set. Create a .env file..."),
    ):
        result = generate_gemini_candidate(
            "Style A", "prompt text", [Path("ref.jpg")], output_dir=tmp_path
        )

    assert result is None


def test_generate_gemini_candidate_reraises_other_runtime_errors(tmp_path: Path) -> None:
    """A real API failure (not a missing-key configuration issue) is NOT swallowed."""
    with (
        patch.object(
            final_concepts.backends,
            "generate_concept",
            side_effect=RuntimeError("Gemini response contained no image part"),
        ),
        pytest.raises(RuntimeError, match="no image part"),
    ):
        generate_gemini_candidate("Style A", "prompt text", [Path("ref.jpg")], output_dir=tmp_path)


def test_generate_gemini_candidate_copies_output_on_success(tmp_path: Path) -> None:
    """A successful Gemini call copies the generated image to a style-labeled destination path."""
    source_path = tmp_path / "source.png"
    source_path.write_bytes(b"fake-png-bytes")
    out_dir = tmp_path / "out"

    with patch.object(final_concepts.backends, "generate_concept", return_value=[source_path]):
        result = generate_gemini_candidate(
            "Ladieswear || Sweater || Knitwear || Beige || Melange",
            "prompt text",
            [Path("ref.jpg")],
            output_dir=out_dir,
        )

    assert result is not None
    assert result.exists()
    assert result.parent == out_dir
