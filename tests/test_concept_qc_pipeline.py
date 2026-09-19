"""Tests for `nss.generate.concept_qc_pipeline` (task C7).

No real API/GPU calls anywhere in this module -- `judge_panel_fn`/`margin_fn`/`generate_fn` are
always fakes injected into `run_qc_with_retries`, and `parse_style_attributes`/
`summarize_calibration` are pure functions tested directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from nss.generate.concept_qc_pipeline import (
    MAX_RETRIES,
    parse_style_attributes,
    run_qc_with_retries,
    summarize_calibration,
)

_ALL_SCALES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _judge_result(available: bool, mean: float | None) -> dict[str, Any]:
    return {
        "judge_name": "x",
        "available": available,
        "raw_extraction": {"product_type": "T-shirt"} if available else None,
        "scores": {"product_type": mean} if available else None,
        "mean_score": mean,
        "excluded_reason": None if available else "unavailable",
    }


# ---------------------------------------------------------------------------
# parse_style_attributes
# ---------------------------------------------------------------------------


def test_parse_style_attributes_maps_five_parts() -> None:
    attrs = parse_style_attributes("Ladieswear || T-shirt || Jersey Basic || Black || Solid")
    assert attrs == {
        "product_type": "T-shirt",
        "colour_family": "Black",
        "graphical_treatment": "Solid",
        "garment_group": "Jersey Basic",
    }


def test_parse_style_attributes_handles_embedded_comma_group() -> None:
    """The underwear style's `Under-, Nightwear` group (embedded comma) still parses correctly --
    `" || "` is the only delimiter, never a bare comma."""
    attrs = parse_style_attributes(
        "Ladieswear || Underwear bottom || Under-, Nightwear || Red || Solid"
    )
    assert attrs["garment_group"] == "Under-, Nightwear"
    assert attrs["product_type"] == "Underwear bottom"


def test_parse_style_attributes_rejects_wrong_part_count() -> None:
    with pytest.raises(ValueError, match="Expected 5"):
        parse_style_attributes("A || B || C")


# ---------------------------------------------------------------------------
# run_qc_with_retries -- max-2-retries cap (mocked judges/generation)
# ---------------------------------------------------------------------------


def test_run_qc_with_retries_passes_immediately_no_retry_needed() -> None:
    """A passing attempt 0 triggers zero retries -- `generate_fn` is never called."""
    generate_calls: list[float] = []

    def judge_panel_fn(image_path: Path) -> dict[str, Any]:
        return {"gemini": _judge_result(True, 0.9), "groq": _judge_result(True, 0.85)}

    def margin_fn(image_path: Path) -> dict[str, Any]:
        return {"clip_margin": 0.05, "clip_in_band": True, "dino_margin": 0.2, "dino_in_band": True}

    def generate_fn(scale: float, attempt_number: int) -> Path:
        generate_calls.append(scale)
        return Path(f"retry_{attempt_number}.png")

    attempts = run_qc_with_retries(
        style_id="style-a",
        original_seed=42,
        original_scale=0.2,
        original_image_path=Path("original.png"),
        original_margin={
            "clip_margin": 0.05,
            "clip_in_band": True,
            "dino_margin": 0.2,
            "dino_in_band": True,
        },
        ground_truth={"product_type": "T-shirt"},
        generation_backend="local_sdxl",
        clip_band=(0.0325, 0.0975),
        dino_band=(0.14, 0.43),
        judge_panel_fn=judge_panel_fn,
        margin_fn=margin_fn,
        generate_fn=generate_fn,
        all_scales=_ALL_SCALES,
    )

    assert len(attempts) == 1
    assert attempts[0]["overall_pass"] is True
    assert generate_calls == []


def test_run_qc_with_retries_caps_at_max_retries_when_always_failing() -> None:
    """A concept that ALWAYS fails still stops at exactly `max_retries` retries (3 attempts total),
    never retries past the cap, and never silently relaxes the pass threshold -- the full retry
    history (all 3 attempts) is preserved, not just the last one."""
    generate_calls: list[float] = []

    def judge_panel_fn(image_path: Path) -> dict[str, Any]:
        return {"gemini": _judge_result(True, 0.9), "groq": _judge_result(False, None)}

    def margin_fn(image_path: Path) -> dict[str, Any]:
        # Always out-of-band on CLIP -- never passes, regardless of scale.
        return {
            "clip_margin": 0.5,
            "clip_in_band": False,
            "dino_margin": 0.7,
            "dino_in_band": False,
        }

    def generate_fn(scale: float, attempt_number: int) -> Path:
        generate_calls.append(scale)
        return Path(f"retry_{attempt_number}.png")

    attempts = run_qc_with_retries(
        style_id="style-b",
        original_seed=42,
        original_scale=0.2,
        original_image_path=Path("original.png"),
        original_margin={
            "clip_margin": 0.5,
            "clip_in_band": False,
            "dino_margin": 0.7,
            "dino_in_band": False,
        },
        ground_truth={"product_type": "T-shirt"},
        generation_backend="local_sdxl",
        clip_band=(0.0325, 0.0975),
        dino_band=(0.14, 0.43),
        judge_panel_fn=judge_panel_fn,
        margin_fn=margin_fn,
        generate_fn=generate_fn,
        all_scales=_ALL_SCALES,
        max_retries=MAX_RETRIES,
    )

    assert len(attempts) == MAX_RETRIES + 1  # original + 2 retries, no more
    assert len(generate_calls) == MAX_RETRIES  # generation only happens for the 2 retries
    assert [a["attempt_number"] for a in attempts] == [0, 1, 2]
    assert all(a["overall_pass"] is False for a in attempts)
    # Full history preserved -- the original AND both retries, not just the final one.
    assert attempts[0]["image_path"] == "original.png"
    assert attempts[1]["image_path"] == "retry_1.png"
    assert attempts[2]["image_path"] == "retry_2.png"
    # The last attempt does not trigger yet another retry (cap respected).
    assert attempts[-1]["retry_triggered"] is False


def test_run_qc_with_retries_passes_on_second_retry_stops_early() -> None:
    """A concept that fails attempt 0 and 1 but passes attempt 2 stops there (no 3rd retry)."""
    call_count = {"n": 0}

    def judge_panel_fn(image_path: Path) -> dict[str, Any]:
        return {"gemini": _judge_result(True, 0.9), "groq": _judge_result(False, None)}

    def margin_fn(image_path: Path) -> dict[str, Any]:
        # margin_fn is only ever called for a RETRY's freshly-generated image (attempt 0 reuses
        # `original_margin` verbatim) -- the first call here scores the 1st retry as passing.
        call_count["n"] += 1
        in_band = call_count["n"] >= 1
        return {
            "clip_margin": 0.05 if in_band else 0.5,
            "clip_in_band": in_band,
            "dino_margin": 0.2 if in_band else 0.7,
            "dino_in_band": in_band,
        }

    def generate_fn(scale: float, attempt_number: int) -> Path:
        return Path(f"retry_{attempt_number}.png")

    attempts = run_qc_with_retries(
        style_id="style-c",
        original_seed=42,
        original_scale=0.2,
        original_image_path=Path("original.png"),
        original_margin={
            "clip_margin": 0.5,
            "clip_in_band": False,
            "dino_margin": 0.7,
            "dino_in_band": False,
        },
        ground_truth={"product_type": "T-shirt"},
        generation_backend="local_sdxl",
        clip_band=(0.0325, 0.0975),
        dino_band=(0.14, 0.43),
        judge_panel_fn=judge_panel_fn,
        margin_fn=margin_fn,
        generate_fn=generate_fn,
        all_scales=_ALL_SCALES,
        max_retries=MAX_RETRIES,
    )

    assert len(attempts) == 2  # original (fails) + 1 retry (passes) -- stops before the 2nd retry
    assert attempts[-1]["overall_pass"] is True
    assert attempts[-1]["retry_triggered"] is False


# ---------------------------------------------------------------------------
# summarize_calibration -- pure pass/fail logic
# ---------------------------------------------------------------------------


def test_summarize_calibration_passes_on_clear_separation() -> None:
    df = pl.DataFrame(
        [
            {
                "judge_name": "gemini",
                "control_type": "positive",
                "available": True,
                "mean_score": 0.9,
            },
            {
                "judge_name": "gemini",
                "control_type": "positive",
                "available": True,
                "mean_score": 0.85,
            },
            {
                "judge_name": "gemini",
                "control_type": "negative",
                "available": True,
                "mean_score": 0.1,
            },
            {
                "judge_name": "gemini",
                "control_type": "negative",
                "available": True,
                "mean_score": 0.2,
            },
        ]
    )
    summary = summarize_calibration(df, pass_gap=0.3)
    assert summary["gemini"]["available"] is True
    assert summary["gemini"]["passed"] is True
    assert summary["gemini"]["gap"] == pytest.approx(0.875 - 0.15)


def test_summarize_calibration_fails_on_insufficient_separation() -> None:
    df = pl.DataFrame(
        [
            {
                "judge_name": "groq",
                "control_type": "positive",
                "available": True,
                "mean_score": 0.6,
            },
            {
                "judge_name": "groq",
                "control_type": "negative",
                "available": True,
                "mean_score": 0.5,
            },
        ]
    )
    summary = summarize_calibration(df, pass_gap=0.3)
    assert summary["groq"]["passed"] is False
    assert summary["groq"]["gap"] == pytest.approx(0.1)


def test_summarize_calibration_is_generic_across_arbitrary_judge_names() -> None:
    """`summarize_calibration` groups purely by whatever `judge_name` values appear in `df` -- not
    hardcoded to `"gemini"`/`"groq"` -- so swapping in a third/different-provider judge (task E6's
    motivating scenario: replacing a dead judge with a differently-named live one) needs zero
    changes to this function."""
    df = pl.DataFrame(
        [
            {
                "judge_name": "openrouter_qwen_vl",
                "control_type": "positive",
                "available": True,
                "mean_score": 0.95,
            },
            {
                "judge_name": "openrouter_qwen_vl",
                "control_type": "negative",
                "available": True,
                "mean_score": 0.05,
            },
        ]
    )
    summary = summarize_calibration(df, pass_gap=0.3)
    assert set(summary.keys()) == {"openrouter_qwen_vl"}
    assert summary["openrouter_qwen_vl"]["passed"] is True
    assert summary["openrouter_qwen_vl"]["gap"] == pytest.approx(0.9)


def test_summarize_calibration_never_available_fails_without_crashing() -> None:
    df = pl.DataFrame(
        [
            {
                "judge_name": "groq",
                "control_type": "positive",
                "available": False,
                "mean_score": None,
            },
            {
                "judge_name": "groq",
                "control_type": "negative",
                "available": False,
                "mean_score": None,
            },
        ]
    )
    summary = summarize_calibration(df, pass_gap=0.3)
    assert summary["groq"]["available"] is False
    assert summary["groq"]["passed"] is False
