"""Tests for the `concept-qc` skill's verdict engine (`skills/concept-qc/run_qc.py`).

The skill module lives outside `src/nss/` on purpose (see that module's docstring) so it's loaded
here via an explicit `sys.path` insertion rather than a normal `nss.*` package import -- mirrors
`tests/test_style_brief_generate_brief.py`'s identical pattern for the `style-brief` skill.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SKILL_DIR = Path(__file__).resolve().parents[1] / "skills" / "concept-qc"
if str(_SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(_SKILL_DIR))

from run_qc import (  # noqa: E402 -- import must follow sys.path setup above
    JudgeResult,
    binarize_scores,
    choose_next_retry_value,
    cohens_kappa,
    combine_judges,
    is_self_scoring_contamination,
    mean_score,
    qc_verdict,
    run_judge,
    score_attribute_match,
    score_attributes,
)

# ---------------------------------------------------------------------------
# score_attribute_match / score_attributes
# ---------------------------------------------------------------------------


def test_score_attribute_match_exact() -> None:
    """Case/whitespace-insensitive exact match scores 1.0."""
    assert score_attribute_match("Black", "black") == pytest.approx(1.0)
    assert score_attribute_match("  T-shirt  ", "T-shirt") == pytest.approx(1.0)


def test_score_attribute_match_substring_containment() -> None:
    """One string fully containing the other scores 0.85, not a full match."""
    assert score_attribute_match("a red solid t-shirt", "t-shirt") == pytest.approx(0.85)


def test_score_attribute_match_partial_token_overlap() -> None:
    """Partial word overlap scores the Jaccard ratio -- hand-verifiable: {"basic","tops"} vs.
    {"jersey","basic"} share 1 of 3 union words."""
    assert score_attribute_match("Basic Tops", "Jersey Basic") == pytest.approx(1 / 3)


def test_score_attribute_match_no_overlap_is_zero() -> None:
    """Completely unrelated free text scores 0.0."""
    assert score_attribute_match("Scarf", "T-shirt") == pytest.approx(0.0)


def test_score_attribute_match_empty_string_is_zero() -> None:
    """An empty extraction (judge omitted the field) never crashes -- scores 0.0."""
    assert score_attribute_match("", "Black") == pytest.approx(0.0)
    assert score_attribute_match("Black", "") == pytest.approx(0.0)


def test_score_attributes_missing_key_scores_zero_not_keyerror() -> None:
    """A judge extraction missing a requested dimension scores that dimension 0.0, never crashes."""
    scores = score_attributes(
        {"product_type": "T-shirt"}, {"product_type": "T-shirt", "colour_family": "Black"}
    )
    assert scores["product_type"] == pytest.approx(1.0)
    assert scores["colour_family"] == pytest.approx(0.0)


def test_score_attributes_extra_judge_keys_are_ignored() -> None:
    """A judge extraction with EXTRA keys beyond ground truth is neither rewarded nor penalized."""
    scores = score_attributes(
        {"product_type": "T-shirt", "brand": "H&M"}, {"product_type": "T-shirt"}
    )
    assert set(scores) == {"product_type"}


# ---------------------------------------------------------------------------
# mean_score / binarize_scores
# ---------------------------------------------------------------------------


def test_mean_score_empty_dict_is_zero() -> None:
    """No `ZeroDivisionError` on an empty score dict."""
    assert mean_score({}) == pytest.approx(0.0)


def test_mean_score_averages() -> None:
    assert mean_score({"a": 1.0, "b": 0.0, "c": 0.5}) == pytest.approx(0.5)


def test_binarize_scores_threshold() -> None:
    """Default threshold 0.5: >= passes, < fails."""
    result = binarize_scores({"a": 0.5, "b": 0.49, "c": 1.0, "d": 0.0})
    assert result == {"a": 1, "b": 0, "c": 1, "d": 0}


# ---------------------------------------------------------------------------
# cohens_kappa -- hand-verifiable synthetic example
# ---------------------------------------------------------------------------


def test_cohens_kappa_hand_verified_example() -> None:
    """judge_a=[1,1,0,0], judge_b=[1,0,0,0]: p_o=3/4, a_rate=0.5, b_rate=0.25,
    p_e = 0.5*0.25 + 0.5*0.75 = 0.5, kappa = (0.75-0.5)/(1-0.5) = 0.5 -- hand-computed."""
    kappa = cohens_kappa([1, 1, 0, 0], [1, 0, 0, 0])
    assert kappa == pytest.approx(0.5)


def test_cohens_kappa_perfect_agreement() -> None:
    """Identical calls -> kappa == 1.0."""
    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)


def test_cohens_kappa_degenerate_all_same_is_one() -> None:
    """Both judges call everything 0 (p_e == 1.0 degenerate case) -- defined as kappa=1.0, not a
    ZeroDivisionError."""
    assert cohens_kappa([0, 0, 0], [0, 0, 0]) == pytest.approx(1.0)


def test_cohens_kappa_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        cohens_kappa([1, 0], [1, 0, 1])


def test_cohens_kappa_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        cohens_kappa([], [])


# ---------------------------------------------------------------------------
# is_self_scoring_contamination
# ---------------------------------------------------------------------------


def test_self_scoring_contamination_same_family_blocked() -> None:
    assert is_self_scoring_contamination("gemini", "gemini") is True


def test_self_scoring_contamination_different_family_allowed() -> None:
    assert is_self_scoring_contamination("gemini", "local_sdxl") is False
    assert is_self_scoring_contamination("groq", "gemini") is False
    assert is_self_scoring_contamination("groq", "local_sdxl") is False


# ---------------------------------------------------------------------------
# run_judge
# ---------------------------------------------------------------------------


def test_run_judge_none_caller_is_unavailable() -> None:
    """`caller=None` -> unavailable with the given reason."""
    result = run_judge(
        "groq", None, Path("x.png"), ("a",), {"a": "1"}, "local_sdxl", unavailable_reason="blocked"
    )
    assert result["available"] is False
    assert result["excluded_reason"] == "blocked"


def test_run_judge_contamination_excludes_without_calling() -> None:
    """A same-family judge/backend pairing is excluded WITHOUT ever calling `caller`."""
    calls: list[Path] = []

    def caller(image_path: Path, dims: tuple[str, ...]) -> dict[str, str]:
        calls.append(image_path)
        return {"a": "1"}

    result = run_judge("gemini", caller, Path("x.png"), ("a",), {"a": "1"}, "gemini")
    assert result["available"] is False
    assert "contamination" in result["excluded_reason"]
    assert calls == []


def test_run_judge_successful_call_scores() -> None:
    def caller(image_path: Path, dims: tuple[str, ...]) -> dict[str, str]:
        return {"product_type": "T-shirt"}

    result = run_judge(
        "gemini",
        caller,
        Path("x.png"),
        ("product_type",),
        {"product_type": "T-shirt"},
        "local_sdxl",
    )
    assert result["available"] is True
    assert result["scores"]["product_type"] == pytest.approx(1.0)
    assert result["mean_score"] == pytest.approx(1.0)
    assert result["excluded_reason"] is None


def test_run_judge_caller_exception_is_captured_not_raised() -> None:
    """A judge-call exception (e.g. a blocked/decommissioned model) is captured into
    `excluded_reason` -- never crashes the whole QC gate, never silently discarded."""

    def caller(image_path: Path, dims: tuple[str, ...]) -> dict[str, str]:
        raise RuntimeError("model not found")

    result = run_judge("groq", caller, Path("x.png"), ("a",), {"a": "1"}, "local_sdxl")
    assert result["available"] is False
    assert "model not found" in result["excluded_reason"]


# ---------------------------------------------------------------------------
# combine_judges / qc_verdict
# ---------------------------------------------------------------------------


def _judge(available: bool, mean: float | None) -> JudgeResult:
    return JudgeResult(
        judge_name="x",
        available=available,
        raw_extraction={"a": "b"} if available else None,
        scores={"a": mean} if available else None,
        mean_score=mean,
        excluded_reason=None if available else "unavailable",
    )


def test_combine_judges_averages_available_only() -> None:
    judges = {"gemini": _judge(True, 0.8), "groq": _judge(False, None)}
    consensus, n = combine_judges(judges)
    assert consensus == pytest.approx(0.8)
    assert n == 1


def test_combine_judges_no_judges_available_returns_zero_and_zero() -> None:
    judges = {"gemini": _judge(False, None), "groq": _judge(False, None)}
    consensus, n = combine_judges(judges)
    assert consensus == pytest.approx(0.0)
    assert n == 0


def test_qc_verdict_overall_pass_requires_both_margin_and_fidelity() -> None:
    """margin_band_pass requires BOTH clip_in_band and dino_in_band (see SKILL.md worked
    examples) -- a fidelity-passing, DINOv2-failing concept still fails overall."""
    margin = {"clip_margin": 0.05, "clip_in_band": True, "dino_margin": 0.7, "dino_in_band": False}
    judges = {"gemini": _judge(True, 0.9), "groq": _judge(False, None)}
    verdict = qc_verdict("style-a", margin, judges)
    assert verdict["margin_band_pass"] is False
    assert verdict["fidelity_pass"] is True
    assert verdict["overall_pass"] is False
    assert verdict["label"] == "LLM-consensus (NOT human ground truth)"


def test_qc_verdict_passes_when_both_conditions_met() -> None:
    margin = {"clip_margin": 0.05, "clip_in_band": True, "dino_margin": 0.2, "dino_in_band": True}
    judges = {"gemini": _judge(True, 0.9), "groq": _judge(True, 0.85)}
    verdict = qc_verdict("style-a", margin, judges)
    assert verdict["overall_pass"] is True


# ---------------------------------------------------------------------------
# choose_next_retry_value
# ---------------------------------------------------------------------------

_ALL_VALUES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def test_choose_next_retry_value_undershoot_moves_up() -> None:
    """Primary metric below lower band -> smallest untried value ABOVE last_value."""
    next_value = choose_next_retry_value(
        last_value=0.2,
        tried_values=frozenset({0.2}),
        all_values=_ALL_VALUES,
        primary_metric_value=-0.04,
        primary_lower=0.0325,
        primary_upper=0.0975,
    )
    assert next_value == pytest.approx(0.3)


def test_choose_next_retry_value_overshoot_moves_down_when_possible() -> None:
    """Primary metric above upper band, a smaller untried value exists -> take it."""
    next_value = choose_next_retry_value(
        last_value=0.5,
        tried_values=frozenset({0.2, 0.5}),
        all_values=_ALL_VALUES,
        primary_metric_value=0.13,
        primary_lower=0.0325,
        primary_upper=0.0975,
    )
    assert next_value == pytest.approx(0.4)


def test_choose_next_retry_value_overshoot_at_floor_takes_smallest_untried_above() -> None:
    """Primary metric above upper band but last_value is already the floor of all_values (no
    smaller value can exist) -> smallest untried value above the floor, documented as an
    evidence-gathering attempt, not an expected fix."""
    next_value = choose_next_retry_value(
        last_value=0.2,
        tried_values=frozenset({0.2}),
        all_values=_ALL_VALUES,
        primary_metric_value=0.1029,
        primary_lower=0.0325,
        primary_upper=0.0975,
    )
    assert next_value == pytest.approx(0.3)


def test_choose_next_retry_value_in_band_prefers_secondary_favoring_higher() -> None:
    """Primary already in-band; secondary bounds given, favors_higher=True -> largest untried."""
    next_value = choose_next_retry_value(
        last_value=0.2,
        tried_values=frozenset({0.2}),
        all_values=_ALL_VALUES,
        primary_metric_value=0.05,
        primary_lower=0.0325,
        primary_upper=0.0975,
        secondary_metric_value=0.72,
        secondary_lower=0.14,
        secondary_upper=0.43,
        secondary_favors_higher=True,
    )
    assert next_value == pytest.approx(0.9)


def test_choose_next_retry_value_in_band_prefers_secondary_favoring_lower() -> None:
    next_value = choose_next_retry_value(
        last_value=0.2,
        tried_values=frozenset({0.2}),
        all_values=_ALL_VALUES,
        primary_metric_value=0.05,
        primary_lower=0.0325,
        primary_upper=0.0975,
        secondary_metric_value=0.72,
        secondary_lower=0.14,
        secondary_upper=0.43,
        secondary_favors_higher=False,
    )
    assert next_value == pytest.approx(0.3)


def test_choose_next_retry_value_in_band_no_secondary_defaults_to_step_up() -> None:
    next_value = choose_next_retry_value(
        last_value=0.2,
        tried_values=frozenset({0.2}),
        all_values=_ALL_VALUES,
        primary_metric_value=0.05,
        primary_lower=0.0325,
        primary_upper=0.0975,
    )
    assert next_value == pytest.approx(0.3)


def test_choose_next_retry_value_raises_when_exhausted() -> None:
    with pytest.raises(ValueError, match="already been tried"):
        choose_next_retry_value(
            last_value=0.2,
            tried_values=frozenset(_ALL_VALUES),
            all_values=_ALL_VALUES,
            primary_metric_value=-0.04,
            primary_lower=0.0325,
            primary_upper=0.0975,
        )
