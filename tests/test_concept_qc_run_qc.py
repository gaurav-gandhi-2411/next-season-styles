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
    copy_anchor_threshold,
    copy_check,
    copy_check_pass,
    fidelity_pass_from_per_judge,
    is_self_scoring_contamination,
    judge_fidelity_threshold,
    mean_score,
    per_judge_fidelity_pass,
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


def test_qc_verdict_overall_pass_requires_both_copy_check_and_fidelity() -> None:
    """overall_pass is copy_check_pass AND fidelity_pass -- a fidelity-passing,
    DINOv2-copy-check-failing concept still fails overall. `margin_band_pass` (the OLD two-sided
    band) is still computed/reported but does NOT affect `overall_pass`."""
    margin = {"clip_margin": 0.05, "clip_in_band": True, "dino_margin": 0.7, "dino_in_band": False}
    copy = copy_check(
        clip_margin=0.05,
        dino_margin=0.7,
        clip_copy_anchor_gen=0.1,
        dino_copy_anchor_gen=0.5,
    )
    judges = {"gemini": _judge(True, 0.9), "groq": _judge(False, None)}
    verdict = qc_verdict("style-a", margin, copy, judges)
    assert verdict["margin_band_pass"] is False
    assert verdict["copy_check_pass"] is False  # dino_margin 0.7 not below its 0.45 threshold
    assert verdict["fidelity_pass"] is True
    assert verdict["overall_pass"] is False
    assert verdict["label"] == "LLM-consensus (NOT human ground truth)"


def test_qc_verdict_passes_when_both_conditions_met() -> None:
    margin = {"clip_margin": 0.05, "clip_in_band": True, "dino_margin": 0.2, "dino_in_band": True}
    copy = copy_check(
        clip_margin=0.05,
        dino_margin=0.2,
        clip_copy_anchor_gen=0.1,
        dino_copy_anchor_gen=0.5,
    )
    judges = {"gemini": _judge(True, 0.9), "groq": _judge(True, 0.85)}
    verdict = qc_verdict("style-a", margin, copy, judges)
    assert verdict["copy_check_pass"] is True
    assert verdict["overall_pass"] is True


def test_qc_verdict_overall_pass_can_differ_from_margin_band_pass() -> None:
    """A concept OUTSIDE the old two-sided band (margin_band_pass=False, e.g. clip_in_band=False)
    can still pass overall under the new gate, since margin_band_pass no longer gates -- this is
    the concrete demonstration that the old band is diagnostic-only now."""
    margin = {"clip_margin": 0.05, "clip_in_band": False, "dino_margin": 0.2, "dino_in_band": True}
    copy = copy_check(
        clip_margin=0.05,
        dino_margin=0.2,
        clip_copy_anchor_gen=0.1,
        dino_copy_anchor_gen=0.5,
    )
    judges = {"gemini": _judge(True, 0.9), "groq": _judge(True, 0.85)}
    verdict = qc_verdict("style-a", margin, copy, judges)
    assert verdict["margin_band_pass"] is False
    assert verdict["overall_pass"] is True


# ---------------------------------------------------------------------------
# copy_anchor_threshold / copy_check / copy_check_pass -- Gate 1
# ---------------------------------------------------------------------------


def test_copy_anchor_threshold_positive_anchor_matches_naive_multiplication() -> None:
    """For a POSITIVE anchor, the sign-safe formula agrees with the naive `anchor * 0.90`."""
    assert copy_anchor_threshold(0.1359, discount=0.10) == pytest.approx(0.1359 * 0.90)


def test_copy_anchor_threshold_negative_anchor_edge_case() -> None:
    """NEGATIVE-ANCHOR EDGE CASE (this project's real Sweater CLIP anchor, -0.0472): the naive
    `anchor * 0.90` would give -0.0425 -- HIGHER (less negative) than the anchor itself, which
    would make Gate 1 easier to pass than an un-discounted anchor, the wrong direction. The
    sign-safe formula instead gives -0.0519 -- MORE negative (stricter), the correct direction."""
    anchor = -0.04717115908861158
    naive_wrong = anchor * 0.90  # -0.04245..., LESS negative than the anchor -- wrong direction
    correct = copy_anchor_threshold(anchor, discount=0.10)
    assert correct == pytest.approx(-0.051888274997472738)
    assert correct < anchor  # stricter: further from zero in the anchor's own direction
    assert naive_wrong > anchor  # the naive formula is demonstrably backwards here
    assert correct != pytest.approx(naive_wrong)


def test_copy_anchor_threshold_always_stricter_than_anchor_regardless_of_sign() -> None:
    """The general, sign-independent invariant: `copy_anchor_threshold(x) < x` for any nonzero `x`
    and any `discount > 0` -- true for both a positive and a negative anchor."""
    for anchor in (0.1359, 0.6811, -0.04717, -0.0001):
        assert copy_anchor_threshold(anchor, discount=0.10) < anchor


def test_copy_check_negative_anchor_gate_behaves_correctly() -> None:
    """End-to-end Gate-1 check on the real Sweater CLIP numbers: a margin between the naive-wrong
    threshold and the correct threshold must FAIL (still too copy-like), not pass."""
    clip_anchor = -0.04717115908861158
    dino_anchor = 0.07352664197484653
    # A margin exactly at the (incorrect) naive threshold -0.0425 would pass a buggy `* 0.90` gate,
    # but must FAIL the correct, stricter -0.0519 threshold.
    result = copy_check(
        clip_margin=-0.0425,
        dino_margin=0.01,
        clip_copy_anchor_gen=clip_anchor,
        dino_copy_anchor_gen=dino_anchor,
    )
    assert result["clip_below_copy_anchor"] is False
    assert copy_check_pass(result) is False

    # A margin genuinely below the correct threshold passes.
    result_passing = copy_check(
        clip_margin=-0.06,
        dino_margin=0.01,
        clip_copy_anchor_gen=clip_anchor,
        dino_copy_anchor_gen=dino_anchor,
    )
    assert result_passing["clip_below_copy_anchor"] is True
    assert copy_check_pass(result_passing) is True


def test_copy_check_requires_both_spaces() -> None:
    """copy_check_pass requires BOTH clip_below_copy_anchor AND dino_below_copy_anchor."""
    result = copy_check(
        clip_margin=0.05,  # below its threshold (0.09)
        dino_margin=0.6,  # NOT below its threshold (0.45)
        clip_copy_anchor_gen=0.1,
        dino_copy_anchor_gen=0.5,
    )
    assert result["clip_below_copy_anchor"] is True
    assert result["dino_below_copy_anchor"] is False
    assert copy_check_pass(result) is False


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


# ---------------------------------------------------------------------------
# copy_check_pass active_metrics -- the metric-dropping mechanism
# ---------------------------------------------------------------------------


def test_copy_check_pass_default_requires_both_metrics() -> None:
    """Unchanged default behavior: both metrics active, CLIP passes but DINOv2 doesn't -> fail."""
    result = copy_check(
        clip_margin=0.05, dino_margin=0.6, clip_copy_anchor_gen=0.1, dino_copy_anchor_gen=0.5
    )
    assert copy_check_pass(result) is False


def test_copy_check_pass_dropping_non_discriminative_metric_changes_verdict() -> None:
    """Dropping DINOv2 (active_metrics={'clip'}) -- a candidate that only fails on DINOv2 now
    PASSES, since only CLIP is being gated on."""
    result = copy_check(
        clip_margin=0.05, dino_margin=0.6, clip_copy_anchor_gen=0.1, dino_copy_anchor_gen=0.5
    )
    assert result["clip_below_copy_anchor"] is True
    assert result["dino_below_copy_anchor"] is False
    assert copy_check_pass(result, active_metrics=frozenset({"clip"})) is True
    assert copy_check_pass(result, active_metrics=frozenset({"dinov2"})) is False


def test_copy_check_pass_rejects_empty_active_metrics() -> None:
    result = copy_check(
        clip_margin=0.05, dino_margin=0.2, clip_copy_anchor_gen=0.1, dino_copy_anchor_gen=0.5
    )
    with pytest.raises(ValueError, match="non-empty"):
        copy_check_pass(result, active_metrics=frozenset())


def test_copy_check_pass_rejects_unknown_metric_name() -> None:
    result = copy_check(
        clip_margin=0.05, dino_margin=0.2, clip_copy_anchor_gen=0.1, dino_copy_anchor_gen=0.5
    )
    with pytest.raises(ValueError, match="unknown metric"):
        copy_check_pass(result, active_metrics=frozenset({"clip", "resnet"}))


# ---------------------------------------------------------------------------
# judge_fidelity_threshold / per_judge_fidelity_pass / fidelity_pass_from_per_judge
# ---------------------------------------------------------------------------


def test_judge_fidelity_threshold_hand_computed() -> None:
    """This project's real Gemini calibration mean, 0.8333... -- hand-verifiable: 0.75 * 0.8333...
    = 0.625."""
    assert judge_fidelity_threshold(0.8333333333333334, fraction=0.75) == pytest.approx(0.625)


def test_judge_fidelity_threshold_groq_ceiling_below_old_flat_threshold() -> None:
    """This project's real Groq calibration mean, 0.5840686274509804 -- BELOW the old flat 0.75,
    so the old threshold was unattainable for Groq; the corrected threshold sits below its own
    ceiling and is therefore reachable."""
    threshold = judge_fidelity_threshold(0.5840686274509804, fraction=0.75)
    assert threshold == pytest.approx(0.4380514705882353)
    assert threshold < 0.5840686274509804  # reachable: below the judge's own ceiling
    assert 0.75 > 0.5840686274509804  # the OLD flat threshold was not


def test_per_judge_fidelity_pass_hand_verifiable() -> None:
    scores = {"gemini": 0.7, "groq": 0.4}
    thresholds = {"gemini": 0.625, "groq": 0.4381}
    result = per_judge_fidelity_pass(scores, thresholds)
    assert result == {"gemini": True, "groq": False}


def test_per_judge_fidelity_pass_raises_on_missing_threshold() -> None:
    with pytest.raises(KeyError):
        per_judge_fidelity_pass({"gemini": 0.7}, thresholds={})


def test_fidelity_pass_from_per_judge_requires_all_available_judges_to_pass() -> None:
    """Strict joint-AND, same convention as Gate 1: one judge failing its own threshold fails
    the whole Gate-2 verdict, even if another judge comfortably passes."""
    thresholds = {"gemini": 0.625, "groq": 0.4381}
    assert fidelity_pass_from_per_judge({"gemini": 0.9, "groq": 0.9}, thresholds) is True
    assert fidelity_pass_from_per_judge({"gemini": 0.9, "groq": 0.1}, thresholds) is False
    # groq absent (not failed) -- gemini alone is sufficient.
    assert fidelity_pass_from_per_judge({"gemini": 0.9}, thresholds) is True


def test_fidelity_pass_from_per_judge_no_available_judges_is_false() -> None:
    """No fidelity signal at all is never an unconditional pass."""
    assert fidelity_pass_from_per_judge({}, {"gemini": 0.625}) is False


def test_qc_verdict_with_per_judge_thresholds_differs_from_flat_threshold() -> None:
    """A concept where Groq's raw mean_score (0.5) is BELOW the old flat 0.75 but ABOVE Groq's own
    corrected per-judge threshold (0.4381) fails under the old convention but passes Gate 2 under
    the new one -- concrete demonstration the two conventions can disagree."""
    margin = {"clip_margin": 0.05, "clip_in_band": True, "dino_margin": 0.2, "dino_in_band": True}
    copy = copy_check(
        clip_margin=0.05, dino_margin=0.2, clip_copy_anchor_gen=0.1, dino_copy_anchor_gen=0.5
    )
    judges = {
        "groq": JudgeResult(
            judge_name="groq",
            available=True,
            raw_extraction={"a": "b"},
            scores={"a": 0.5},
            mean_score=0.5,
            excluded_reason=None,
        )
    }

    old_verdict = qc_verdict("style-a", margin, copy, judges)
    assert old_verdict["fidelity_pass"] is False  # 0.5 < flat 0.75

    new_verdict = qc_verdict(
        "style-a", margin, copy, judges, per_judge_fidelity_thresholds={"groq": 0.4381}
    )
    assert new_verdict["fidelity_pass"] is True  # 0.5 >= groq's own corrected 0.4381
    assert new_verdict["overall_pass"] is True
