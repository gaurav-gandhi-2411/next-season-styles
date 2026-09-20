"""Tests for `nss.generate.final_concepts_v2`.

No real GPU/API calls anywhere in this module -- `generate_fn`/`score_fn` are always fakes
injected into `run_style_with_retries`, and `select_final_candidate`/`write_results_table` are
pure functions tested directly on hand-built candidate dicts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from nss.generate.final_concepts import Candidate
from nss.generate.final_concepts_v2 import (
    F5_OUTPUT_DIR,
    F5_OUTPUT_TABLE_PATH,
    F5_RETRY_SEED_ROUNDS,
    INITIAL_SEEDS,
    OUTPUT_DIR,
    OUTPUT_TABLE_PATH,
    RETRY_SEED_ROUNDS,
    apply_visual_qc_and_rewrite,
    run_style_with_retries,
    select_final_candidate,
    write_results_table,
)

_STYLE = "Ladieswear || T-shirt || Jersey Basic || Black || Solid"


def _scored(
    seed: int,
    *,
    overall_pass: bool,
    mean_attribute_fidelity: float = 0.8,
    clip_margin: float = 0.10,
    dino_margin: float = 0.60,
    clip_below_copy_anchor: bool = True,
    dino_below_copy_anchor: bool = True,
    retry_round: int = 0,
    style_id: str = _STYLE,
) -> dict[str, Any]:
    """Build a minimal scored-candidate dict (shape of `score_candidate`'s output)."""
    return {
        "style_id": style_id,
        "seed": seed,
        "retry_round": retry_round,
        "image_path": f"seed{seed}.png",
        "clip_margin": clip_margin,
        "clip_copy_anchor_gen": 0.14,
        "clip_copy_anchor_threshold": 0.1223,
        "clip_below_copy_anchor": clip_below_copy_anchor,
        "dino_margin": dino_margin,
        "dino_copy_anchor_gen": 0.68,
        "dino_copy_anchor_threshold": 0.6130,
        "dino_below_copy_anchor": dino_below_copy_anchor,
        "copy_check_pass": clip_below_copy_anchor and dino_below_copy_anchor,
        "gemini_available": True,
        "gemini_mean_score": mean_attribute_fidelity,
        "gemini_excluded_reason": None,
        "groq_available": True,
        "groq_mean_score": mean_attribute_fidelity,
        "groq_excluded_reason": None,
        "mean_attribute_fidelity": mean_attribute_fidelity,
        "n_contributing_judges": 2,
        "fidelity_pass": mean_attribute_fidelity >= 0.75,
        "overall_pass": overall_pass,
    }


# ---------------------------------------------------------------------------
# select_final_candidate
# ---------------------------------------------------------------------------


def test_select_final_candidate_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        select_final_candidate([])


def test_select_final_candidate_picks_highest_fidelity_among_passers() -> None:
    """Among candidates that pass both gates, the highest attribute fidelity wins -- a failing
    candidate with an even higher fidelity score must never win over a passer."""
    candidates = [
        _scored(42, overall_pass=True, mean_attribute_fidelity=0.80),
        _scored(43, overall_pass=True, mean_attribute_fidelity=0.92),
        _scored(44, overall_pass=False, mean_attribute_fidelity=0.99),
    ]
    result = select_final_candidate(candidates)

    assert result["passed"] is True
    assert result["selection_mode"] == "gated_pass"
    assert result["selected"]["seed"] == 43


def test_select_final_candidate_ties_broken_by_lowest_clip_margin() -> None:
    """Equal fidelity among passers: the LOWER CLIP margin (more novel, further below its style's
    copy-anchor threshold) wins."""
    candidates = [
        _scored(42, overall_pass=True, mean_attribute_fidelity=0.85, clip_margin=0.08),
        _scored(43, overall_pass=True, mean_attribute_fidelity=0.85, clip_margin=0.03),
    ]
    result = select_final_candidate(candidates)

    assert result["selected"]["seed"] == 43


def test_select_final_candidate_ties_broken_by_seed_when_fully_tied() -> None:
    candidates = [
        _scored(45, overall_pass=True, mean_attribute_fidelity=0.85, clip_margin=0.05),
        _scored(42, overall_pass=True, mean_attribute_fidelity=0.85, clip_margin=0.05),
    ]
    result = select_final_candidate(candidates)

    assert result["selected"]["seed"] == 42


def test_select_final_candidate_falls_back_to_composite_when_nothing_passes() -> None:
    """No candidate passes both gates: falls back to the composite score, explicitly marked
    `passed=False` -- never silently presented as a pass."""
    candidates = [
        _scored(
            42,
            overall_pass=False,
            mean_attribute_fidelity=0.60,
            clip_below_copy_anchor=True,
            dino_below_copy_anchor=False,
        ),
        _scored(
            43,
            overall_pass=False,
            mean_attribute_fidelity=0.70,
            clip_below_copy_anchor=True,
            dino_below_copy_anchor=True,
        ),
    ]
    result = select_final_candidate(candidates)

    assert result["passed"] is False
    assert result["selection_mode"] == "fallback_no_pass"
    # seed 43: composite = 1 + 1 + 0.70 = 2.70 beats seed 42's 1 + 0 + 0.60 = 1.60
    assert result["selected"]["seed"] == 43


def test_select_final_candidate_fallback_ties_broken_by_seed() -> None:
    candidates = [
        _scored(45, overall_pass=False, mean_attribute_fidelity=0.5),
        _scored(42, overall_pass=False, mean_attribute_fidelity=0.5),
    ]
    result = select_final_candidate(candidates)

    assert result["selected"]["seed"] == 42


# ---------------------------------------------------------------------------
# run_style_with_retries -- adaptive same-scale seed retries (mocked generate/score)
# ---------------------------------------------------------------------------


def test_run_style_with_retries_no_retry_needed_when_initial_batch_passes() -> None:
    """A pass somewhere in the initial 4-seed batch means generate_fn is never called again."""
    generate_calls: list[tuple[int, ...]] = []

    def generate_fn(
        style_id: str,
        prompt: str,
        negative_prompt: str,
        references: list[Path],
        seeds: tuple[int, ...],
        retry_round: int,
    ) -> list[Candidate]:
        generate_calls.append(seeds)
        return [Candidate(style_id=style_id, seed=s, image_path=Path(f"{s}.png")) for s in seeds]

    def score_fn(candidate: Candidate, retry_round: int) -> dict[str, Any]:
        passing = candidate.seed == 44
        return _scored(candidate.seed, overall_pass=passing, retry_round=retry_round)

    result = run_style_with_retries(
        _STYLE, "prompt", "negative", [Path("ref.jpg")], generate_fn, score_fn
    )

    assert generate_calls == [INITIAL_SEEDS]
    assert result["n_retry_rounds_used"] == 0
    assert result["seeds_tried"] == list(INITIAL_SEEDS)
    assert result["selection"]["passed"] is True
    assert result["selection"]["selected"]["seed"] == 44


def test_run_style_with_retries_retries_once_then_stops_on_pass() -> None:
    """Zero passes in the initial batch triggers exactly one retry round; a pass there stops
    further retries (round 2 of `RETRY_SEED_ROUNDS` is never generated)."""
    generate_calls: list[tuple[int, ...]] = []

    def generate_fn(
        style_id: str,
        prompt: str,
        negative_prompt: str,
        references: list[Path],
        seeds: tuple[int, ...],
        retry_round: int,
    ) -> list[Candidate]:
        generate_calls.append(seeds)
        return [Candidate(style_id=style_id, seed=s, image_path=Path(f"{s}.png")) for s in seeds]

    def score_fn(candidate: Candidate, retry_round: int) -> dict[str, Any]:
        passing = retry_round == 1  # nothing in round 0 passes, everything in round 1 does.
        return _scored(candidate.seed, overall_pass=passing, retry_round=retry_round)

    result = run_style_with_retries(
        _STYLE, "prompt", "negative", [Path("ref.jpg")], generate_fn, score_fn
    )

    assert generate_calls == [INITIAL_SEEDS, RETRY_SEED_ROUNDS[0]]
    assert result["n_retry_rounds_used"] == 1
    assert result["seeds_tried"] == list(INITIAL_SEEDS) + list(RETRY_SEED_ROUNDS[0])
    assert result["selection"]["passed"] is True


def test_run_style_with_retries_exhausts_both_rounds_and_reports_failure() -> None:
    """Nothing ever passes: both retry rounds run (the full cap), then the pipeline stops and
    honestly reports a fallback selection -- it never loosens the gate to force a pass."""
    generate_calls: list[tuple[int, ...]] = []

    def generate_fn(
        style_id: str,
        prompt: str,
        negative_prompt: str,
        references: list[Path],
        seeds: tuple[int, ...],
        retry_round: int,
    ) -> list[Candidate]:
        generate_calls.append(seeds)
        return [Candidate(style_id=style_id, seed=s, image_path=Path(f"{s}.png")) for s in seeds]

    def score_fn(candidate: Candidate, retry_round: int) -> dict[str, Any]:
        return _scored(candidate.seed, overall_pass=False, retry_round=retry_round)

    result = run_style_with_retries(
        _STYLE, "prompt", "negative", [Path("ref.jpg")], generate_fn, score_fn
    )

    assert generate_calls == [INITIAL_SEEDS, RETRY_SEED_ROUNDS[0], RETRY_SEED_ROUNDS[1]]
    assert result["n_retry_rounds_used"] == len(RETRY_SEED_ROUNDS)
    all_seeds = list(INITIAL_SEEDS) + list(RETRY_SEED_ROUNDS[0]) + list(RETRY_SEED_ROUNDS[1])
    assert result["seeds_tried"] == all_seeds
    assert result["selection"]["passed"] is False
    assert result["selection"]["selection_mode"] == "fallback_no_pass"


def test_run_style_with_retries_never_changes_scale_between_rounds() -> None:
    """The retry loop's own contract: it only ever varies seeds, passed straight through to
    generate_fn -- no scale/ip_adapter parameter exists on generate_fn's signature at all, so
    there is nothing for the retry loop to drift."""
    seen_seed_batches: list[tuple[int, ...]] = []

    def generate_fn(
        style_id: str,
        prompt: str,
        negative_prompt: str,
        references: list[Path],
        seeds: tuple[int, ...],
        retry_round: int,
    ) -> list[Candidate]:
        seen_seed_batches.append(seeds)
        return [Candidate(style_id=style_id, seed=s, image_path=Path(f"{s}.png")) for s in seeds]

    def score_fn(candidate: Candidate, retry_round: int) -> dict[str, Any]:
        return _scored(candidate.seed, overall_pass=False, retry_round=retry_round)

    run_style_with_retries(_STYLE, "prompt", "negative", [Path("ref.jpg")], generate_fn, score_fn)

    # Every seed across every round/batch is distinct -- no seed is ever retried at a new scale.
    flat = [s for batch in seen_seed_batches for s in batch]
    assert len(flat) == len(set(flat))


# ---------------------------------------------------------------------------
# write_results_table
# ---------------------------------------------------------------------------


def test_write_results_table_flags_the_selected_row(tmp_path: Path) -> None:
    out_path = tmp_path / "final_concepts_v2.csv"
    results = {
        _STYLE: {
            "all_scored": [
                _scored(42, overall_pass=False, mean_attribute_fidelity=0.5),
                _scored(43, overall_pass=True, mean_attribute_fidelity=0.9),
            ],
            "selection": {
                "selected": _scored(43, overall_pass=True, mean_attribute_fidelity=0.9),
                "passed": True,
                "selection_mode": "gated_pass",
                "all_disqualified": False,
            },
            "n_retry_rounds_used": 0,
            "seeds_tried": [42, 43],
        }
    }

    df = write_results_table(results, path=out_path)

    assert out_path.exists()
    assert len(df) == 2
    selected_rows = df.filter(df["is_selected"])
    assert selected_rows["seed"].to_list() == [43]
    assert df["selection_passed"].to_list() == [True, True]


# ---------------------------------------------------------------------------
# select_final_candidate -- visual-QC disqualified_seeds veto
# ---------------------------------------------------------------------------


def test_select_final_candidate_disqualified_seed_never_selected_even_with_best_score() -> None:
    """A manually vetoed seed (e.g. a candidate visually confirmed to show a human model) is
    excluded from selection no matter how good its automated scores are -- the same convention as
    `final_concepts.select_best_candidate`'s `disqualified_seeds`."""
    candidates = [
        _scored(46, overall_pass=True, mean_attribute_fidelity=0.99),  # best score, but vetoed
        _scored(45, overall_pass=True, mean_attribute_fidelity=0.80),
    ]
    result = select_final_candidate(candidates, disqualified_seeds=frozenset({46}))

    assert result["selected"]["seed"] == 45
    assert result["all_disqualified"] is False


def test_select_final_candidate_disqualified_applies_to_fallback_pool_too() -> None:
    """The veto also excludes a disqualified seed from the no-pass composite fallback ranking."""
    candidates = [
        _scored(46, overall_pass=False, mean_attribute_fidelity=0.99),  # best composite, vetoed
        _scored(45, overall_pass=False, mean_attribute_fidelity=0.50),
    ]
    result = select_final_candidate(candidates, disqualified_seeds=frozenset({46}))

    assert result["selected"]["seed"] == 45
    assert result["passed"] is False


def test_select_final_candidate_all_disqualified_falls_back_to_full_pool_not_a_crash() -> None:
    """Unlike the `final_concepts` veto (which raises when every candidate is disqualified), this
    veto degrades gracefully to the full pool and flags `all_disqualified=True` -- reporting a
    genuine failure with mechanism, never crashing the pipeline, when an entire style's candidates
    all fail visual inspection (e.g. this run's Sweater style)."""
    candidates = [
        _scored(42, overall_pass=False, mean_attribute_fidelity=0.50),
        _scored(43, overall_pass=False, mean_attribute_fidelity=0.80),
    ]
    result = select_final_candidate(candidates, disqualified_seeds=frozenset({42, 43}))

    assert result["all_disqualified"] is True
    assert result["selected"]["seed"] == 43  # still the best-available by composite score


# ---------------------------------------------------------------------------
# apply_visual_qc_and_rewrite -- post-hoc reselection over already-logged scores
# ---------------------------------------------------------------------------


def test_apply_visual_qc_and_rewrite_reselects_around_a_vetoed_seed(tmp_path: Path) -> None:
    """A style whose best-scoring candidate is vetoed gets re-selected to the best-scoring
    NON-vetoed candidate, purely from already-logged CSV data -- no regeneration."""
    csv_path = tmp_path / "final_concepts_v2.csv"
    write_results_table(
        {
            _STYLE: {
                "all_scored": [
                    _scored(45, overall_pass=False, mean_attribute_fidelity=0.60),
                    _scored(46, overall_pass=False, mean_attribute_fidelity=0.90),
                ],
                "selection": {
                    "selected": _scored(46, overall_pass=False, mean_attribute_fidelity=0.90),
                    "passed": False,
                    "selection_mode": "fallback_no_pass",
                    "all_disqualified": False,
                },
                "n_retry_rounds_used": 0,
                "seeds_tried": [45, 46],
            }
        },
        path=csv_path,
    )

    rewritten = apply_visual_qc_and_rewrite(
        path=csv_path, disqualified_seeds_by_style={_STYLE: frozenset({46})}
    )

    selected = rewritten.filter(pl.col("is_selected"))
    assert selected["seed"].to_list() == [45]
    assert selected["all_disqualified"].to_list() == [False]


def test_apply_visual_qc_and_rewrite_is_idempotent(tmp_path: Path) -> None:
    """Calling twice with the same veto set produces byte-identical reselection results."""
    csv_path = tmp_path / "final_concepts_v2.csv"
    write_results_table(
        {
            _STYLE: {
                "all_scored": [
                    _scored(45, overall_pass=False, mean_attribute_fidelity=0.60),
                    _scored(46, overall_pass=False, mean_attribute_fidelity=0.90),
                ],
                "selection": {
                    "selected": _scored(46, overall_pass=False, mean_attribute_fidelity=0.90),
                    "passed": False,
                    "selection_mode": "fallback_no_pass",
                    "all_disqualified": False,
                },
                "n_retry_rounds_used": 0,
                "seeds_tried": [45, 46],
            }
        },
        path=csv_path,
    )
    vetoes = {_STYLE: frozenset({46})}

    once = apply_visual_qc_and_rewrite(path=csv_path, disqualified_seeds_by_style=vetoes)
    twice = apply_visual_qc_and_rewrite(path=csv_path, disqualified_seeds_by_style=vetoes)

    assert once.equals(twice)


# ---------------------------------------------------------------------------
# Final-run constants -- exactly 4 seeds/style (12 candidates total), own output paths
# ---------------------------------------------------------------------------


def test_f5_retry_seed_rounds_is_empty() -> None:
    """The final run uses exactly 12 candidates (4 seeds x 3 styles) and reports a genuine gate
    failure rather than retrying with fresh seeds -- a regression here would mean the final
    run silently started generating more than 12 candidates."""
    assert F5_RETRY_SEED_ROUNDS == ()


def test_f5_output_paths_are_isolated_from_e5s() -> None:
    """The final run must never overwrite `main()`'s `final_concepts_v2.csv`/
    `data/generated/final_concepts_v2/` deliverable -- both stay independently auditable (mirrors
    `screen_references.py`'s CANONICAL SOURCE convention)."""
    assert F5_OUTPUT_TABLE_PATH != OUTPUT_TABLE_PATH
    assert F5_OUTPUT_DIR != OUTPUT_DIR


# ---------------------------------------------------------------------------
# rescore_f5_judges -- fakes only, no real API calls (judge-quota-exhaustion retry)
# ---------------------------------------------------------------------------


def test_rescore_f5_judges_refreshes_judge_columns_without_touching_copy_check(
    tmp_path: Path,
) -> None:
    """A row whose original run had ZERO contributing judges (both providers quota-exhausted) gets
    fresh judge scores on rescore; `copy_check_pass`/margins (deterministic, CPU-only) are left
    untouched."""
    from nss.generate.final_concepts_v2 import rescore_f5_judges

    csv_path = tmp_path / "final_concepts_v3.csv"
    unscored = _scored(42, overall_pass=False, mean_attribute_fidelity=0.0)
    unscored.update(
        gemini_available=False,
        gemini_mean_score=None,
        gemini_excluded_reason="gemini judge call failed: 429 RESOURCE_EXHAUSTED",
        groq_available=False,
        groq_mean_score=None,
        groq_excluded_reason="groq judge call failed: 429 rate_limit_exceeded",
        n_contributing_judges=0,
    )
    write_results_table(
        {
            _STYLE: {
                "all_scored": [unscored],
                "selection": {
                    "selected": unscored,
                    "passed": False,
                    "selection_mode": "fallback_no_pass",
                    "all_disqualified": False,
                },
                "n_retry_rounds_used": 0,
                "seeds_tried": [42],
            }
        },
        path=csv_path,
    )

    def fake_judge_panel_fn(
        image_path: Path,
        ground_truth: dict[str, str],
        backend: str,
        groq_available: bool,
        groq_detail: str,
    ) -> dict[str, Any]:
        return {
            "gemini": {"available": False, "mean_score": None, "excluded_reason": "still 429"},
            "groq": {"available": True, "mean_score": 0.9, "excluded_reason": None},
        }

    def fake_groq_check_fn() -> tuple[bool, str]:
        return True, "OK"

    def fake_combine_judges_fn(judges: dict[str, Any]) -> tuple[float, int]:
        return 0.9, 1

    def fake_fidelity_pass_fn(scores: dict[str, float], thresholds: dict[str, float]) -> bool:
        return all(scores[name] >= thresholds[name] for name in scores)

    rescored = rescore_f5_judges(
        path=csv_path,
        judge_panel_fn=fake_judge_panel_fn,
        groq_check_fn=fake_groq_check_fn,
        combine_judges_fn=fake_combine_judges_fn,
        fidelity_pass_fn=fake_fidelity_pass_fn,
        fidelity_thresholds={"gemini": 0.625, "groq": 0.4381},
    )

    row = rescored.row(0, named=True)
    assert row["gemini_available"] is False
    assert row["groq_available"] is True
    assert row["groq_mean_score"] == 0.9
    assert row["mean_attribute_fidelity"] == 0.9
    assert row["n_contributing_judges"] == 1
    assert row["fidelity_pass"] is True
    assert row["copy_check_pass"] == unscored["copy_check_pass"]  # untouched
    assert row["clip_margin"] == unscored["clip_margin"]  # untouched
    assert row["overall_pass"] == (row["copy_check_pass"] and True)


def test_rescore_f5_judges_never_re_queries_a_row_that_already_has_a_real_score(
    tmp_path: Path,
) -> None:
    """Regression test for a real defect found and fixed before committing: a row that
    ALREADY has `n_contributing_judges > 0` (a genuine judge score from a prior run) must be left
    byte-identical, never re-queried and silently overwritten by a fresh (possibly failed) attempt.
    """
    from nss.generate.final_concepts_v2 import rescore_f5_judges

    csv_path = tmp_path / "final_concepts_v3.csv"
    already_scored = _scored(42, overall_pass=False, mean_attribute_fidelity=0.2125)
    already_scored.update(
        gemini_available=False,
        gemini_mean_score=None,
        gemini_excluded_reason="gemini judge call failed: 429 RESOURCE_EXHAUSTED",
        groq_available=True,
        groq_mean_score=0.2125,
        groq_excluded_reason=None,
        n_contributing_judges=1,
    )
    write_results_table(
        {
            _STYLE: {
                "all_scored": [already_scored],
                "selection": {
                    "selected": already_scored,
                    "passed": False,
                    "selection_mode": "fallback_no_pass",
                    "all_disqualified": False,
                },
                "n_retry_rounds_used": 0,
                "seeds_tried": [42],
            }
        },
        path=csv_path,
    )

    def fail_if_called(*args: object, **kwargs: object) -> dict[str, Any]:
        raise AssertionError("judge_panel_fn must never be called for an already-scored row")

    rescored = rescore_f5_judges(
        path=csv_path,
        judge_panel_fn=fail_if_called,
        groq_check_fn=lambda: (True, "OK"),
        combine_judges_fn=lambda judges: (0.0, 0),
    )

    row = rescored.row(0, named=True)
    assert row["groq_mean_score"] == 0.2125
    assert row["n_contributing_judges"] == 1
