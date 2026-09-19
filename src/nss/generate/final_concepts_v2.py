"""Fix + regenerate the 3 final winning-style concept images (task E5).

WHY A NEW MODULE, NOT A HEAVILY-BRANCHED EDIT OF `final_concepts.py`: C6's `final_concepts.py`
selects on the OLD two-sided real-space CLIP/DINOv2 band (`select_best_candidate`), never runs the
VLM attribute-fidelity judges, and never gates on the E2 copy-check. Reusing it here would mean
bolting the entire E2 gate + a different selection rule + an adaptive per-style retry loop onto a
module whose docstring and tests are explicitly about the C6 band-based rule -- more surface area
changed than added. This module instead REUSES `final_concepts.py`'s still-valid, unchanged
primitives (`load_design_briefs`, `build_generation_spec`, `generate_candidates_for_style`,
`strengthen_underwear_prompts`, `Candidate`, `_slugify` -- `load_final_three_references` itself was
later swapped for `screen_references.load_screened_references`, task F3, see below) and
`concept_qc_pipeline.py`'s still-valid E2 gate primitives (`run_judge_panel`,
`load_copy_anchors_gen`, `parse_style_attributes`, `SKILL`), and adds only what task E5 actually
needs: the corrected prompts (`reports/tables/design_briefs.json`, edited directly -- not
duplicated here), `ip_adapter_scale=0.45`, the full-gate scoring + "best passing, most novel among
passers" selection rule, and the same-scale seed-only retry loop.

WHAT CHANGED FROM C6/C7 (mechanism, not just outcome):
1. Sweater framing: `design_briefs.json`'s Sweater entry now explicitly demands full-garment
   product-catalogue framing (flat-lay/mannequin, complete silhouette in frame) and its
   `negative_prompt` now excludes `close-up, macro, fabric swatch, texture detail, cropped, zoomed`
   -- C6/C7's candidates were all degenerate fabric-texture close-ups because neither the prompt
   nor the negative prompt ever said anything about framing/crop.
2. `ip_adapter_scale`: 0.2 (C6) -> 0.45. C3/E4's sweep evidence showed 0.2 already loses defining
   attributes (black rendering as grey) -- weakening reference conditioning further to chase
   novelty was the wrong lever. Novelty now comes from the prompt (`applied_changes`, below), not
   from starving the IP-Adapter reference.
3. Novelty source: each style's `design_briefs.json` entry now carries an explicit
   `applied_changes` field (>=2 concrete, brief-grounded changes -- see that file) baked directly
   into `rendered_prompt`'s "Novel accents to introduce" clause, so the evidence chain shows
   intentional prompt-driven novelty, not accidental drift from a starved reference.
4. Underwear: `build_generation_spec` (imported unchanged from `final_concepts.py`) still applies
   `strengthen_underwear_prompts` at generation time -- the C6/C7 no-human-model negative-prompt
   strengthening is NOT weakened here.
5. Selection: C6 selected on CLIP-band proximity alone, never ran the VLM judges pre-selection.
   This module scores EVERY candidate against the FULL E2 gate (Gate 1 copy-check + Gate 2 VLM
   attribute fidelity, both judges) before selecting -- see `select_final_candidate`.

DEFECT FOUND AND FIXED DURING THIS TASK, NOT IN THE ORIGINAL BRIEF (documented, not silently
patched): both SDXL text encoders truncate at 77 CLIP tokens (confirmed directly --
`CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")` against the first drafted prompts
logged `Token indices sequence length ... (175 > 77)` and the pipeline's own truncation warning
named the ENTIRE "Novel accents to introduce" clause as the truncated remainder -- i.e. the E5
task's core requirement, 2 concrete named changes per style actually reaching the model, would have
silently failed even though `design_briefs.json` correctly recorded them). `design_briefs.json`'s
3 `rendered_prompt` fields were rewritten to fit within 77 tokens (measured, not guessed -- every
prompt/negative_prompt pair, including the underwear style's `strengthen_underwear_prompts`-
strengthened form, is <= 76 tokens), with the framing/novelty content moved EARLY in each prompt
(truncation always drops the tail) so it survives regardless of any future prompt edit nudging the
count back up.

SWEATER MECHANISM (found during this run's mandatory visual verification, task E5 step 6 -- NOT
a hypothesis, confirmed by directly reading both the reference image and all 8 generated
candidates): despite the prompt/negative_prompt fix, EVERY sweater candidate generated this run
(seeds 42-49, both original and both retry rounds) is still a degenerate fabric-texture close-up,
not a full-garment shot. Root cause is NOT the prompt this time -- it is the IP-Adapter reference
image. `backends.py`'s documented single-image IP-Adapter limitation (module docstring note 1)
means only `references[0]` is ever used for conditioning; for the Sweater style that image
(`data/images/0673677023.jpg`) is ITSELF a close-up fabric-texture product photo (H&M's own
catalogue photography for this article includes texture-detail shots, not just full-garment
shots). At `ip_adapter_scale=0.45`, that reference image's own close-up framing dominates the
generation regardless of how explicitly the text prompt demands "full-garment... NOT a close-up".
This is a reference-image-selection issue, not a prompt-engineering or scale issue, and fixing it
(e.g. choosing a different index into the reference list, or reordering `references`) is OUT OF
SCOPE for task E5's authorized changes (framing prompt fix, scale=0.45, novelty-in-prompt) --
reported here as a genuine, mechanism-backed failure per the task's explicit instruction not to
loosen the gate or silently paper over it.

FIXED IN TASK F3 (not deferred again -- see `nss.generate.screen_references`'s module docstring for
the full mechanism/fix): `main()` below now loads references via
`screen_references.load_screened_references` instead of `final_concepts.
load_final_three_references` -- a VLM-screened, full-garment-only, best-selling-first ordering, so
`reference_images[0]` (the only image `backends._generate_local_sdxl` ever conditions on) is no
longer the Sweater's arbitrary alphabetically-first candidate (which happened to be a texture
close-up) but the best-selling image every reachable judge classified as a full-garment shot.

TWO-PHASE VRAM-SAFE PROCESS, PER SEED-ROUND (not once for the whole run, unlike C6's
`final_concepts.main`): this module's retry loop is ADAPTIVE -- whether a retry round is even
needed depends on whether the previous round's CPU-scored, API-judged results passed. That
scoring can only happen after `free_sdxl_pipeline()`, but the next (possible) retry round needs the
SDXL pipeline again. So each seed round is: generate (loads/reuses the cached pipeline) -> free
-> score (CPU CLIP/DINOv2 + network VLM judges) -- exactly `concept_qc_pipeline.py`'s own
per-retry convention (`generate_retry_candidate` + `free_sdxl_pipeline()` inside `generate_fn`),
extended here to a whole batch of seeds per round instead of one retry at a time.

WIRED TO TASK F4's SECOND-TEXT-ENCODER SUPPORT (post-F4): `main()`'s `generate_fn` closure now
passes `final_concepts.build_prompt_2(style_id)` as `generate_candidates_for_style`'s `prompt_2`
(and reuses the already-built `negative_prompt` as `negative_prompt_2`), so the active E5/F5
generation path benefits from F4's short, budget-safe attribute-clause reinforcement on SDXL's
second text encoder, not just the superseded `final_concepts.main` C6 path.

Usage:
    uv run python -m nss.generate.final_concepts_v2
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from nss.generate import backends, screen_references, vlm_judges
from nss.generate.concept_qc_pipeline import (
    ATTRIBUTE_FIDELITY_THRESHOLD,
    CALIBRATION_RESULTS_PATH,
    COPY_ANCHOR_DISCOUNT,
    SKILL,
    compute_fidelity_thresholds,
    load_copy_anchors_gen,
    parse_style_attributes,
    run_judge_panel,
    summarize_calibration,
)
from nss.generate.derive_margin_band import CONTROL_MANIFEST_PATH, load_control_pool
from nss.generate.final_concepts import (
    Candidate,
    build_generation_spec,
    build_prompt_2,
    generate_candidates_for_style,
    load_design_briefs,
)
from nss.generate.scale_sweep import free_sdxl_pipeline

OUTPUT_DIR = Path("data/generated/final_concepts_v2")
OUTPUT_TABLE_PATH = Path("reports/tables/final_concepts_v2.csv")

IP_ADAPTER_SCALE = 0.45  # see module docstring point 2 -- E4's fix, novelty moved into the prompt.

INITIAL_SEEDS: tuple[int, ...] = (42, 43, 44, 45)
# Each retry round adds 2 fresh, never-before-tried seeds at the SAME scale (no scale drift -- the
# retry-cap discipline task E5/C7 both require: loosening the lever that was already fixed on
# principled grounds would be result-chasing). 2 seeds/round balances GPU time against giving each
# round a real chance to surface a pass; 2 rounds is the task's explicit retry cap.
RETRY_SEED_ROUNDS: tuple[tuple[int, ...], ...] = ((46, 47), (48, 49))
MAX_RETRY_ROUNDS = len(RETRY_SEED_ROUNDS)

# Task F5 -- final generation. The task brief specifies exactly 4 seeds/style (12 candidates
# total) and instructs reporting a genuine gate failure rather than retrying with fresh seeds if a
# style still fails both gates after all 4 seeds -- so F5's run is capped at INITIAL_SEEDS with NO
# adaptive retry rounds (unlike E5's up-to-8-seed sweep, `RETRY_SEED_ROUNDS` above). Writes to its
# OWN output paths so E5's `final_concepts_v2.csv`/`data/generated/final_concepts_v2/` deliverable
# is never touched -- same "new file, don't overwrite a predecessor's deliverable" convention
# `screen_references.py`'s CANONICAL SOURCE note documents.
F5_OUTPUT_DIR = Path("data/generated/final_concepts_v3")
F5_OUTPUT_TABLE_PATH = Path("reports/tables/final_concepts_v3.csv")
F5_RETRY_SEED_ROUNDS: tuple[tuple[int, ...], ...] = ()


def score_candidate(
    candidate: Candidate,
    clip_refs: Sequence[Any],
    clip_control: Sequence[Any],
    dino_refs: Sequence[Any],
    dino_control: Sequence[Any],
    clip_copy_anchor_gen: float,
    dino_copy_anchor_gen: float,
    ground_truth: dict[str, str],
    groq_available: bool,
    groq_detail: str,
    retry_round: int,
    copy_anchor_discount: float = COPY_ANCHOR_DISCOUNT,
    attribute_fidelity_threshold: float = ATTRIBUTE_FIDELITY_THRESHOLD,
    fidelity_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Score one candidate against the FULL E2 gate: Gate 1 (copy-check) AND Gate 2 (VLM fidelity).

    Computes CLIP/DINOv2 margins directly (mirrors `final_concepts.score_candidates`), runs
    `SKILL.copy_check` (Gate 1) against this style's per-style, per-embedding-space
    `copy_anchor_gen` threshold, and runs both VLM judges (Gate 2) via
    `concept_qc_pipeline.run_judge_panel`. Must be called only after `free_sdxl_pipeline()` for the
    round that generated `candidate` -- CLIP/DINOv2 are CPU-only (safe), but this keeps the same
    VRAM-safety discipline as every other scoring call in this project.

    Args:
        candidate: One generated `local_sdxl` candidate.
        clip_refs: Pre-computed CLIP embeddings of the target style's real reference images.
        clip_control: Pre-computed CLIP embeddings of the shared control pool.
        dino_refs: Pre-computed DINOv2 embeddings of the target style's real reference images.
        dino_control: Pre-computed DINOv2 embeddings of the shared control pool.
        clip_copy_anchor_gen: This style's `copy_anchor_gen` CLIP mean (task E1).
        dino_copy_anchor_gen: This style's `copy_anchor_gen` DINOv2 mean (task E1).
        ground_truth: Output of `parse_style_attributes` for this style.
        groq_available: One-time `vlm_judges.check_groq_availability()` result.
        groq_detail: Detail string from that same one-time check.
        retry_round: 0 for the initial batch, 1/2 for retry rounds -- recorded, not used in scoring.
        copy_anchor_discount: See `skills/concept-qc/run_qc.py`'s `copy_anchor_threshold`.
        attribute_fidelity_threshold: Minimum consensus mean attribute fidelity to pass Gate 2
            UNDER THE OLD, flat-threshold convention -- only used when `fidelity_thresholds` is
            `None` (the default, backward-compatible path E5/`main()` still uses unchanged).
        fidelity_thresholds: Task F2's per-judge Gate-2 thresholds
            (`concept_qc_pipeline.compute_fidelity_thresholds`'s output, e.g. `{"gemini": 0.625,
            "groq": 0.4381}`). When supplied (task F5), Gate 2 uses `SKILL.fidelity_pass_from_per_
            judge` (every AVAILABLE judge independently above ITS OWN calibrated threshold) INSTEAD
            OF the old flat-threshold-on-the-consensus-mean check -- mirrors `run_qc.qc_verdict`'s
            own `per_judge_fidelity_thresholds` opt-in exactly (see `SKILL.md`'s "Gate 2" section).
            `None` (the default) preserves `main()`/E5's original behavior unchanged.

    Returns:
        A flat dict: style/seed/retry_round identifiers, both margins, both Gate-1 thresholds and
        pass booleans, both judges' availability/mean score/threshold-used, consensus fidelity,
        Gate-2 pass, and `overall_pass` (Gate 1 AND Gate 2).
    """
    from nss.generate import clip_scoring, dino_scoring
    from nss.generate.margin_scoring import margin as margin_fn

    clip_embedding = clip_scoring.embed_image(candidate.image_path)
    dino_embedding = dino_scoring.embed_image(candidate.image_path)
    clip_margin_value = margin_fn(clip_embedding, clip_refs, clip_control)
    dino_margin_value = margin_fn(dino_embedding, dino_refs, dino_control)

    copy_result = SKILL.copy_check(
        clip_margin_value,
        dino_margin_value,
        clip_copy_anchor_gen,
        dino_copy_anchor_gen,
        copy_anchor_discount,
    )
    copy_pass = SKILL.copy_check_pass(copy_result)

    judges = run_judge_panel(
        candidate.image_path, ground_truth, backends.LOCAL_SDXL, groq_available, groq_detail
    )
    consensus_fidelity, n_contributing = SKILL.combine_judges(judges)
    if fidelity_thresholds is not None:
        fidelity_pass = SKILL.fidelity_pass_from_per_judge(
            SKILL._available_judge_scores(judges), fidelity_thresholds
        )
    else:
        fidelity_pass = consensus_fidelity >= attribute_fidelity_threshold

    return {
        "style_id": candidate.style_id,
        "seed": candidate.seed,
        "retry_round": retry_round,
        "image_path": str(candidate.image_path),
        "clip_margin": clip_margin_value,
        "clip_copy_anchor_gen": clip_copy_anchor_gen,
        "clip_copy_anchor_threshold": copy_result["clip_copy_anchor_threshold"],
        "clip_below_copy_anchor": copy_result["clip_below_copy_anchor"],
        "dino_margin": dino_margin_value,
        "dino_copy_anchor_gen": dino_copy_anchor_gen,
        "dino_copy_anchor_threshold": copy_result["dino_copy_anchor_threshold"],
        "dino_below_copy_anchor": copy_result["dino_below_copy_anchor"],
        "copy_check_pass": copy_pass,
        "gemini_available": judges["gemini"]["available"],
        "gemini_mean_score": judges["gemini"]["mean_score"],
        "gemini_excluded_reason": judges["gemini"]["excluded_reason"],
        "gemini_fidelity_threshold": (fidelity_thresholds or {}).get("gemini"),
        "groq_available": judges["groq"]["available"],
        "groq_mean_score": judges["groq"]["mean_score"],
        "groq_excluded_reason": judges["groq"]["excluded_reason"],
        "groq_fidelity_threshold": (fidelity_thresholds or {}).get("groq"),
        "mean_attribute_fidelity": consensus_fidelity,
        "n_contributing_judges": n_contributing,
        "fidelity_pass": fidelity_pass,
        "overall_pass": copy_pass and fidelity_pass,
    }


def select_final_candidate(
    scored_candidates: list[dict[str, Any]],
    disqualified_seeds: frozenset[int] = frozenset(),
) -> dict[str, Any]:
    """Select the final candidate for one style from every scored attempt across every round.

    SELECTION RULE (task E5): among candidates that PASS BOTH GATES (`overall_pass`), rank by
    HIGHEST `mean_attribute_fidelity`; ties broken by LOWEST `clip_margin` (CLIP is this project's
    primary metric throughout C6/C7/E2 -- a lower CLIP margin sits further below this style's
    Gate-1 copy-anchor threshold, i.e. is the more novel of two otherwise-equally-faithful
    candidates); remaining ties broken by seed, for determinism.

    If NO candidate passes both gates, falls back to a C8-style composite score
    (`float(clip_below_copy_anchor) + float(dino_below_copy_anchor) + mean_attribute_fidelity` --
    the E2-gate analogue of `final_deliverables.select_best_attempt`'s
    `clip_in_band + dino_in_band + mean_attribute_fidelity`, substituting the E2 Gate-1 booleans
    for the OLD two-sided band booleans that composite was originally built on) and returns the
    best-available candidate, explicitly marked `passed=False` -- never silently presented as a
    pass.

    VISUAL-QC VETO (task E5 step 6, same convention as C6's `final_concepts.select_best_candidate`
    `disqualified_seeds`): automated CLIP/DINOv2/VLM scoring cannot detect "does this image show a
    person" or "is this actually a full-garment shot, not a texture close-up" -- a human (here, the
    operating agent) must inspect every candidate and pass disqualifying seeds in explicitly. A
    disqualified candidate is EXCLUDED from both the `passing` and the composite-fallback pools --
    it can never be `selected`, no matter how good its automated scores are -- UNLESS every
    candidate for this style is disqualified, in which case selection falls back to the FULL
    (still-disqualified) pool rather than raising (unlike C6's veto, which hard-stops when every
    candidate is disqualified): task E5 requires reporting a genuine failure with mechanism, not
    crashing the pipeline, when an entire style's candidates all fail visual inspection (e.g. this
    run's Sweater style -- see module docstring SWEATER MECHANISM note). `all_disqualified` in the
    return value flags exactly this case so callers never mistake a fully-veto'd fallback for a
    clean one.

    Args:
        scored_candidates: Output of `score_candidate`, all for the SAME style_id, across every
            seed/retry round actually generated (non-empty).
        disqualified_seeds: Seeds VETOED by a manual visual QC check. Never inferred from the
            automated scores.

    Returns:
        `{"selected": dict, "passed": bool, "selection_mode": "gated_pass" | "fallback_no_pass",
        "all_disqualified": bool}`.

    Raises:
        ValueError: if `scored_candidates` is empty.
    """
    if not scored_candidates:
        raise ValueError("scored_candidates must be non-empty")

    eligible = [c for c in scored_candidates if c["seed"] not in disqualified_seeds]
    all_disqualified = not eligible
    pool = eligible if eligible else scored_candidates

    passing = [c for c in pool if c["overall_pass"]]
    if passing:
        ranked = sorted(
            passing, key=lambda c: (-c["mean_attribute_fidelity"], c["clip_margin"], c["seed"])
        )
        return {
            "selected": ranked[0],
            "passed": True,
            "selection_mode": "gated_pass",
            "all_disqualified": all_disqualified,
        }

    composite_ranked = sorted(
        pool,
        key=lambda c: (
            -(
                float(c["clip_below_copy_anchor"])
                + float(c["dino_below_copy_anchor"])
                + c["mean_attribute_fidelity"]
            ),
            c["seed"],
        ),
    )
    return {
        "selected": composite_ranked[0],
        "passed": False,
        "selection_mode": "fallback_no_pass",
        "all_disqualified": all_disqualified,
    }


def run_style_with_retries(
    style_id: str,
    prompt: str,
    negative_prompt: str,
    reference_images: list[Path],
    generate_fn: Callable[[str, str, str, list[Path], tuple[int, ...], int], list[Candidate]],
    score_fn: Callable[[Candidate, int], dict[str, Any]],
    initial_seeds: tuple[int, ...] = INITIAL_SEEDS,
    retry_seed_rounds: tuple[tuple[int, ...], ...] = RETRY_SEED_ROUNDS,
) -> dict[str, Any]:
    """Generate+score one style's initial seed batch; retry with fresh seeds (same scale) if
    nothing passes both gates, up to `len(retry_seed_rounds)` rounds.

    Never drifts `ip_adapter_scale` across retries -- the retry-cap discipline task E5/C7 both
    require: the scale was already fixed on principled evidence (E4), so a retry only ever adds
    more seeds at that SAME scale, never a different one. `generate_fn`/`score_fn` are injected so
    this function is fully unit-testable with fakes (no real GPU/API calls) -- see
    `tests/test_final_concepts_v2.py`.

    Args:
        style_id: The style's `design_briefs.json` `style_id`.
        prompt: Text prompt (already underwear-strengthened if applicable -- see
            `final_concepts.build_generation_spec`).
        negative_prompt: Negative prompt (same).
        reference_images: IP-Adapter reference images for this style.
        generate_fn: `(style_id, prompt, negative_prompt, reference_images, seeds, retry_round) ->
            list[Candidate]` -- generates one candidate per seed (`retry_round=0` is the initial
            batch, `1..len(retry_seed_rounds)` are retries).
        score_fn: `(candidate, retry_round) -> scored dict` (output of `score_candidate`).
        initial_seeds: Seeds for the initial (round 0) batch.
        retry_seed_rounds: One tuple of fresh seeds per possible retry round, tried in order, only
            as long as needed.

    Returns:
        `{"style_id": str, "all_scored": list[dict], "selection": dict (output of
        `select_final_candidate`), "n_retry_rounds_used": int, "seeds_tried": list[int]}`.
    """
    all_scored: list[dict[str, Any]] = []
    seeds_tried: list[int] = list(initial_seeds)

    candidates = generate_fn(style_id, prompt, negative_prompt, reference_images, initial_seeds, 0)
    all_scored.extend(score_fn(c, 0) for c in candidates)

    n_retry_rounds_used = 0
    for round_idx, seeds in enumerate(retry_seed_rounds, start=1):
        if any(c["overall_pass"] for c in all_scored):
            break
        n_retry_rounds_used = round_idx
        seeds_tried.extend(seeds)
        candidates = generate_fn(
            style_id, prompt, negative_prompt, reference_images, seeds, round_idx
        )
        all_scored.extend(score_fn(c, round_idx) for c in candidates)

    selection = select_final_candidate(all_scored)
    return {
        "style_id": style_id,
        "all_scored": all_scored,
        "selection": selection,
        "n_retry_rounds_used": n_retry_rounds_used,
        "seeds_tried": seeds_tried,
    }


def write_results_table(
    results: dict[str, dict[str, Any]], path: Path = OUTPUT_TABLE_PATH
) -> pl.DataFrame:
    """Flatten every style's full scored history (every round, every seed) into one CSV.

    Args:
        results: `{style_id: output of run_style_with_retries}`.
        path: Destination CSV path.

    Returns:
        The written `pl.DataFrame` (one row per scored candidate, every style).
    """
    rows: list[dict[str, Any]] = []
    for result in results.values():
        selected_image_path = result["selection"]["selected"]["image_path"]
        for scored in result["all_scored"]:
            rows.append(
                {
                    **scored,
                    "is_selected": scored["image_path"] == selected_image_path,
                    "selection_passed": result["selection"]["passed"],
                    "selection_mode": result["selection"]["selection_mode"],
                    "n_retry_rounds_used": result["n_retry_rounds_used"],
                    "all_disqualified": result["selection"]["all_disqualified"],
                }
            )
    df = pl.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)
    return df


# Per-style seeds manually vetoed after visually inspecting (Read tool, every candidate this run
# actually generated -- task E5 step 6) the full 8-seed history of each style. A style absent from
# this dict, or mapped to an empty frozenset, means every one of its candidates passed manual
# visual inspection -- NOT that visual inspection was skipped (see the task report for the
# per-image finding this dict is built from).
#
# T-shirt seeds 46-49 (every round-1/round-2 retry candidate, ALL 4 of them) each show a human
# model in frame -- a direct violation of the design brief's own `negative_prompt` ("worn by a
# human model, face, skin, lifestyle photography"). Seeds 42-45 (the original round-0 batch) do
# not -- confirmed individually, not inferred from the round number.
#
# Underwear bottom: every one of the 8 candidates (both rounds) is confirmed free of any human
# model, worn-on-body shot, or person in frame -- the C6/C7 defect (3 of 4 original candidates
# showed a human model) did NOT recur here. Left absent from this dict (fully qualified).
#
# Sweater: every one of the 8 candidates is a degenerate fabric-texture close-up, not a
# full-garment shot -- see module docstring SWEATER MECHANISM note for why (the sole IP-Adapter
# reference image is itself a close-up). Deliberately left OUT of this dict rather than mapped to
# a disqualify-all-8 entry: `select_final_candidate`'s `all_disqualified` fallback would select the
# exact same composite-ranked candidate either way (there is no non-close-up candidate to prefer),
# so an explicit disqualify-all entry would be a no-op that only obscures the real finding -- the
# genuine failure is reported via `passed=False` + this constant's absence + the task report's
# mechanism note, not via a redundant veto list.
VISUAL_QC_DISQUALIFIED_SEEDS: dict[str, frozenset[int]] = {
    "Ladieswear || T-shirt || Jersey Basic || Black || Solid": frozenset({46, 47, 48, 49}),
}

_DERIVED_SELECTION_COLUMNS = frozenset(
    {
        "is_selected",
        "selection_passed",
        "selection_mode",
        "n_retry_rounds_used",
        "all_disqualified",
    }
)


def apply_visual_qc_and_rewrite(
    path: Path = OUTPUT_TABLE_PATH,
    disqualified_seeds_by_style: dict[str, frozenset[int]] = VISUAL_QC_DISQUALIFIED_SEEDS,
) -> pl.DataFrame:
    """Re-select the final candidate per style from an already-written `final_concepts_v2.csv`,
    applying the manual visual-QC vetoes in `disqualified_seeds_by_style`, and rewrite the file.

    Pure post-processing over already-logged scores -- never regenerates an image, recomputes an
    embedding, or re-calls a VLM judge (mirrors `concept_qc_pipeline.rescore_results_csv`'s
    "re-score already-computed values" convention). Idempotent: the derived selection columns
    (`is_selected`/`selection_passed`/`selection_mode`/`n_retry_rounds_used`/`all_disqualified`)
    are recomputed from the row-level score columns every call, never from each other, so calling
    this twice in a row with the same `disqualified_seeds_by_style` is a no-op.

    Args:
        path: Path to an already-written `final_concepts_v2.csv` (read AND rewritten in place).
        disqualified_seeds_by_style: `{style_id: frozenset[int]}` -- seeds vetoed by manual visual
            inspection, per style. A style absent from this dict is treated as fully qualified.

    Returns:
        The rewritten `pl.DataFrame`.
    """
    df = pl.read_csv(path)
    score_cols = [c for c in df.columns if c not in _DERIVED_SELECTION_COLUMNS]
    style_ids = df["style_id"].unique(maintain_order=True).to_list()

    rows: list[dict[str, Any]] = []
    for style_id in style_ids:
        style_df = df.filter(pl.col("style_id") == style_id)
        scored = style_df.select(score_cols).to_dicts()
        n_retry_rounds_used = int(style_df["n_retry_rounds_used"].max())
        disqualified = disqualified_seeds_by_style.get(style_id, frozenset())
        selection = select_final_candidate(scored, disqualified_seeds=disqualified)
        selected_image_path = selection["selected"]["image_path"]
        for candidate in scored:
            rows.append(
                {
                    **candidate,
                    "is_selected": candidate["image_path"] == selected_image_path,
                    "selection_passed": selection["passed"],
                    "selection_mode": selection["selection_mode"],
                    "n_retry_rounds_used": n_retry_rounds_used,
                    "all_disqualified": selection["all_disqualified"],
                }
            )

    out_df = pl.DataFrame(rows)
    out_df.write_csv(path)
    return out_df


def main() -> None:
    """Run the full E5 pipeline: per-style adaptive generate -> free VRAM -> full-gate score ->
    select -> write, across every winning style."""
    briefs = load_design_briefs()
    style_references = screen_references.load_screened_references()
    control_images = load_control_pool(CONTROL_MANIFEST_PATH)
    copy_anchors = load_copy_anchors_gen()
    groq_available, groq_detail = vlm_judges.check_groq_availability()
    print(f"Groq judge availability: {groq_available} ({groq_detail})")

    from nss.generate import clip_scoring, dino_scoring

    clip_control = [clip_scoring.embed_image(p) for p in control_images]
    dino_control = [dino_scoring.embed_image(p) for p in control_images]

    results: dict[str, dict[str, Any]] = {}
    for style_id, brief in briefs.items():
        prompt, negative_prompt = build_generation_spec(style_id, brief)
        prompt_2 = build_prompt_2(style_id)  # task F4 -- SDXL's second text encoder
        references = style_references[style_id]
        ground_truth = parse_style_attributes(style_id)
        clip_refs = [clip_scoring.embed_image(p) for p in references]
        dino_refs = [dino_scoring.embed_image(p) for p in references]

        print(f"\nStyle: {style_id}")
        print(f"Prompt: {prompt!r}")
        print(f"Negative prompt: {negative_prompt!r}")
        print(f"Prompt (encoder 2): {prompt_2!r}")

        def generate_fn(
            style_id: str,
            prompt: str,
            negative_prompt: str,
            references: list[Path],
            seeds: tuple[int, ...],
            retry_round: int,
            prompt_2: str = prompt_2,
        ) -> list[Candidate]:
            candidates = generate_candidates_for_style(
                style_id,
                prompt,
                negative_prompt,
                references,
                seeds=seeds,
                ip_adapter_scale=IP_ADAPTER_SCALE,
                output_dir=OUTPUT_DIR,
                prompt_2=prompt_2,
                negative_prompt_2=negative_prompt,
            )
            vram_before, vram_after = free_sdxl_pipeline()
            print(
                f"  [round {retry_round}] {len(candidates)} candidate(s) generated, "
                f"VRAM before/after free: {vram_before:.3f}/{vram_after:.3f} GB"
            )
            return candidates

        def score_fn(
            candidate: Candidate,
            retry_round: int,
            style_id: str = style_id,
            clip_refs: Sequence[Any] = clip_refs,
            dino_refs: Sequence[Any] = dino_refs,
            ground_truth: dict[str, str] = ground_truth,
        ) -> dict[str, Any]:
            return score_candidate(
                candidate,
                clip_refs,
                clip_control,
                dino_refs,
                dino_control,
                copy_anchors[style_id]["clip"],
                copy_anchors[style_id]["dinov2"],
                ground_truth,
                groq_available,
                groq_detail,
                retry_round,
            )

        result = run_style_with_retries(
            style_id, prompt, negative_prompt, references, generate_fn, score_fn
        )
        results[style_id] = result

        selected = result["selection"]["selected"]
        clip_t = selected["clip_copy_anchor_threshold"]
        dino_t = selected["dino_copy_anchor_threshold"]
        print(
            f"  [select] seed={selected['seed']} round={selected['retry_round']} "
            f"clip_margin={selected['clip_margin']:.4f} (threshold={clip_t:.4f}) "
            f"dino_margin={selected['dino_margin']:.4f} (threshold={dino_t:.4f}) "
            f"fidelity={selected['mean_attribute_fidelity']:.3f} "
            f"overall_pass={selected['overall_pass']} "
            f"(selection_mode={result['selection']['selection_mode']}, "
            f"n_retry_rounds_used={result['n_retry_rounds_used']}, "
            f"seeds_tried={result['seeds_tried']})"
        )

    write_results_table(results)
    print(f"\nWrote {OUTPUT_TABLE_PATH}")

    print(
        "\n=== Applying manual visual-QC vetoes (task E5 step 6 -- see "
        "VISUAL_QC_DISQUALIFIED_SEEDS docstring for the per-image finding each entry is built "
        "from) ==="
    )
    for style_id, seeds in VISUAL_QC_DISQUALIFIED_SEEDS.items():
        if seeds:
            print(f"  {style_id}: disqualifying seeds {sorted(seeds)}")
    final_df = apply_visual_qc_and_rewrite()
    print(f"Rewrote {OUTPUT_TABLE_PATH} with visual-QC vetoes applied")

    selected_df = final_df.filter(pl.col("is_selected"))
    n_pass = int(selected_df["selection_passed"].sum())
    n_styles = selected_df.height
    print(f"\n=== FINAL: {n_pass}/{n_styles} styles have a fully-passing candidate ===")
    for row in selected_df.iter_rows(named=True):
        status = "PASS" if row["selection_passed"] else "FAIL (fallback, best-available)"
        print(f"  {status}: {row['style_id']} (seed={row['seed']}, {row['image_path']})")


def run_f5(
    output_dir: Path = F5_OUTPUT_DIR,
    output_table_path: Path = F5_OUTPUT_TABLE_PATH,
    retry_seed_rounds: tuple[tuple[int, ...], ...] = F5_RETRY_SEED_ROUNDS,
) -> dict[str, dict[str, Any]]:
    """Task F5 final generation: same F1-F4-corrected gate/prompt/reference pipeline as `main()`
    (E5) -- `ip_adapter_scale=0.45`, `screen_references.load_screened_references`, the token-
    budget-safe prompts, the per-style negative-prompt rules, the full E2 gate (Gate 1 copy-check +
    Gate 2 per-judge-calibrated VLM fidelity) -- but capped at exactly `INITIAL_SEEDS` (4) seeds
    per style / 12 candidates total via `retry_seed_rounds=()`, never `main()`'s adaptive up-to-
    8-seed retry sweep. Reuses every one of `main()`'s primitives verbatim
    (`run_style_with_retries`, `score_candidate`, `select_final_candidate`, `write_results_table`);
    differs from `main()` ONLY in the seed-round count and the output paths
    (`F5_OUTPUT_DIR`/`F5_OUTPUT_TABLE_PATH`, never
    `OUTPUT_DIR`/`OUTPUT_TABLE_PATH` -- E5's deliverable stays untouched and independently
    auditable).

    Does NOT apply any visual-QC veto -- per this project's established two-phase convention (see
    `main()`'s own flow below), call `apply_visual_qc_and_rewrite(output_table_path,
    disqualified_seeds_by_style=...)` separately once the mandatory Read-tool visual inspection of
    every generated candidate (task F5 step 4) is complete.

    Also differs from `main()` in Gate 2's threshold: this function computes task F2's per-judge
    calibrated thresholds (`compute_fidelity_thresholds`, from the already-committed
    `vlm_calibration_results.csv` -- no re-derivation) and passes them to `score_candidate`'s
    `fidelity_thresholds` parameter, so F5 actually applies the corrected Gate 2 `SKILL.md`
    documents as current -- `main()`/E5 (a task run before F2 existed) is unaffected, since that
    parameter defaults to `None` there.

    Args:
        output_dir: Directory to save generated candidate images into.
        output_table_path: Destination CSV for the full scored history.
        retry_seed_rounds: Retry seed rounds to try if nothing passes both gates (empty by default
            -- see module-level comment above `F5_RETRY_SEED_ROUNDS`).

    Returns:
        `{style_id: output of run_style_with_retries}`.
    """
    briefs = load_design_briefs()
    style_references = screen_references.load_screened_references()
    control_images = load_control_pool(CONTROL_MANIFEST_PATH)
    copy_anchors = load_copy_anchors_gen()
    groq_available, groq_detail = vlm_judges.check_groq_availability()
    print(f"Groq judge availability: {groq_available} ({groq_detail})")

    # Task F2's per-judge Gate-2 thresholds, from the already-committed real calibration run
    # (`reports/tables/vlm_calibration_results.csv`) -- NOT re-derived here, per the task brief's
    # explicit "no re-derivation needed". `score_candidate`'s OLD flat-threshold default
    # (`ATTRIBUTE_FIDELITY_THRESHOLD`, 0.75) was never actually achievable for Groq (its own
    # calibration ceiling is 0.5841, BELOW the old flat 0.75 -- see `SKILL.md`'s "Gate 2" section).
    calibration_df = pl.read_csv(CALIBRATION_RESULTS_PATH)
    fidelity_thresholds = compute_fidelity_thresholds(summarize_calibration(calibration_df))
    print(f"Gate 2 per-judge fidelity thresholds (task F2): {fidelity_thresholds}")

    from nss.generate import clip_scoring, dino_scoring

    clip_control = [clip_scoring.embed_image(p) for p in control_images]
    dino_control = [dino_scoring.embed_image(p) for p in control_images]

    results: dict[str, dict[str, Any]] = {}
    for style_id, brief in briefs.items():
        prompt, negative_prompt = build_generation_spec(style_id, brief)
        prompt_2 = build_prompt_2(style_id)  # task F4 -- SDXL's second text encoder
        references = style_references[style_id]
        ground_truth = parse_style_attributes(style_id)
        clip_refs = [clip_scoring.embed_image(p) for p in references]
        dino_refs = [dino_scoring.embed_image(p) for p in references]

        print(f"\nStyle: {style_id}")
        print(f"Prompt: {prompt!r}")
        print(f"Negative prompt: {negative_prompt!r}")
        print(f"Prompt (encoder 2): {prompt_2!r}")

        def generate_fn(
            style_id: str,
            prompt: str,
            negative_prompt: str,
            references: list[Path],
            seeds: tuple[int, ...],
            retry_round: int,
            prompt_2: str = prompt_2,
            output_dir: Path = output_dir,
        ) -> list[Candidate]:
            candidates = generate_candidates_for_style(
                style_id,
                prompt,
                negative_prompt,
                references,
                seeds=seeds,
                ip_adapter_scale=IP_ADAPTER_SCALE,
                output_dir=output_dir,
                prompt_2=prompt_2,
                negative_prompt_2=negative_prompt,
            )
            vram_before, vram_after = free_sdxl_pipeline()
            print(
                f"  [round {retry_round}] {len(candidates)} candidate(s) generated, "
                f"VRAM before/after free: {vram_before:.3f}/{vram_after:.3f} GB"
            )
            return candidates

        def score_fn(
            candidate: Candidate,
            retry_round: int,
            style_id: str = style_id,
            clip_refs: Sequence[Any] = clip_refs,
            dino_refs: Sequence[Any] = dino_refs,
            ground_truth: dict[str, str] = ground_truth,
        ) -> dict[str, Any]:
            return score_candidate(
                candidate,
                clip_refs,
                clip_control,
                dino_refs,
                dino_control,
                copy_anchors[style_id]["clip"],
                copy_anchors[style_id]["dinov2"],
                ground_truth,
                groq_available,
                groq_detail,
                retry_round,
                fidelity_thresholds=fidelity_thresholds,
            )

        result = run_style_with_retries(
            style_id,
            prompt,
            negative_prompt,
            references,
            generate_fn,
            score_fn,
            retry_seed_rounds=retry_seed_rounds,
        )
        results[style_id] = result

        selected = result["selection"]["selected"]
        clip_t = selected["clip_copy_anchor_threshold"]
        dino_t = selected["dino_copy_anchor_threshold"]
        print(
            f"  [select] seed={selected['seed']} round={selected['retry_round']} "
            f"clip_margin={selected['clip_margin']:.4f} (threshold={clip_t:.4f}) "
            f"dino_margin={selected['dino_margin']:.4f} (threshold={dino_t:.4f}) "
            f"fidelity={selected['mean_attribute_fidelity']:.3f} "
            f"overall_pass={selected['overall_pass']} "
            f"(selection_mode={result['selection']['selection_mode']}, "
            f"n_retry_rounds_used={result['n_retry_rounds_used']}, "
            f"seeds_tried={result['seeds_tried']})"
        )

    write_results_table(results, path=output_table_path)
    print(f"\nWrote {output_table_path}")
    return results


def rescore_f5_judges(
    path: Path = F5_OUTPUT_TABLE_PATH,
    judge_panel_fn: Callable[[Path, dict[str, str], str, bool, str], dict[str, Any]] = (
        run_judge_panel
    ),
    groq_check_fn: Callable[[], tuple[bool, str]] = vlm_judges.check_groq_availability,
    combine_judges_fn: Callable[[dict[str, Any]], tuple[float, int]] = SKILL.combine_judges,
    fidelity_pass_fn: Callable[[dict[str, float], dict[str, float]], bool] = (
        SKILL.fidelity_pass_from_per_judge
    ),
    fidelity_thresholds: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Re-run the VLM judge panel ONLY for `run_f5` candidates with ZERO contributing judges last
    time, in place -- no GPU regeneration, no CLIP/DINOv2 re-embedding (both are deterministic/
    CPU-only and untouched here), and (critically) no re-querying a row that already got a REAL
    judge score, which would just as likely overwrite it with a fresh failed attempt as improve it.

    WHY THIS EXISTS (task F5's actual run, 2026-09-19): both judge providers are SHARED, scarce,
    externally-rate-limited resources -- Gemini's free-tier daily cap (20 requests/day) and Groq's
    shared TPD token budget. `run_f5`'s own single pass hit both mid-run: Gemini was already
    exhausted from earlier in this session (confirmed still exhausted by a direct live re-check
    immediately after `run_f5` finished, not assumed) and Groq's TPD budget ran out partway through
    scoring the 12 candidates (visible in each row's `groq_excluded_reason`'s `Used`/`Limit` token
    counts climbing toward the 200000 cap) -- so only 1 of 12 candidates (`n_contributing_judges ==
    1`) got a real judge score from EITHER provider; the other 11 have `n_contributing_judges == 0`.
    Re-running the FULL `run_f5` (GPU regeneration + scoring) just to retry the judge calls would
    burn ~8 minutes of GPU time for zero benefit (the images/margins are already correct and
    deterministic) -- this function isolates the retry to exactly the piece that can still fail:
    judge availability. `judge_panel_fn`/`groq_check_fn`/`combine_judges_fn` are injected so this is
    fully unit-testable with fakes (no real API calls) -- see `tests/test_final_concepts_v2.py`.

    ONLY-MISSING RETRY (mirrors `screen_references.retry_inconclusive_candidates`'s established
    convention exactly): a row already carrying `n_contributing_judges > 0` is left byte-identical
    -- a REAL DEFECT this function's first version had, caught before being committed (see the task
    F5 report): unconditionally re-querying every row, including the one that already had a genuine
    Groq score, silently overwrote that real score with a fresh (failed, quota-exhausted-again)
    attempt the very next time it ran. Safe to call repeatedly as quota recovers over time, same
    idempotency guarantee `retry_inconclusive_candidates` documents.

    Does NOT recompute selection -- call `apply_visual_qc_and_rewrite(path, ...)` afterward (same
    two-phase convention `run_f5`/`main` already use) to re-select per style from the refreshed
    judge scores.

    Args:
        path: Path to an already-written `run_f5` output CSV (read AND rewritten in place).
        judge_panel_fn: `concept_qc_pipeline.run_judge_panel`'s signature -- injected for testing.
        groq_check_fn: `vlm_judges.check_groq_availability`'s signature -- injected for testing.
        combine_judges_fn: `SKILL.combine_judges`'s signature -- injected for testing.
        fidelity_pass_fn: `SKILL.fidelity_pass_from_per_judge`'s signature -- injected for testing.
        fidelity_thresholds: Task F2's per-judge Gate-2 thresholds (see `run_f5`'s docstring for
            why this project uses per-judge thresholds, not a flat one). `None` (the default)
            loads them from the already-committed `vlm_calibration_results.csv`, same as `run_f5`.

    Returns:
        The rewritten `pl.DataFrame` (judge/fidelity/`overall_pass` columns refreshed only for
        rows that had `n_contributing_judges == 0`; every other row/column left untouched).
    """
    if fidelity_thresholds is None:
        calibration_df = pl.read_csv(CALIBRATION_RESULTS_PATH)
        fidelity_thresholds = compute_fidelity_thresholds(summarize_calibration(calibration_df))

    df = pl.read_csv(path)
    groq_available, groq_detail = groq_check_fn()
    print(f"Groq judge availability: {groq_available} ({groq_detail})")

    ground_truth_cache: dict[str, dict[str, str]] = {}
    rows: list[dict[str, Any]] = []
    for row in df.iter_rows(named=True):
        if row["n_contributing_judges"] > 0:
            rows.append(row)
            continue
        style_id = row["style_id"]
        if style_id not in ground_truth_cache:
            ground_truth_cache[style_id] = parse_style_attributes(style_id)
        judges = judge_panel_fn(
            Path(row["image_path"]),
            ground_truth_cache[style_id],
            backends.LOCAL_SDXL,
            groq_available,
            groq_detail,
        )
        consensus_fidelity, n_contributing = combine_judges_fn(judges)
        available_scores = {
            name: judges[name]["mean_score"]
            for name in ("gemini", "groq")
            if judges[name]["available"] and judges[name]["mean_score"] is not None
        }
        fidelity_pass = fidelity_pass_fn(available_scores, fidelity_thresholds)
        print(
            f"  [rescore] style={style_id!r} seed={row['seed']} "
            f"gemini_available={judges['gemini']['available']} "
            f"groq_available={judges['groq']['available']} "
            f"fidelity={consensus_fidelity:.3f} fidelity_pass={fidelity_pass}"
        )
        rows.append(
            {
                **row,
                "gemini_available": judges["gemini"]["available"],
                "gemini_mean_score": judges["gemini"]["mean_score"],
                "gemini_excluded_reason": judges["gemini"]["excluded_reason"],
                "gemini_fidelity_threshold": fidelity_thresholds.get("gemini"),
                "groq_available": judges["groq"]["available"],
                "groq_mean_score": judges["groq"]["mean_score"],
                "groq_excluded_reason": judges["groq"]["excluded_reason"],
                "groq_fidelity_threshold": fidelity_thresholds.get("groq"),
                "mean_attribute_fidelity": consensus_fidelity,
                "n_contributing_judges": n_contributing,
                "fidelity_pass": fidelity_pass,
                "overall_pass": bool(row["copy_check_pass"]) and fidelity_pass,
            }
        )

    rescored = pl.DataFrame(rows).select(df.columns)
    rescored.write_csv(path)
    return rescored


if __name__ == "__main__":
    main()
