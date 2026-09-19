"""H&M-specific `concept-qc` skill pipeline (task C7): QC gate + retry loop over C6's 3 selected
final concepts.

DATASET-SPECIFIC ADAPTER CODE LIVES HERE, NOT IN THE SKILL: `skills/concept-qc/run_qc.py`
deliberately knows nothing about H&M `style_id`s, `final_concepts.csv`, or `ip_adapter_scale` --
this module is exactly the "calling code" that skill's docstring describes. It (1) maps an H&M
`style_id` onto the skill's generic attribute-checklist ground truth, (2) wires the skill's
`qc_verdict`/`run_judge`/`choose_next_retry_value` primitives to this project's real margin scoring
(`nss.generate.clip_scoring`/`dino_scoring`/`margin_scoring`) and real judges
(`nss.generate.vlm_judges`), (3) implements the max-2-retry loop by reusing C6's own generation
primitive (`nss.generate.backends.generate_concept`, via `nss.generate.final_concepts`'s prompt-
building helpers) with an adjusted `ip_adapter_scale`, and (4) writes the two output artifacts.

RETRY-VALUE RATIONALE FOR THIS PROJECT (the concrete inputs to the skill's generic
`choose_next_retry_value`): C3's scale sweep (`nss.generate.scale_sweep`, T-shirt style only,
`SCALES = (0.2, ..., 0.9)`) found CLIP margin rises roughly monotonically with `ip_adapter_scale`
(0.2 -> 0.097, up to 0.9 -> 0.144) while DINOv2 margin is U-shaped, MINIMIZED at the highest tested
scale (0.9 -> 0.688, vs. 0.5's peak of 0.739) -- still nowhere near DINOv2's band regardless. This
pipeline therefore treats CLIP as the primary metric (`primary_*` in `choose_next_retry_value`,
matching C6's own CLIP-primary selection-time ranking convention) and DINOv2 as secondary with
`secondary_favors_higher=True`. These trends were measured ONLY on the T-shirt style -- applying
them to the underwear/sweater styles is a documented EXTRAPOLATION, not re-verified evidence; this
pipeline still follows rule 101c (never conclude "can't be done" without trying) and attempts every
retry for direct per-style evidence regardless of whether the T-shirt trend predicts success.

UNDERWEAR VISUAL QC: C6 found 3 of 4 original underwear candidates showed a human model despite a
strengthened negative prompt (`nss.generate.final_concepts.UNDERWEAR_VISUAL_QC_DISQUALIFIED_SEEDS`)
-- automated CLIP/DINOv2/VLM-attribute scoring cannot detect "does this image show a person." Any
NEW underwear image this pipeline generates (a retry) carries the same risk and REQUIRES the same
manual visual inspection C6's task report documents; this module cannot automate that check and
does not claim to -- see the C7 task report for the manual finding on any retry image actually
generated for the underwear style.

Usage:
    uv run python -m nss.generate.concept_qc_pipeline
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import polars as pl

from nss.generate import backends, final_concepts, vlm_judges
from nss.generate.derive_margin_band import CONTROL_MANIFEST_PATH, load_control_pool
from nss.generate.scale_sweep import (
    CLIP_BAND_PATH,
    DINO_BAND_PATH,
    SCALES,
    free_sdxl_pipeline,
    load_margin_band,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SKILL_MODULE_PATH = _REPO_ROOT / "skills" / "concept-qc" / "run_qc.py"


def _load_skill_module(module_path: Path = _SKILL_MODULE_PATH) -> ModuleType:
    """Load `skills/concept-qc/run_qc.py`. See `nss.generate.vlm_judges._load_skill_module`."""
    if not module_path.exists():
        raise FileNotFoundError(f"concept-qc skill module not found at {module_path}")
    spec = importlib.util.spec_from_file_location("concept_qc_run_qc", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build an import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Reuse the SAME loaded instance `vlm_judges` already loaded -- see that module's `SKILL` docstring
# for why loading a second, independent copy here would be a subtle correctness hazard (isinstance
# checks against a distinct-but-identical-looking exception class).
SKILL = vlm_judges.SKILL

FINAL_CONCEPTS_PATH = Path("reports/tables/final_concepts.csv")
QC_RESULTS_PATH = Path("reports/tables/concept_qc_results.csv")
CALIBRATION_RESULTS_PATH = Path("reports/tables/vlm_calibration_results.csv")
RETRY_OUTPUT_DIR = Path("data/generated/concept_qc_retries")

MAX_RETRIES = 2
ATTRIBUTE_FIDELITY_THRESHOLD = SKILL.DEFAULT_ATTRIBUTE_FIDELITY_THRESHOLD  # 0.75

# A judge PASSES calibration iff mean(positive control scores) - mean(negative control scores) >=
# this gap -- a deliberately conservative, clear-separation requirement (NOT "any positive gap
# counts," which would pass on noise given only 3 positive + 3 negative controls per judge), fixed
# before looking at the actual calibration run's numbers.
CALIBRATION_PASS_GAP = 0.3

# Positive controls: one REAL catalogue image per target style, checked against its OWN true
# attributes (`reports/tables/exemplar_images_final_three.csv`, first `final_rank_*` image per
# style) -- expected to score HIGH.
POSITIVE_CONTROLS: tuple[tuple[str, Path], ...] = (
    ("Ladieswear || T-shirt || Jersey Basic || Black || Solid", Path("data/images/0554598001.jpg")),
    (
        "Ladieswear || Underwear bottom || Under-, Nightwear || Red || Solid",
        Path("data/images/0803986005.jpg"),
    ),
    ("Ladieswear || Sweater || Knitwear || Beige || Melange", Path("data/images/0863646003.jpg")),
)
# Negative controls: REAL images of the UNRELATED control style (`Menswear || Scarf || Accessories
# || Grey || Melange`, `role == "control"` in `exemplar_images_final_three.csv`), each checked
# against a DIFFERENT target style's attribute checklist (mismatched on purpose) -- expected to
# score LOW, since a scarf shares none of a T-shirt/underwear/sweater's true attributes.
NEGATIVE_CONTROLS: tuple[tuple[str, Path], ...] = (
    ("Ladieswear || T-shirt || Jersey Basic || Black || Solid", Path("data/images/0798620003.jpg")),
    (
        "Ladieswear || Underwear bottom || Under-, Nightwear || Red || Solid",
        Path("data/images/0760878001.jpg"),
    ),
    ("Ladieswear || Sweater || Knitwear || Beige || Melange", Path("data/images/0783732002.jpg")),
)


def parse_style_attributes(style_id: str) -> dict[str, str]:
    """Map an H&M `" || "`-separated `style_id` onto the skill's 4 blind ground-truth dimensions.

    Mirrors `nss.generate.scale_sweep.style_description`'s identical 5-part split.

    Args:
        style_id: `"Department || ProductType || ProductGroup || Colour || Pattern"`.

    Returns:
        `{"product_type": ProductType, "colour_family": Colour, "graphical_treatment": Pattern,
        "garment_group": ProductGroup}` -- keyed to match
        `nss.generate.vlm_judges.ATTRIBUTE_DIMENSIONS`.

    Raises:
        ValueError: if `style_id` does not split into exactly 5 `" || "`-separated parts.
    """
    parts = [p.strip() for p in style_id.split(" || ")]
    if len(parts) != 5:
        raise ValueError(
            f"Expected 5 ' || '-separated parts in style_id, got {len(parts)}: {style_id!r}"
        )
    _department, product_type, garment_group, colour, pattern = parts
    return {
        "product_type": product_type,
        "colour_family": colour,
        "graphical_treatment": pattern,
        "garment_group": garment_group,
    }


def run_judge_panel(
    image_path: Path,
    ground_truth: dict[str, str],
    generation_backend: str,
    groq_available: bool,
    groq_unavailable_detail: str = "",
) -> dict[str, Any]:
    """Run both project judges (Gemini always attempted; Groq only if `groq_available`).

    Args:
        image_path: The image to score.
        ground_truth: Output of `parse_style_attributes` (or the target style's true attributes
            for a calibration control).
        generation_backend: Identifier of whatever generated `image_path` (`"local_sdxl"`,
            `"gemini"`, or `"real_catalogue"` for calibration images -- never a generation backend,
            so the contamination rule can never trigger for real photos).
        groq_available: Result of a ONE-TIME `nss.generate.vlm_judges.check_groq_availability()`
            call (never re-checked per image -- see that function's docstring).
        groq_unavailable_detail: The detail string from that same one-time check, folded into
            Groq's `excluded_reason` when `groq_available` is `False`.

    Returns:
        `{"gemini": JudgeResult, "groq": JudgeResult}`.
    """
    gemini_result = SKILL.run_judge(
        "gemini",
        vlm_judges.extract_attributes_gemini,
        image_path,
        vlm_judges.ATTRIBUTE_DIMENSIONS,
        ground_truth,
        generation_backend,
    )
    groq_caller = vlm_judges.extract_attributes_groq if groq_available else None
    groq_unavailable_reason = (
        None
        if groq_available
        else (
            f"GROQ_API_KEY set, but the requested vision judge model is unreachable for this "
            f"account: {groq_unavailable_detail}"
            if groq_unavailable_detail
            else "GROQ_API_KEY not set"
        )
    )
    groq_result = SKILL.run_judge(
        "groq",
        groq_caller,
        image_path,
        vlm_judges.ATTRIBUTE_DIMENSIONS,
        ground_truth,
        generation_backend,
        unavailable_reason=groq_unavailable_reason,
    )
    return {"gemini": gemini_result, "groq": groq_result}


def score_margins(
    image_path: Path,
    clip_refs: Sequence[Any],
    clip_control: Sequence[Any],
    dino_refs: Sequence[Any],
    dino_control: Sequence[Any],
    clip_band: tuple[float, float],
    dino_band: tuple[float, float],
) -> dict[str, Any]:
    """Score one image's CLIP + DINOv2 margin against already-embedded references/control pool.

    Args:
        image_path: The image to score.
        clip_refs: Pre-computed CLIP embeddings of the target style's real reference images.
        clip_control: Pre-computed CLIP embeddings of the control pool.
        dino_refs: Pre-computed DINOv2 embeddings of the target style's real reference images.
        dino_control: Pre-computed DINOv2 embeddings of the control pool.
        clip_band: `(lower, upper)` CLIP margin band.
        dino_band: `(lower, upper)` DINOv2 margin band.

    Returns:
        A `MarginBandResult`-shaped dict (see `skills/concept-qc/run_qc.py`).
    """
    from nss.generate import clip_scoring, dino_scoring
    from nss.generate.margin_scoring import margin as margin_fn

    clip_margin_value = margin_fn(clip_scoring.embed_image(image_path), clip_refs, clip_control)
    dino_margin_value = margin_fn(dino_scoring.embed_image(image_path), dino_refs, dino_control)
    clip_lower, clip_upper = clip_band
    dino_lower, dino_upper = dino_band
    return {
        "clip_margin": clip_margin_value,
        "clip_in_band": clip_lower <= clip_margin_value <= clip_upper,
        "dino_margin": dino_margin_value,
        "dino_in_band": dino_lower <= dino_margin_value <= dino_upper,
    }


def generate_retry_candidate(
    style_id: str,
    prompt: str,
    negative_prompt: str,
    reference_images: list[Path],
    seed: int,
    ip_adapter_scale: float,
    attempt_number: int,
    output_dir: Path = RETRY_OUTPUT_DIR,
) -> Path:
    """Generate ONE retry candidate via C6's own generation primitive (same seed, new scale).

    Calls `nss.generate.backends.generate_concept` directly (the same primitive
    `nss.generate.final_concepts.generate_candidates_for_style` wraps) rather than that wrapper,
    because the wrapper's fixed `{slug}_seed{seed}.png` filename would silently overwrite an
    earlier retry attempt at the SAME seed but a DIFFERENT scale -- this project's retry loop
    deliberately holds the seed fixed across retries (isolating `ip_adapter_scale` as the only
    changed variable), so every attempt needs a distinct, scale+attempt-labeled filename, the same
    labeling convention `nss.generate.scale_sweep.run_generation_phase` uses for its own sweep.

    Args:
        style_id: The style's `design_briefs.json` `style_id`.
        prompt: Text prompt (already underwear-strengthened if applicable -- see
            `nss.generate.final_concepts.build_generation_spec`).
        negative_prompt: Negative prompt (same).
        reference_images: IP-Adapter reference images for this style.
        seed: The seed to reuse (fixed across retries).
        ip_adapter_scale: The new (adjusted) scale for this retry.
        attempt_number: 1 or 2 (this project's retry numbering; 0 is the original C6 candidate).
        output_dir: Directory to write the labeled retry image into.

    Returns:
        Path to the saved, labeled retry image.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = final_concepts._slugify(style_id)
    paths = backends.generate_concept(
        prompt=prompt,
        reference_images=reference_images,
        backend=backends.LOCAL_SDXL,
        ip_adapter_scale=ip_adapter_scale,
        seed=seed,
        n=1,
        negative_prompt=negative_prompt,
    )
    source_path = paths[0]
    dest_path = (
        output_dir / f"{slug}_seed{seed}_scale{ip_adapter_scale:.2f}_retry{attempt_number}.png"
    )
    shutil.copy2(source_path, dest_path)
    source_meta = source_path.with_suffix(".json")
    if source_meta.exists():
        shutil.copy2(source_meta, dest_path.with_suffix(".json"))
    return dest_path


def run_qc_with_retries(
    style_id: str,
    original_seed: int,
    original_scale: float,
    original_image_path: Path,
    original_margin: dict[str, Any],
    ground_truth: dict[str, str],
    generation_backend: str,
    clip_band: tuple[float, float],
    dino_band: tuple[float, float],
    judge_panel_fn: Callable[[Path], dict[str, Any]],
    margin_fn: Callable[[Path], dict[str, Any]],
    generate_fn: Callable[[float, int], Path],
    all_scales: Sequence[float] = SCALES,
    max_retries: int = MAX_RETRIES,
    attribute_fidelity_threshold: float = ATTRIBUTE_FIDELITY_THRESHOLD,
    secondary_favors_higher: bool = True,
) -> list[dict[str, Any]]:
    """Run the C7 QC gate for one style, retrying up to `max_retries` times on failure.

    Every attempt (the original C6 candidate, attempt 0, plus up to `max_retries` retries) is
    scored and appended to the returned list -- nothing is discarded, per the task's explicit
    requirement that the full retry history is itself the deliverable. `judge_panel_fn`/
    `margin_fn`/`generate_fn` are injected so this function is fully unit-testable with fakes (no
    real API/GPU calls) -- see `tests/test_concept_qc_pipeline.py`.

    Args:
        style_id: Opaque identifier, passed through.
        original_seed: The seed C6 selected (held fixed across every retry).
        original_scale: The `ip_adapter_scale` C6 used for its original selection
            (`nss.generate.final_concepts.IP_ADAPTER_SCALE`).
        original_image_path: Path to C6's originally selected candidate image.
        original_margin: `MarginBandResult`-shaped dict, C6's already-computed margin scores for
            the original candidate (reused verbatim for attempt 0 -- never recomputed).
        ground_truth: Output of `parse_style_attributes`.
        generation_backend: Identifier of the generation backend (`nss.generate.backends.LOCAL_SDXL`
            for every retry in this project).
        clip_band: `(lower, upper)` CLIP margin band.
        dino_band: `(lower, upper)` DINOv2 margin band.
        judge_panel_fn: `image_path -> {"gemini": JudgeResult, "groq": JudgeResult}`.
        margin_fn: `image_path -> MarginBandResult`.
        generate_fn: `(new_scale, attempt_number) -> new_image_path` -- generates one retry
            candidate (only called when a retry is actually triggered).
        all_scales: The full tested `ip_adapter_scale` range (`nss.generate.scale_sweep.SCALES`).
        max_retries: Maximum number of retries (task C7: 2).
        attribute_fidelity_threshold: Minimum consensus mean attribute fidelity to pass.
        secondary_favors_higher: See `skills/concept-qc/run_qc.py`'s
            `choose_next_retry_value` -- this project's own evidence (module docstring
            RETRY-VALUE RATIONALE) says `True`.

    Returns:
        One dict per attempt (0-indexed `attempt_number`), each carrying the full `QCVerdict` plus
        generation metadata (`seed`, `ip_adapter_scale`, `image_path`, `retry_triggered`,
        `next_scale`).
    """
    attempts: list[dict[str, Any]] = []
    scale = original_scale
    image_path = original_image_path
    margin = original_margin
    tried_scales: set[float] = set()

    for attempt_number in range(max_retries + 1):
        tried_scales.add(scale)
        judges = judge_panel_fn(image_path)
        verdict = SKILL.qc_verdict(style_id, margin, judges, attribute_fidelity_threshold)
        is_last = attempt_number == max_retries
        retry_triggered = not verdict["overall_pass"] and not is_last

        next_scale: float | None = None
        if retry_triggered:
            next_scale = SKILL.choose_next_retry_value(
                last_value=scale,
                tried_values=frozenset(tried_scales),
                all_values=all_scales,
                primary_metric_value=margin["clip_margin"],
                primary_lower=clip_band[0],
                primary_upper=clip_band[1],
                secondary_metric_value=margin["dino_margin"],
                secondary_lower=dino_band[0],
                secondary_upper=dino_band[1],
                secondary_favors_higher=secondary_favors_higher,
            )

        attempts.append(
            {
                "style_id": style_id,
                "attempt_number": attempt_number,
                "seed": original_seed,
                "ip_adapter_scale": scale,
                "image_path": str(image_path),
                "clip_margin": margin["clip_margin"],
                "clip_in_band": margin["clip_in_band"],
                "dino_margin": margin["dino_margin"],
                "dino_in_band": margin["dino_in_band"],
                "margin_band_pass": verdict["margin_band_pass"],
                "judges": judges,
                "mean_attribute_fidelity": verdict["consensus_mean_attribute_fidelity"],
                "n_contributing_judges": verdict["n_contributing_judges"],
                "fidelity_pass": verdict["fidelity_pass"],
                "overall_pass": verdict["overall_pass"],
                "label": verdict["label"],
                "retry_triggered": retry_triggered,
                "next_scale": next_scale,
            }
        )

        if not retry_triggered:
            break

        image_path = generate_fn(next_scale, attempt_number + 1)
        margin = margin_fn(image_path)
        scale = next_scale

    return attempts


def run_calibration(
    groq_available: bool,
    groq_unavailable_detail: str = "",
    positive_controls: tuple[tuple[str, Path], ...] = POSITIVE_CONTROLS,
    negative_controls: tuple[tuple[str, Path], ...] = NEGATIVE_CONTROLS,
) -> pl.DataFrame:
    """Score every positive/negative calibration control with both judges.

    Args:
        groq_available: One-time `check_groq_availability()` result.
        groq_unavailable_detail: Detail string from that check.
        positive_controls: `(style_id, real_image_path)` pairs checked against their OWN true
            attributes.
        negative_controls: `(style_id, real_image_path)` pairs checked against a DIFFERENT style's
            attributes (mismatched on purpose).

    Returns:
        One row per `(judge, control)` pair; columns match `vlm_calibration_results.csv`.
    """
    rows: list[dict[str, Any]] = []
    control_groups = (("positive", positive_controls), ("negative", negative_controls))
    for control_type, controls in control_groups:
        for style_id, image_path in controls:
            ground_truth = parse_style_attributes(style_id)
            judges = run_judge_panel(
                image_path, ground_truth, "real_catalogue", groq_available, groq_unavailable_detail
            )
            for judge_name, result in judges.items():
                rows.append(
                    {
                        "judge_name": judge_name,
                        "control_type": control_type,
                        "style_id_checked_against": style_id,
                        "image_path": str(image_path),
                        "available": result["available"],
                        "raw_extraction_json": (
                            json.dumps(result["raw_extraction"])
                            if result["raw_extraction"]
                            else None
                        ),
                        "scores_json": json.dumps(result["scores"]) if result["scores"] else None,
                        "mean_score": result["mean_score"],
                        "excluded_reason": result["excluded_reason"],
                    }
                )
    return pl.DataFrame(rows)


def summarize_calibration(
    df: pl.DataFrame, pass_gap: float = CALIBRATION_PASS_GAP
) -> dict[str, dict[str, Any]]:
    """Per-judge calibration PASS/FAIL: `mean(positive scores) - mean(negative scores) >= pass_gap`.

    Args:
        df: Output of `run_calibration`.
        pass_gap: Minimum required separation (see `CALIBRATION_PASS_GAP`'s module-level docstring).

    Returns:
        `{judge_name: {"available": bool, "passed": bool, "positive_mean": float,
        "negative_mean": float, "gap": float}}` (missing `positive_mean`/`negative_mean`/`gap` and
        `passed=False` when `available=False`, i.e. the judge was never reachable for calibration).
    """
    summary: dict[str, dict[str, Any]] = {}
    for judge_name in df["judge_name"].unique(maintain_order=True).to_list():
        judge_df = df.filter(pl.col("judge_name") == judge_name)
        available_df = judge_df.filter(pl.col("available"))
        if available_df.is_empty():
            summary[judge_name] = {
                "available": False,
                "passed": False,
                "detail": "judge never available during calibration",
            }
            continue
        positive_scores = available_df.filter(pl.col("control_type") == "positive")[
            "mean_score"
        ].to_list()
        negative_scores = available_df.filter(pl.col("control_type") == "negative")[
            "mean_score"
        ].to_list()
        positive_mean = sum(positive_scores) / len(positive_scores) if positive_scores else 0.0
        negative_mean = sum(negative_scores) / len(negative_scores) if negative_scores else 0.0
        gap = positive_mean - negative_mean
        summary[judge_name] = {
            "available": True,
            "positive_mean": positive_mean,
            "negative_mean": negative_mean,
            "gap": gap,
            "passed": gap >= pass_gap,
        }
    return summary


def compute_overall_kappa(
    attempts: list[dict[str, Any]], threshold: float = SKILL.DEFAULT_BINARIZE_THRESHOLD
) -> tuple[float | None, int]:
    """Cohen's kappa between Gemini and Groq's binarized per-attribute calls, across every attempt.

    Args:
        attempts: Concatenated output of `run_qc_with_retries` across every style.
        threshold: Binarization threshold (`skills/concept-qc/run_qc.py`'s
            `binarize_scores`).

    Returns:
        `(kappa, n_pairs)` -- `(None, 0)` if no `(style, attempt, attribute)` triple has BOTH
        judges available (e.g. Groq never available in this run -- no kappa is computable, and this
        function says so via the `None` rather than a misleading `0.0`).
    """
    judge_a_calls: list[int] = []
    judge_b_calls: list[int] = []
    for attempt in attempts:
        gemini = attempt["judges"]["gemini"]
        groq = attempt["judges"]["groq"]
        if not (gemini["available"] and groq["available"]):
            continue
        gemini_binary = SKILL.binarize_scores(gemini["scores"], threshold)
        groq_binary = SKILL.binarize_scores(groq["scores"], threshold)
        for dimension, value in gemini_binary.items():
            if dimension in groq_binary:
                judge_a_calls.append(value)
                judge_b_calls.append(groq_binary[dimension])
    if not judge_a_calls:
        return None, 0
    return SKILL.cohens_kappa(judge_a_calls, judge_b_calls), len(judge_a_calls)


def write_results_csv(attempts: list[dict[str, Any]], path: Path = QC_RESULTS_PATH) -> None:
    """Flatten the full retry history (every style, every attempt) into `concept_qc_results.csv`.

    Args:
        attempts: Concatenated output of `run_qc_with_retries` across every style, with
            `style_final_pass`/`n_attempts_for_style` already added per attempt.
        path: Destination CSV path.
    """
    rows: list[dict[str, Any]] = []
    for a in attempts:
        gemini = a["judges"]["gemini"]
        groq = a["judges"]["groq"]
        rows.append(
            {
                "style_id": a["style_id"],
                "attempt_number": a["attempt_number"],
                "seed": a["seed"],
                "ip_adapter_scale": a["ip_adapter_scale"],
                "image_path": a["image_path"],
                "clip_margin": a["clip_margin"],
                "clip_in_band": a["clip_in_band"],
                "dino_margin": a["dino_margin"],
                "dino_in_band": a["dino_in_band"],
                "margin_band_pass": a["margin_band_pass"],
                "gemini_available": gemini["available"],
                "gemini_mean_score": gemini["mean_score"],
                "gemini_raw_json": (
                    json.dumps(gemini["raw_extraction"]) if gemini["raw_extraction"] else None
                ),
                "gemini_scores_json": json.dumps(gemini["scores"]) if gemini["scores"] else None,
                "gemini_excluded_reason": gemini["excluded_reason"],
                "groq_available": groq["available"],
                "groq_mean_score": groq["mean_score"],
                "groq_raw_json": (
                    json.dumps(groq["raw_extraction"]) if groq["raw_extraction"] else None
                ),
                "groq_scores_json": json.dumps(groq["scores"]) if groq["scores"] else None,
                "groq_excluded_reason": groq["excluded_reason"],
                "mean_attribute_fidelity": a["mean_attribute_fidelity"],
                "n_contributing_judges": a["n_contributing_judges"],
                "fidelity_pass": a["fidelity_pass"],
                "overall_pass": a["overall_pass"],
                "retry_triggered": a["retry_triggered"],
                "next_scale": a["next_scale"],
                "style_final_pass": a.get("style_final_pass"),
                "n_attempts_for_style": a.get("n_attempts_for_style"),
                "llm_consensus_label": a["label"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(path)


def load_selected_candidates(path: Path = FINAL_CONCEPTS_PATH) -> dict[str, dict[str, Any]]:
    """Load C6's `final_concepts.csv`, keyed by `style_id`."""
    df = pl.read_csv(path)
    return {row["style_id"]: row for row in df.iter_rows(named=True)}


def main() -> None:
    """Run the full C7 pipeline: calibration -> per-style QC-with-retries -> write artifacts."""
    briefs = final_concepts.load_design_briefs()
    style_references = final_concepts.load_final_three_references()
    control_images = load_control_pool(CONTROL_MANIFEST_PATH)
    selected = load_selected_candidates()

    clip_band = load_margin_band(CLIP_BAND_PATH)
    dino_band = load_margin_band(DINO_BAND_PATH)
    print(f"CLIP band: {clip_band}")
    print(f"DINOv2 band: {dino_band}")

    groq_available, groq_detail = vlm_judges.check_groq_availability()
    print(f"\nGroq judge availability: {groq_available} ({groq_detail})")

    print("\n=== Calibration ===")
    calibration_df = run_calibration(groq_available, groq_detail)
    CALIBRATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    calibration_df.write_csv(CALIBRATION_RESULTS_PATH)
    print(f"Wrote {CALIBRATION_RESULTS_PATH}")
    calibration_summary = summarize_calibration(calibration_df)
    for judge_name, result in calibration_summary.items():
        print(f"  {judge_name}: {result}")

    print("\n=== QC gate with retries ===")
    all_attempts: list[dict[str, Any]] = []
    style_final_pass: dict[str, bool] = {}
    for style_id, brief in briefs.items():
        row = selected[style_id]
        ground_truth = parse_style_attributes(style_id)
        prompt, negative_prompt = final_concepts.build_generation_spec(style_id, brief)
        references = style_references[style_id]

        from nss.generate import clip_scoring, dino_scoring

        clip_refs = [clip_scoring.embed_image(p) for p in references]
        clip_control = [clip_scoring.embed_image(p) for p in control_images]
        dino_refs = [dino_scoring.embed_image(p) for p in references]
        dino_control = [dino_scoring.embed_image(p) for p in control_images]

        def judge_panel_fn(
            image_path: Path, ground_truth: dict[str, str] = ground_truth
        ) -> dict[str, Any]:
            return run_judge_panel(
                image_path, ground_truth, backends.LOCAL_SDXL, groq_available, groq_detail
            )

        def margin_fn(
            image_path: Path,
            clip_refs: Sequence[Any] = clip_refs,
            clip_control: Sequence[Any] = clip_control,
            dino_refs: Sequence[Any] = dino_refs,
            dino_control: Sequence[Any] = dino_control,
        ) -> dict[str, Any]:
            return score_margins(
                image_path, clip_refs, clip_control, dino_refs, dino_control, clip_band, dino_band
            )

        def generate_fn(
            scale: float,
            attempt_number: int,
            style_id: str = style_id,
            prompt: str = prompt,
            negative_prompt: str = negative_prompt,
            references: list[Path] = references,
            seed: int = int(row["chosen_seed"]),
        ) -> Path:
            new_path = generate_retry_candidate(
                style_id, prompt, negative_prompt, references, seed, scale, attempt_number
            )
            vram_before, vram_after = free_sdxl_pipeline()
            print(
                f"  [retry-gen] {style_id} scale={scale} -> {new_path} "
                f"(VRAM before/after free: {vram_before:.3f}/{vram_after:.3f} GB)"
            )
            return new_path

        original_margin = {
            "clip_margin": float(row["clip_margin"]),
            "clip_in_band": bool(row["clip_in_band"]),
            "dino_margin": float(row["dino_margin"]),
            "dino_in_band": bool(row["dino_in_band"]),
        }

        attempts = run_qc_with_retries(
            style_id=style_id,
            original_seed=int(row["chosen_seed"]),
            original_scale=final_concepts.IP_ADAPTER_SCALE,
            original_image_path=Path(row["local_path"]),
            original_margin=original_margin,
            ground_truth=ground_truth,
            generation_backend=backends.LOCAL_SDXL,
            clip_band=clip_band,
            dino_band=dino_band,
            judge_panel_fn=judge_panel_fn,
            margin_fn=margin_fn,
            generate_fn=generate_fn,
        )
        final_attempt = attempts[-1]
        style_final_pass[style_id] = final_attempt["overall_pass"]
        for attempt in attempts:
            attempt["style_final_pass"] = final_attempt["overall_pass"]
            attempt["n_attempts_for_style"] = len(attempts)
        all_attempts.extend(attempts)
        print(
            f"\n[{style_id}] final overall_pass={final_attempt['overall_pass']} after "
            f"{len(attempts)} attempt(s) (0=original + up to {MAX_RETRIES} retries)"
        )
        for attempt in attempts:
            print(
                f"    attempt={attempt['attempt_number']} scale={attempt['ip_adapter_scale']} "
                f"clip_in_band={attempt['clip_in_band']} dino_in_band={attempt['dino_in_band']} "
                f"mean_fidelity={attempt['mean_attribute_fidelity']:.3f} "
                f"(n_judges={attempt['n_contributing_judges']}) "
                f"overall_pass={attempt['overall_pass']}"
            )

    write_results_csv(all_attempts)
    print(f"\nWrote {QC_RESULTS_PATH}")

    kappa, n_pairs = compute_overall_kappa(all_attempts)
    if kappa is None:
        print("\nCohen's kappa: NOT COMPUTABLE (Groq judge never available -- see above).")
    else:
        print(
            f"\nCohen's kappa (Gemini vs. Groq, binarized, n={n_pairs} attribute x attempt "
            f"pairs): {kappa:.4f}"
        )

    n_pass = sum(1 for passed in style_final_pass.values() if passed)
    print(
        f"\n=== FINAL: {n_pass}/{len(style_final_pass)} concepts passed QC within the "
        f"{MAX_RETRIES}-retry cap ==="
    )
    for style_id, passed in style_final_pass.items():
        print(f"  {'PASS' if passed else 'FAIL'}: {style_id}")


if __name__ == "__main__":
    main()
