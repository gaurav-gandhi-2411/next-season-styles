"""Generate + select the 3 final winning-style concept images (task C6).

Consumes the 3 design briefs C5 wrote (`reports/tables/design_briefs.json`), generates 4
`local_sdxl` candidates per style (one per seed in `SEEDS`), scores every candidate's margin in
both CLIP and DINOv2 embedding space (`nss.generate.margin_scoring.margin`, against that style's
own real references + the shared control pool), and selects the best candidate per style.

OPERATING POINT: `ip_adapter_scale=0.2` -- the only evidence-based in-band candidate from C3's
scale sweep (`nss.generate.scale_sweep`). C3 found no scale tested landed in-band for BOTH CLIP
and DINOv2 simultaneously: CLIP's only in-band scale was 0.2 (at the edge of its upper bound),
DINOv2 never entered its band at any tested scale (0.2-0.9). This module does NOT try to work
around that finding by hunting for a different scale that was never actually validated -- it uses
the one scale the sweep evidence supports, and reports (never suppresses) a DINOv2 band miss if
one occurs. See `select_best_candidate`'s docstring for exactly how CLIP/DINOv2 disagreement is
handled at selection time.

SELECTION RULE (documented, not left implicit): CLIP in-band status is PRIMARY, DINOv2 margin is
a reported SECONDARY signal only (per C3's finding that DINOv2 may never be in-band -- a strict
DINOv2 requirement would then block every candidate, which is not a workable selection rule given
that finding). Candidates are ranked by distance-to-CLIP-band (0 if inside the band), then by
higher CLIP margin, then by seed for a fully deterministic tie-break. C7 (a later task, the VLM
attribute panel) may still re-rank or reject a selection made here via its own QC gate -- this
module does not attempt to anticipate that, per the task's explicit instruction not to block C6 on
a not-yet-run later task.

UNDERWEAR HARD REQUIREMENT: for the underwear style (`UNDERWEAR_STYLE_KEY`), the design brief's
prompt/negative_prompt are strengthened (`strengthen_underwear_prompts`) with explicit
no-human-model exclusions before generation -- flat-lay/mannequin product-catalogue framing only,
never a worn/editorial/human shot. This is enforced two ways: (1) via SDXL's own `negative_prompt`
conditioning (see `nss.generate.backends.generate_concept`'s `negative_prompt` parameter -- newly
added by this task, since the design briefs' `negative_prompt` field was previously computed but
never actually wired into generation), and (2) via a mandatory MANUAL visual check the task
requires a human (here, the operating agent) to perform on the selected candidate before accepting
it -- automated CLIP/DINOv2 margin scoring cannot verify "does this image show a person," so it is
not treated as a substitute for actually looking at the image.

Two-phase VRAM-safe process (same hard constraint as `nss.generate.scale_sweep`, reused verbatim):
generate ALL local_sdxl candidates first, explicitly free the cached SDXL pipeline
(`nss.generate.scale_sweep.free_sdxl_pipeline`), THEN import/run the CPU-only CLIP/DINOv2 scoring
modules -- never in the same process turn as an SDXL generation call, on this 8 GB VRAM machine.

Usage:
    uv run python -m nss.generate.final_concepts
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from nss.generate import backends
from nss.generate.derive_margin_band import CONTROL_MANIFEST_PATH, load_control_pool
from nss.generate.derive_similarity_band import group_by_style
from nss.generate.scale_sweep import (
    CLIP_BAND_PATH,
    DINO_BAND_PATH,
    free_sdxl_pipeline,
    load_margin_band,
)

DESIGN_BRIEFS_PATH = Path("reports/tables/design_briefs.json")
FINAL_THREE_MANIFEST_PATH = Path("reports/tables/exemplar_images_final_three.csv")

OUTPUT_DIR = Path("data/generated/final_concepts")
GEMINI_OUTPUT_DIR = Path("data/generated/final_concepts_gemini")
OUTPUT_TABLE_PATH = Path("reports/tables/final_concepts.csv")

IP_ADAPTER_SCALE = 0.2  # the only evidence-based candidate from C3 -- see module docstring.
SEEDS = (42, 43, 44, 45)

UNDERWEAR_STYLE_KEY = "Ladieswear || Underwear bottom || Under-, Nightwear || Red || Solid"

# Explicit, thorough exclusion terms for the underwear style -- the design brief's own
# negative_prompt already has "worn by a human model, face, skin" but not bare "person"/"model"/
# "body"/"nude" terms that could still let a human figure into frame without technically being
# "worn by a model" (e.g. a person merely holding or posed near the garment). Each term is only
# appended if not already present in the brief's negative_prompt (idempotent).
UNDERWEAR_NEGATIVE_TERMS = (
    "person",
    "human",
    "model",
    "human model",
    "face",
    "skin",
    "body",
    "worn",
    "nude",
    "nudity",
    "lingerie editorial",
    "underwear model",
    "human figure",
    "torso",
    "hips",
    "legs",
)
UNDERWEAR_PROMPT_SUFFIX = (
    " Framing: flat-lay or mannequin product-catalogue framing ONLY -- no human model, no "
    "worn-on-body shot, no person in frame."
)

# MANUAL VISUAL QC FINDING (task C6, recorded after actually inspecting all 4 generated
# candidates -- see the C6 task report): seeds 42/43/44 each show a human model (torso/hips/legs)
# despite the strengthened negative_prompt above; only seed 45 is a clean flat-lay/product shot.
# The underwear style's OWN reference images (`exemplar_images_final_three.csv`, final_rank_2) are
# themselves clean flat-lay product photos with no human model -- confirmed by inspection, not
# assumed -- so the drift is not explained by IP-Adapter image conditioning pulling toward a human
# figure already present in the reference. The likely mechanism: SDXL's base training data for
# "underwear"/"lingerie" prompts is dominated by worn-by-a-model product photography, and at
# ip_adapter_scale=0.2 (a deliberately light conditioning strength) that base-model prior
# apparently outweighs both the negative text prompt and the (correctly flat-lay) reference image
# for 3 of 4 seeds. This is a genuine QC failure worth reporting, not suppressed -- see
# `select_best_candidate`'s `disqualified_seeds` parameter, the mechanism this constant feeds.
UNDERWEAR_VISUAL_QC_DISQUALIFIED_SEEDS: frozenset[int] = frozenset({42, 43, 44})


def load_design_briefs(path: Path = DESIGN_BRIEFS_PATH) -> dict[str, dict[str, Any]]:
    """Load `design_briefs.json`, keyed by `style_id`.

    Args:
        path: Path to `reports/tables/design_briefs.json` (C5's output).

    Returns:
        Mapping of `style_id -> brief dict` (carries `rendered_prompt`, `negative_prompt`, etc.).
    """
    briefs = json.loads(path.read_text(encoding="utf-8"))
    return {brief["style_id"]: brief for brief in briefs}


def load_final_three_references(
    manifest_path: Path = FINAL_THREE_MANIFEST_PATH,
) -> dict[str, list[Path]]:
    """Load the 3 winning styles' on-disk reference image paths, grouped by `style_key`.

    Only `role` starting with `final_rank_` and `fetch_success == True` rows with an existing file
    on disk are included (reuses `nss.generate.derive_similarity_band.group_by_style`'s existence
    check).

    Args:
        manifest_path: Path to `exemplar_images_final_three.csv`.

    Returns:
        Mapping of `style_key -> sorted list of image Path`s, for the 3 `final_rank_*` styles.
    """
    manifest = pl.read_csv(manifest_path)
    filtered = manifest.filter(
        pl.col("role").str.starts_with("final_rank_") & pl.col("fetch_success")
    )
    return group_by_style(filtered)


def strengthen_underwear_prompts(rendered_prompt: str, negative_prompt: str) -> tuple[str, str]:
    """Hard-enforce no-human-model, flat-lay/mannequin-only framing for the underwear style.

    Idempotent: safe to call more than once on the same strings -- only missing terms are
    appended, so re-running this on an already-strengthened pair of strings is a no-op.

    Args:
        rendered_prompt: The design brief's `rendered_prompt`.
        negative_prompt: The design brief's `negative_prompt`.

    Returns:
        `(strengthened_prompt, strengthened_negative_prompt)`.
    """
    prompt = rendered_prompt
    if "flat-lay or mannequin product-catalogue framing only" not in prompt.lower():
        prompt = prompt.rstrip() + UNDERWEAR_PROMPT_SUFFIX

    negative_lower = negative_prompt.lower()
    missing_terms = [t for t in UNDERWEAR_NEGATIVE_TERMS if t not in negative_lower]
    if missing_terms:
        negative_prompt = negative_prompt.rstrip().rstrip(",") + ", " + ", ".join(missing_terms)

    return prompt, negative_prompt


def build_generation_spec(style_id: str, brief: dict[str, Any]) -> tuple[str, str]:
    """Build the (prompt, negative_prompt) pair actually used for generation for one style.

    The underwear style gets `strengthen_underwear_prompts` applied (see module docstring HARD
    REQUIREMENT); every other style uses its design brief's `rendered_prompt`/`negative_prompt`
    verbatim.

    Args:
        style_id: The style's `design_briefs.json` `style_id`.
        brief: That style's brief dict (must have `rendered_prompt`, `negative_prompt`).

    Returns:
        `(prompt, negative_prompt)` to pass to `generate_concept`.
    """
    prompt, negative_prompt = brief["rendered_prompt"], brief["negative_prompt"]
    if style_id == UNDERWEAR_STYLE_KEY:
        prompt, negative_prompt = strengthen_underwear_prompts(prompt, negative_prompt)
    return prompt, negative_prompt


def gemini_prompt_text(style_id: str, prompt: str, negative_prompt: str) -> str:
    """Build the text prompt sent to the `gemini` backend (which has no `negative_prompt` slot).

    For the underwear style, the hard no-human-model constraint is folded directly into the
    prompt text as an explicit instruction, since Gemini's image API exposes no negative-prompt
    equivalent (`generate_concept` raises if a non-None `negative_prompt` is passed with
    `backend="gemini"`). For the other two styles, the (already backend-agnostic) prompt is used
    verbatim.

    Args:
        style_id: The style's `design_briefs.json` `style_id`.
        prompt: The (possibly underwear-strengthened) `rendered_prompt`.
        negative_prompt: The (possibly underwear-strengthened) `negative_prompt`; folded into the
            returned text only for the underwear style.

    Returns:
        The text prompt to pass to `generate_concept(backend="gemini", ...)`.
    """
    if style_id != UNDERWEAR_STYLE_KEY:
        return prompt
    return (
        f"{prompt} STRICT REQUIREMENT: flat-lay or mannequin product-catalogue framing ONLY. "
        f"Do NOT include a human model, person, face, skin, or body of any kind. Avoid: "
        f"{negative_prompt}."
    )


def _slugify(style_id: str) -> str:
    """Filesystem-safe (Windows-safe: no `,`/`:`) slug for a `" || "`-separated `style_id`."""
    return (
        style_id.lower().replace(" || ", "_").replace(", ", "-").replace(",", "-").replace(" ", "-")
    )


@dataclass(frozen=True)
class Candidate:
    """One generated `local_sdxl` candidate: its style, seed, and saved path."""

    style_id: str
    seed: int
    image_path: Path


def generate_candidates_for_style(
    style_id: str,
    prompt: str,
    negative_prompt: str,
    reference_images: list[Path],
    seeds: tuple[int, ...] = SEEDS,
    ip_adapter_scale: float = IP_ADAPTER_SCALE,
    output_dir: Path = OUTPUT_DIR,
) -> list[Candidate]:
    """Generate one `local_sdxl` candidate per seed for one style.

    Each generated image is copied to a style+seed-labeled path under `output_dir` immediately
    after its `generate_concept` call returns (the same seed is reused across the 3 styles, and
    `generate_concept` itself saves under a seed-only filename -- see
    `nss.generate.scale_sweep.run_generation_phase`'s identical convention/rationale).

    Args:
        style_id: The style's `design_briefs.json` `style_id`.
        prompt: Text prompt (see `build_generation_spec`).
        negative_prompt: Negative prompt (see `build_generation_spec`).
        reference_images: IP-Adapter reference images for this style (only the first is used per
            call -- see `nss.generate.backends` module docstring note 1).
        seeds: Seeds to generate one candidate per.
        ip_adapter_scale: IP-Adapter conditioning strength.
        output_dir: Directory to copy each labeled candidate image into.

    Returns:
        One `Candidate` per seed, in `seeds` order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify(style_id)
    candidates: list[Candidate] = []
    for seed in seeds:
        print(f"[generate] style={style_id!r} seed={seed} ...")
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
        dest_path = output_dir / f"{slug}_seed{seed}.png"
        shutil.copy2(source_path, dest_path)
        source_meta = source_path.with_suffix(".json")
        if source_meta.exists():
            shutil.copy2(source_meta, dest_path.with_suffix(".json"))
        candidates.append(Candidate(style_id=style_id, seed=seed, image_path=dest_path))
        print(f"  -> {dest_path}")
    return candidates


def score_candidates(
    candidates: list[Candidate],
    style_references: list[Path],
    control_images: list[Path],
) -> list[dict[str, Any]]:
    """Score each candidate's CLIP + DINOv2 margin against its style's references + control pool.

    Must be called only after `free_sdxl_pipeline()` -- see module docstring VRAM-safety note.
    Imports the CPU-only scoring modules lazily, at call time, for the same reason.

    Args:
        candidates: Output of `generate_candidates_for_style` (all for the SAME style_id).
        style_references: Real reference images for the candidates' target style.
        control_images: Real images for the shared control pool.

    Returns:
        One dict per candidate: `style_id`, `seed`, `image_path`, `clip_margin`, `dino_margin`.
    """
    from nss.generate import clip_scoring, dino_scoring
    from nss.generate.margin_scoring import margin

    clip_refs = [clip_scoring.embed_image(p) for p in style_references]
    clip_control = [clip_scoring.embed_image(p) for p in control_images]
    dino_refs = [dino_scoring.embed_image(p) for p in style_references]
    dino_control = [dino_scoring.embed_image(p) for p in control_images]

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        print(f"[score] style={candidate.style_id!r} seed={candidate.seed} ...")
        clip_concept = clip_scoring.embed_image(candidate.image_path)
        dino_concept = dino_scoring.embed_image(candidate.image_path)
        rows.append(
            {
                "style_id": candidate.style_id,
                "seed": candidate.seed,
                "image_path": str(candidate.image_path),
                "clip_margin": margin(clip_concept, clip_refs, clip_control),
                "dino_margin": margin(dino_concept, dino_refs, dino_control),
            }
        )
    return rows


def _distance_to_band(value: float, lower: float, upper: float) -> float:
    """Distance from `value` to the nearer edge of `[lower, upper]`; 0.0 if already inside.

    Args:
        value: The value to check.
        lower: Band lower bound.
        upper: Band upper bound.

    Returns:
        `0.0` if `lower <= value <= upper`, else the (positive) distance to the nearer bound.
    """
    if value < lower:
        return lower - value
    if value > upper:
        return value - upper
    return 0.0


def select_best_candidate(
    scored_candidates: list[dict[str, Any]],
    clip_band: tuple[float, float],
    dino_band: tuple[float, float],
    disqualified_seeds: frozenset[int] = frozenset(),
) -> dict[str, Any]:
    """Select the best of one style's scored candidates: CLIP in-band is PRIMARY.

    Ranking key: `(visual_qc_disqualified, clip_distance_to_band, -clip_margin, seed)` ascending
    -- i.e. a `disqualified_seeds` veto always outranks margin score (see `disqualified_seeds`
    below); among non-disqualified candidates, prefer one whose CLIP margin is inside the CLIP
    band (distance 0) over one that isn't; among equal distances, prefer the higher CLIP margin
    (closer to the band's center from below, or further past a near-miss from above); ties broken
    by the lower seed for determinism. DINOv2's band membership is computed and reported on every
    candidate but does NOT enter the ranking key -- see module docstring SELECTION RULE for why
    (C3's finding that DINOv2 may never be in-band for any tested scale).

    Args:
        scored_candidates: Output of `score_candidates`, all for the SAME style_id (non-empty).
        clip_band: `(lower, upper)` CLIP margin band.
        dino_band: `(lower, upper)` DINOv2 margin band.
        disqualified_seeds: Seeds VETOED by a manual visual QC check (e.g. a candidate showing a
            human model despite the underwear style's hard no-human-model requirement --  see
            module docstring UNDERWEAR HARD REQUIREMENT). Automated CLIP/DINOv2 margin scoring
            cannot detect this on its own -- a human (here, the operating agent) must inspect the
            candidate images and pass disqualifying seeds in explicitly; this is never inferred
            from the margin scores. Disqualified candidates are still margin-ranked and appear in
            `discarded` (tagged `visual_qc_disqualified=True`) for transparency, but can never be
            `selected`.

    Returns:
        `{"selected": dict, "discarded": list[dict]}` -- each candidate dict is enriched with
        `clip_in_band`, `dino_in_band`, `clip_distance_to_band`, `visual_qc_disqualified`.
        `discarded` is sorted the same way, worst-ranked last.

    Raises:
        ValueError: if `scored_candidates` is empty, or if every candidate is disqualified (no
            eligible selection remains).
    """
    if not scored_candidates:
        raise ValueError("scored_candidates must be non-empty")

    clip_lower, clip_upper = clip_band
    dino_lower, dino_upper = dino_band
    enriched = [
        {
            **candidate,
            "clip_in_band": clip_lower <= candidate["clip_margin"] <= clip_upper,
            "dino_in_band": dino_lower <= candidate["dino_margin"] <= dino_upper,
            "clip_distance_to_band": _distance_to_band(
                candidate["clip_margin"], clip_lower, clip_upper
            ),
            "visual_qc_disqualified": candidate["seed"] in disqualified_seeds,
        }
        for candidate in scored_candidates
    ]
    ranked = sorted(
        enriched,
        key=lambda c: (
            c["visual_qc_disqualified"],
            c["clip_distance_to_band"],
            -c["clip_margin"],
            c["seed"],
        ),
    )
    eligible = [c for c in ranked if not c["visual_qc_disqualified"]]
    if not eligible:
        raise ValueError(
            "every candidate is visual_qc_disqualified -- no eligible selection remains "
            "(regenerate with a stronger negative prompt / different seeds)"
        )
    selected = eligible[0]
    discarded = [c for c in ranked if c is not selected]
    return {"selected": selected, "discarded": discarded}


def generate_gemini_candidate(
    style_id: str,
    prompt_text: str,
    reference_images: list[Path],
    seed: int = SEEDS[0],
    output_dir: Path = GEMINI_OUTPUT_DIR,
) -> Path | None:
    """Attempt one `gemini`-backend candidate for `style_id`; returns `None` if blocked.

    Task C6's Gemini backend-parity appendix must not block the `local_sdxl` primary deliverable.
    This wraps `generate_concept(backend="gemini", ...)` and converts two documented, expected
    failure modes into a `None` return + printed report line instead of crashing the whole C6
    pipeline: (1) a missing `GEMINI_API_KEY` (`RuntimeError`), and (2) a Gemini-side API failure --
    `google.genai.errors.APIError` and subclasses, e.g. `ClientError` 429 `RESOURCE_EXHAUSTED`
    (quota exhausted on the free tier) or a `ServerError` 5xx. Any OTHER exception is NOT swallowed
    -- it still propagates, since only these two known, expected, must-not-block conditions are
    treated as "this specific sub-task is blocked," not a silent catch-all.

    Args:
        style_id: The style's `design_briefs.json` `style_id`.
        prompt_text: Output of `gemini_prompt_text`.
        reference_images: Reference images for this style (Gemini uses all of them, unlike
            `local_sdxl`'s single-image limit -- see `nss.generate.backends` module docstring
            note 2).
        seed: Base seed passed to Gemini (best-effort reproducibility only).
        output_dir: Directory to copy the resulting image into.

    Returns:
        The saved image path, or `None` if blocked on a missing `GEMINI_API_KEY` or a Gemini API
        error (quota/rate-limit/server failure).

    Raises:
        RuntimeError: any Gemini failure that is NOT a missing-`GEMINI_API_KEY` configuration
            issue (re-raised, not swallowed).
    """
    from google.genai import errors as genai_errors

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        paths = backends.generate_concept(
            prompt=prompt_text,
            reference_images=reference_images,
            backend=backends.GEMINI,
            ip_adapter_scale=None,
            seed=seed,
            n=1,
        )
    except RuntimeError as exc:
        if "GEMINI_API_KEY" not in str(exc):
            raise
        print(f"[gemini] BLOCKED for {style_id!r} (missing API key): {exc}")
        return None
    except genai_errors.APIError as exc:
        print(f"[gemini] BLOCKED for {style_id!r} (Gemini API error, e.g. quota/rate-limit): {exc}")
        return None
    source_path = paths[0]
    dest_path = output_dir / f"{_slugify(style_id)}.png"
    shutil.copy2(source_path, dest_path)
    source_meta = source_path.with_suffix(".json")
    if source_meta.exists():
        shutil.copy2(source_meta, dest_path.with_suffix(".json"))
    print(f"[gemini] wrote {dest_path}")
    return dest_path


def main() -> None:
    """Run the full C6 pipeline: generate -> free VRAM -> score -> select -> write -> Gemini."""
    briefs = load_design_briefs()
    style_references = load_final_three_references()
    control_images = load_control_pool(CONTROL_MANIFEST_PATH)
    print(f"Control pool ({len(control_images)}): {[str(p) for p in control_images]}")

    all_candidates: dict[str, list[Candidate]] = {}
    gen_specs: dict[str, tuple[str, str]] = {}
    for style_id, brief in briefs.items():
        prompt, negative_prompt = build_generation_spec(style_id, brief)
        gen_specs[style_id] = (prompt, negative_prompt)
        print(f"\nStyle: {style_id}")
        print(f"Prompt: {prompt!r}")
        print(f"Negative prompt: {negative_prompt!r}")
        references = style_references[style_id]
        print(f"References ({len(references)}): {[str(p) for p in references]}")
        all_candidates[style_id] = generate_candidates_for_style(
            style_id, prompt, negative_prompt, references
        )

    vram_before, vram_after = free_sdxl_pipeline()
    print(f"\nVRAM before free: {vram_before:.3f} GB, after free: {vram_after:.3f} GB")
    if vram_after >= vram_before:
        print("WARNING: VRAM did not measurably drop after freeing the SDXL pipeline.")

    clip_band = load_margin_band(CLIP_BAND_PATH)
    dino_band = load_margin_band(DINO_BAND_PATH)
    print(f"\nCLIP band: {clip_band}")
    print(f"DINOv2 band: {dino_band}")

    rows: list[dict[str, Any]] = []
    for style_id, candidates in all_candidates.items():
        scored = score_candidates(candidates, style_references[style_id], control_images)
        disqualified = (
            UNDERWEAR_VISUAL_QC_DISQUALIFIED_SEEDS
            if style_id == UNDERWEAR_STYLE_KEY
            else frozenset()
        )
        result = select_best_candidate(
            scored, clip_band, dino_band, disqualified_seeds=disqualified
        )
        selected = result["selected"]
        discarded = result["discarded"]
        print(
            f"\n[select] {style_id}: chosen seed={selected['seed']} "
            f"clip_margin={selected['clip_margin']:.4f} (in_band={selected['clip_in_band']}) "
            f"dino_margin={selected['dino_margin']:.4f} (in_band={selected['dino_in_band']})"
        )
        if disqualified:
            print(f"  visual_qc_disqualified_seeds={sorted(disqualified)} (manual override)")
        rows.append(
            {
                "style_id": style_id,
                "chosen_seed": selected["seed"],
                "local_path": selected["image_path"],
                "clip_margin": selected["clip_margin"],
                "dino_margin": selected["dino_margin"],
                "clip_in_band": selected["clip_in_band"],
                "dino_in_band": selected["dino_in_band"],
                "visual_qc_disqualified_seeds": json.dumps(sorted(disqualified)),
                "discarded_candidates_json": json.dumps(
                    [
                        {
                            "seed": d["seed"],
                            "clip_margin": d["clip_margin"],
                            "dino_margin": d["dino_margin"],
                            "clip_in_band": d["clip_in_band"],
                            "dino_in_band": d["dino_in_band"],
                            "visual_qc_disqualified": d["visual_qc_disqualified"],
                        }
                        for d in discarded
                    ]
                ),
            }
        )

    OUTPUT_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(OUTPUT_TABLE_PATH)
    print(f"\nWrote {OUTPUT_TABLE_PATH}")

    print("\n=== Gemini backend parity appendix ===")
    for style_id in briefs:
        prompt, negative_prompt = gen_specs[style_id]
        text = gemini_prompt_text(style_id, prompt, negative_prompt)
        generate_gemini_candidate(style_id, text, style_references[style_id])


if __name__ == "__main__":
    main()
