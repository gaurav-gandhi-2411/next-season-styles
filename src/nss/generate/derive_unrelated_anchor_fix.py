"""Fix E1's contaminated `unrelated_anchor_gen` anchor (task F1).

ROOT CAUSE BEING FIXED (documented here, not just in the task brief, per project convention): E1's
`nss.generate.derive_generated_space_anchors` generated BOTH `copy_anchor_gen` and
`unrelated_anchor_gen` at the SAME `ip_adapter_scale=1.0` (full-strength IP-Adapter conditioning).
E1's own upstream evidence (`nss.generate.scale_sweep`'s C3 sweep) already showed IP-Adapter's
image conditioning dominates the text prompt at high scale -- but E1 applied full-strength
conditioning to `unrelated_anchor_gen` anyway, using a DELIBERATELY-DIFFERENT-GARMENT text prompt to
try to pull the embedding away from the reference. The measured consequence
(`reports/tables/margin_anchor_realspace_vs_genspace_gap.csv`, E1's own output, pooled across all 3
final-three styles): `copy_anchor_gen` and `unrelated_anchor_gen` scored almost identically -- CLIP
0.0960 vs 0.0931 (gap 0.0029), DINOv2 0.5150 vs 0.4899 (gap 0.0251). Gate 1's "not a copy" side had
no real discriminative power under this construction: "genuinely a copy" and "genuinely unrelated"
produced almost the same margin, because at scale=1.0 the reference IMAGE, not the text prompt,
dominates what gets generated regardless of what the prompt asks for.

THE FIX: regenerate ONLY `unrelated_anchor_gen` at `ip_adapter_scale=0.0` -- pure text-to-image,
ZERO IP-Adapter image conditioning. This is the correct construction for "what does this
reference's own unrelated-garment text prompt produce with zero influence from the reference
image" -- a genuinely different-garment generation uncontaminated by IP-Adapter's image-conditioning
pull. `copy_anchor_gen` (E1, `ip_adapter_scale=1.0`) is UNCHANGED by this task and NOT
regenerated -- it is a correct construction as-is (a literal-restatement prompt AT full image
conditioning is exactly "what does a copy look like"; the task F1 brief confirms this explicitly).

REUSES, VERBATIM, FROM `nss.generate.derive_generated_space_anchors` (E1): `SEEDS` (the same 3
seeds, for direct before/after comparability), `build_unrelated_anchor_prompt` +
`UNRELATED_GARMENT_DESCRIPTIONS` (the same unrelated-garment prompts, JUDGMENT CALL already made and
documented in E1, not re-litigated here), `AnchorImage`, `_slugify`, `score_anchor_images`
(embedding-space-agnostic, works for any `AnchorImage` list regardless of `anchor_type`),
`summarize_margins_by_style_and_pooled`, `load_realspace_anchor_summary`, `build_gap_table`,
`REALSPACE_PATHS`/`REALSPACE_ROW_TYPE`, `ALL_STYLES_POOLED_KEY`.

MERGE STRATEGY, NOT A FULL REWRITE: this module reads the EXISTING
`reports/tables/margin_anchors_generated_space.csv` /
`reports/tables/margin_anchor_realspace_vs_genspace_gap.csv`, keeps every `copy_anchor_gen` row
VERBATIM (never recomputed, never re-embedded), and REPLACES only the `unrelated_anchor_gen` rows
with freshly generated-and-scored values -- the smallest change that corrects the contaminated
anchor without touching the anchor task F1's own brief confirms is correct.

METRIC-DROPPING DECISION (task F1's explicit instruction): if a metric's CORRECTED pooled
copy-vs-unrelated gap is still under `NON_DISCRIMINATIVE_GAP_THRESHOLD` (0.05), that metric remains
non-discriminative in generated space even after the fix and must be DROPPED from Gate 1 entirely
(`is_discriminative`, evaluated in `main()` against the actual corrected numbers -- see the task F1
report for which metric(s), if any, this run drops, and
`nss.generate.concept_qc_pipeline.GATE1_ACTIVE_METRICS` for where the decision is wired into the
actual gate).

Two-phase VRAM-safe process, reused verbatim from E1 (8 GB VRAM machine, SDXL alone peaks at
~6.5 GB): generate all 9 `local_sdxl` images first, explicitly free the cached pipeline
(`nss.generate.scale_sweep.free_sdxl_pipeline`), THEN import/run the CPU-only CLIP/DINOv2 scoring
modules.

Usage:
    uv run python -m nss.generate.derive_unrelated_anchor_fix
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import polars as pl

from nss.generate import backends
from nss.generate.derive_generated_space_anchors import (
    ALL_STYLES_POOLED_KEY,
    OUT_ANCHORS_PATH,
    OUT_GAP_PATH,
    REALSPACE_PATHS,
    SEEDS,
    UNRELATED_ANCHOR,
    AnchorImage,
    _slugify,
    build_gap_table,
    build_unrelated_anchor_prompt,
    load_realspace_anchor_summary,
    score_anchor_images,
    summarize_margins_by_style_and_pooled,
)
from nss.generate.derive_margin_band import CONTROL_MANIFEST_PATH, load_control_pool
from nss.generate.final_concepts import FINAL_THREE_MANIFEST_PATH, load_final_three_references
from nss.generate.scale_sweep import free_sdxl_pipeline

# Pure text-to-image, zero IP-Adapter image conditioning -- see module docstring THE FIX.
CORRECTED_IP_ADAPTER_SCALE = 0.0

OUTPUT_DIR = Path("data/generated/anchors_generated_space_fixed")

# See module docstring METRIC-DROPPING DECISION -- both E1's contaminated gaps (CLIP 0.0029,
# DINOv2 0.0251) were far below this bar, which is why the contamination was visible at all; 0.05
# is an order-of-magnitude-above-noise cutoff chosen BEFORE looking at this run's corrected
# numbers, not fit to them.
NON_DISCRIMINATIVE_GAP_THRESHOLD = 0.05


def is_discriminative(gap: float, threshold: float = NON_DISCRIMINATIVE_GAP_THRESHOLD) -> bool:
    """Whether a copy-vs-unrelated pooled margin gap carries enough signal to gate on.

    Args:
        gap: `copy_anchor_gen mean - unrelated_anchor_gen mean`, pooled across the final-three
            styles, in ONE embedding space (`compute_gap`'s convention, but evaluated here on the
            CORRECTED `unrelated_anchor_gen`, not E1's contaminated one).
        threshold: Minimum absolute gap to trust the metric as discriminative (see module
            docstring METRIC-DROPPING DECISION for why `0.05` was chosen).

    Returns:
        `True` iff `abs(gap) >= threshold`.
    """
    return abs(gap) >= threshold


def generate_unrelated_anchor_images(
    style_to_references: dict[str, list[Path]],
    seeds: tuple[int, ...] = SEEDS,
    ip_adapter_scale: float = CORRECTED_IP_ADAPTER_SCALE,
    output_dir: Path = OUTPUT_DIR,
) -> list[AnchorImage]:
    """Generate `unrelated_anchor_gen` candidates ONLY, at the corrected scale, for every style x
    seed.

    Mirrors `nss.generate.derive_generated_space_anchors.generate_anchor_images`, restricted to
    the single anchor type this task fixes -- see that function's docstring for the shared
    per-call conventions (labeled output path, metadata sidecar copy).

    Args:
        style_to_references: `style_key -> that style's real reference image Path`s.
        seeds: Seeds to generate one candidate per, for each style (same 3 seeds as E1, for direct
            before/after comparability).
        ip_adapter_scale: IP-Adapter conditioning strength (`CORRECTED_IP_ADAPTER_SCALE`, `0.0`, by
            default -- see module docstring THE FIX).
        output_dir: Directory to copy each labeled candidate image into.

    Returns:
        One `AnchorImage` per `(style_key, seed)` pair, `style_to_references`-major, `seeds`-minor
        order. `anchor_type` is always `UNRELATED_ANCHOR`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[AnchorImage] = []
    for style_key, references in style_to_references.items():
        slug = _slugify(style_key)
        prompt = build_unrelated_anchor_prompt(style_key)
        for seed in seeds:
            print(
                f"[generate] style={style_key!r} anchor={UNRELATED_ANCHOR} "
                f"ip_adapter_scale={ip_adapter_scale} seed={seed} prompt={prompt!r} ..."
            )
            paths = backends.generate_concept(
                prompt=prompt,
                reference_images=references,
                backend=backends.LOCAL_SDXL,
                ip_adapter_scale=ip_adapter_scale,
                seed=seed,
                n=1,
            )
            source_path = paths[0]
            dest_path = output_dir / f"{slug}_{UNRELATED_ANCHOR}_seed{seed}.png"
            shutil.copy2(source_path, dest_path)
            source_meta = source_path.with_suffix(".json")
            if source_meta.exists():
                shutil.copy2(source_meta, dest_path.with_suffix(".json"))
            results.append(
                AnchorImage(
                    style_key=style_key,
                    anchor_type=UNRELATED_ANCHOR,
                    seed=seed,
                    image_path=dest_path,
                )
            )
            print(f"  -> {dest_path}")
    return results


def merge_corrected_unrelated_rows(
    corrected_summaries: list[dict[str, Any]], existing_anchors_path: Path = OUT_ANCHORS_PATH
) -> list[dict[str, Any]]:
    """Keep every already-written `copy_anchor_gen` row verbatim; replace `unrelated_anchor_gen`.

    Args:
        corrected_summaries: Output of `summarize_margins_by_style_and_pooled` for the freshly
            generated-and-scored `unrelated_anchor_gen` images (this task).
        existing_anchors_path: Path to the already-written `margin_anchors_generated_space.csv`
            (E1's output) -- read, never overwritten in place until `main()` explicitly writes the
            merged result back.

    Returns:
        `copy_anchor_gen` rows from `existing_anchors_path`, verbatim, PLUS `corrected_summaries`
        -- sorted by `(embedding_space, anchor_type, style_key)` for determinism.
    """
    existing = pl.read_csv(existing_anchors_path)
    copy_rows = existing.filter(pl.col("anchor_type") != UNRELATED_ANCHOR).to_dicts()
    merged = copy_rows + corrected_summaries
    return sorted(merged, key=lambda r: (r["embedding_space"], r["anchor_type"], r["style_key"]))


def main() -> None:
    """Run the full F1 pipeline: generate (corrected scale) -> free VRAM -> score -> merge -> gap
    -> write -> metric-dropping decision."""
    wall_start = time.perf_counter()

    style_to_references = load_final_three_references(FINAL_THREE_MANIFEST_PATH)
    control_images = load_control_pool(CONTROL_MANIFEST_PATH)
    print(f"Styles ({len(style_to_references)}): {list(style_to_references)}")
    print(f"Control pool ({len(control_images)}): {[str(p) for p in control_images]}")
    print(f"Corrected ip_adapter_scale for unrelated_anchor_gen: {CORRECTED_IP_ADAPTER_SCALE}")

    anchor_images = generate_unrelated_anchor_images(style_to_references)
    print(
        f"\nGenerated {len(anchor_images)} unrelated_anchor_gen images "
        f"({len(style_to_references)} styles x {len(SEEDS)} seeds)."
    )

    vram_before, vram_after = free_sdxl_pipeline()
    print(f"VRAM before free: {vram_before:.3f} GB, after free: {vram_after:.3f} GB")
    if vram_after >= vram_before:
        print("WARNING: VRAM did not measurably drop after freeing the SDXL pipeline.")

    scored_rows = score_anchor_images(anchor_images, style_to_references, control_images)
    wall_seconds = time.perf_counter() - wall_start

    corrected_summaries = summarize_margins_by_style_and_pooled(scored_rows)
    merged_rows = merge_corrected_unrelated_rows(corrected_summaries)
    merged_df = pl.DataFrame(merged_rows).select(
        ["style_key", "anchor_type", "embedding_space", "n", "mean", "median", "std"]
    )
    OUT_ANCHORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged_df.write_csv(OUT_ANCHORS_PATH)
    print(
        f"\nWrote {OUT_ANCHORS_PATH} (copy_anchor_gen rows unchanged, unrelated_anchor_gen fixed)"
    )
    for row in merged_df.sort(["embedding_space", "anchor_type", "style_key"]).iter_rows(
        named=True
    ):
        print(
            f"  {row['embedding_space']:6s} {row['anchor_type']:18s} {row['style_key']:70s} "
            f"n={row['n']} mean={row['mean']:.4f} std={row['std']:.4f}"
        )

    realspace_summaries = {
        space: load_realspace_anchor_summary(path) for space, path in REALSPACE_PATHS.items()
    }
    gap_df = build_gap_table(merged_rows, realspace_summaries)
    OUT_GAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    gap_df.write_csv(OUT_GAP_PATH)
    print(f"\nWrote {OUT_GAP_PATH}")

    print("\n=== Corrected copy-vs-unrelated pooled gap (task F1) ===")
    is_pooled = pl.col("style_key") == ALL_STYLES_POOLED_KEY
    pooled_copy = merged_df.filter(is_pooled & (pl.col("anchor_type") == "copy_anchor_gen"))
    pooled_unrelated = merged_df.filter(is_pooled & (pl.col("anchor_type") == UNRELATED_ANCHOR))
    active_metrics: set[str] = set()
    for space in ("clip", "dinov2"):
        copy_mean = pooled_copy.filter(pl.col("embedding_space") == space)["mean"].item()
        unrelated_mean = pooled_unrelated.filter(pl.col("embedding_space") == space)["mean"].item()
        gap = copy_mean - unrelated_mean
        discriminative = is_discriminative(gap)
        if discriminative:
            active_metrics.add(space)
        print(
            f"  {space:6s} copy_mean={copy_mean:.4f} unrelated_mean={unrelated_mean:.4f} "
            f"gap={gap:+.4f} discriminative(>= {NON_DISCRIMINATIVE_GAP_THRESHOLD})={discriminative}"
        )
    print(f"\nGate-1 active metrics after correction: {sorted(active_metrics) or 'NONE'}")
    if not active_metrics:
        print(
            "WARNING: neither metric is discriminative even after correction -- Gate 1's "
            "copy-check would have no signal to gate on. See the task F1 report."
        )

    print(f"\nTotal wall-clock (generation + scoring): {wall_seconds:.1f}s")


if __name__ == "__main__":
    main()
