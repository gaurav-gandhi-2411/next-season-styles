"""Re-derive the real-space margin anchors in GENERATED-image space, and quantify the real-vs-gen
gap.

ROOT CAUSE BEING FIXED (documented here, per project convention):
`reports/tables/margin_anchors_clip.csv` / `margin_anchors_dinov2.csv`
(`nss.generate.derive_margin_band`) were derived entirely from REAL H&M catalogue images -- the
"copy" anchor (`upper_anchor` row) is a real image's leave-one-out margin against its OWN style's
other real references; the "unrelated" anchor (`lower_anchor` row) is a real image's margin against
a DIFFERENT real style's references. Both anchors were then APPLIED, unchanged, to score GENERATED
images (`nss.generate.scale_sweep`, `nss.generate.final_concepts`) -- a cross-distribution
comparison: SDXL+IP-Adapter output has a different photographic character than H&M's studio catalog
photography (lighting, background texture, rendering artifacts, framing), so a generated image's
margin against real references is not on the same footing as a real image's margin against real
references, even when both are "genuinely the same style." This is the same defect class as the
earlier absolute-cosine bug (an absolute-cosine band derived from catalogue-vs-catalogue pairs,
applied to generated-vs-catalogue pairs), one level up: the margin band fixed the WITHIN-real-space
confound (product photography inflating raw cosine) but never validated that the resulting
real-space anchors transfer to generated images at all.

CONSEQUENCE: all 3 final concepts (`reports/tables/final_concepts.csv`) scored between
75% and 100% of the real-image copy anchor (the `upper_anchor` mean) and were judged "not a
copy" only by an arbitrary, unvalidated 75% cutoff -- see `nss.generate.final_concepts`'s
`UPPER_MARGIN_FRACTION` band. If generated images are systematically lower-margin than real images
even for a genuine, intentional near-copy (i.e. if there is a real real-space-vs-gen-space gap),
that 75-100% reading is miscalibrated: a generated copy might legitimately top out well below 100%
of the real anchor for reasons that have nothing to do with novelty.

THIS MODULE re-derives both anchors DIRECTLY in generated-image space, per final-three style:

1. `copy_anchor_gen` ("what a perfect copy looks like in generated space"): `local_sdxl` +
   IP-Adapter at `ip_adapter_scale=1.0` (full-strength conditioning), prompted with a LITERAL
   restatement of the reference garment (`build_copy_anchor_prompt` -- e.g. "a black solid jersey
   basic t-shirt, product photography, plain background" for the T-shirt style), conditioned on
   that style's own reference images. This is the generated-space analogue of the real-space
   `upper_anchor` (leave-one-out own-style real margin).

2. `unrelated_anchor_gen` ("how much does IP-Adapter's image conditioning alone pull toward the
   reference, even when the text prompt asks for something else"): same reference images, same
   `ip_adapter_scale=1.0`, but a prompt for a DELIBERATELY DIFFERENT garment
   (`build_unrelated_anchor_prompt`, `UNRELATED_GARMENT_DESCRIPTIONS`). This is the generated-space
   analogue of the real-space `lower_anchor` (cross-style real margin) -- except here the
   "cross-style" pull comes from IP-Adapter's own visual conditioning on the SAME reference images
   that would otherwise feed the copy anchor, not from a different real style's images.

3 styles x 2 anchor types x 3 seeds (`SEEDS`) = 18 generated images, scored in both CLIP and DINOv2
embedding space (`nss.generate.margin_scoring.margin`) against that style's own real references and
the SAME control pool used everywhere else in this project
(`nss.generate.derive_margin_band.load_control_pool`).

OUTPUTS:

- `reports/tables/margin_anchors_generated_space.csv`: per (style_key, anchor_type,
  embedding_space) mean/median/std/n across the 3 seeds, plus a pooled-across-all-3-styles row
  (`style_key == ALL_STYLES_POOLED_KEY`) per (anchor_type, embedding_space) for direct comparability
  with the real-space anchors, which were themselves pooled across 23 styles rather than reported
  per-style.
- `reports/tables/margin_anchor_realspace_vs_genspace_gap.csv`: every generated-space summary row
  (style-level and pooled) joined against the matching real-space anchor
  (`copy_anchor_gen -> upper_anchor`, `unrelated_anchor_gen -> lower_anchor`), with
  `gap = genspace_mean - realspace_mean` computed explicitly (`compute_gap`). THIS GAP IS THE
  MEASUREMENT ERROR the real-space anchors introduced when applied to generated images.

Two-phase VRAM-safe process, reused verbatim from `nss.generate.scale_sweep` /
`nss.generate.final_concepts` (8 GB VRAM machine, SDXL alone peaks at ~6.5 GB): generate ALL 18
`local_sdxl` images first, explicitly free the cached pipeline
(`nss.generate.scale_sweep.free_sdxl_pipeline`), THEN import/run the CPU-only CLIP/DINOv2 scoring
modules.

MODELLING IS FROZEN here -- no retraining, no backtest re-runs. This is pure generation +
scoring against the existing, unmodified `margin()` primitive and existing real-space anchor CSVs.

Usage:
    uv run python -m nss.generate.derive_generated_space_anchors
"""

from __future__ import annotations

import shutil
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nss.generate import backends
from nss.generate.derive_margin_band import CONTROL_MANIFEST_PATH, OUT_PATHS, load_control_pool
from nss.generate.derive_margin_band import summarize as _summarize
from nss.generate.final_concepts import FINAL_THREE_MANIFEST_PATH, load_final_three_references
from nss.generate.margin_scoring import margin
from nss.generate.scale_sweep import free_sdxl_pipeline, style_description

COPY_ANCHOR = "copy_anchor_gen"
UNRELATED_ANCHOR = "unrelated_anchor_gen"
ANCHOR_TYPES = (COPY_ANCHOR, UNRELATED_ANCHOR)

# Which real-space `margin_anchors_*.csv` `row_type` each generated-space anchor is compared
# against -- see module docstring points 1-2 for why these are the matching pair.
REALSPACE_ROW_TYPE: dict[str, str] = {
    COPY_ANCHOR: "upper_anchor",
    UNRELATED_ANCHOR: "lower_anchor",
}

SEEDS = (42, 43, 44)
IP_ADAPTER_SCALE = 1.0  # full-strength conditioning -- see module docstring points 1-2.

ALL_STYLES_POOLED_KEY = "ALL_STYLES_POOLED"

OUTPUT_DIR = Path("data/generated/anchors_generated_space")
OUT_ANCHORS_PATH = Path("reports/tables/margin_anchors_generated_space.csv")
OUT_GAP_PATH = Path("reports/tables/margin_anchor_realspace_vs_genspace_gap.csv")

# The real-space anchor CSVs this module compares against -- reused, not
# recomputed.
REALSPACE_PATHS: dict[str, Path] = OUT_PATHS

# UNRELATED-GARMENT PROMPTS (JUDGMENT CALL, documented per project convention):
# each swap is a garment from a clearly different `product_type`/`garment_group` than the style's
# own -- top<->bottom<->footwear, never a same-category variant -- chosen BEFORE generating or
# scoring any image, so the choice cannot be read as reverse-engineered from a favorable result.
UNRELATED_GARMENT_DESCRIPTIONS: dict[str, str] = {
    "Ladieswear || T-shirt || Jersey Basic || Black || Solid": "a pair of denim trousers",
    "Ladieswear || Underwear bottom || Under-, Nightwear || Red || Solid": (
        "a chunky cable-knit sweater"
    ),
    "Ladieswear || Sweater || Knitwear || Beige || Melange": "a pair of leather ankle boots",
}


def build_copy_anchor_prompt(style_key: str) -> str:
    """Literal-restatement prompt for the `copy_anchor_gen` anchor -- "what a copy looks like".

    Reuses `nss.generate.scale_sweep.style_description` (e.g. "black solid jersey basic t-shirt")
    rather than that module's `build_prompt` (which wraps it in "a new ... fashion concept for
    <department>" creative-interpretation framing) -- this anchor is deliberately a direct
    restatement, not a creative reinterpretation.

    Args:
        style_key: `"Department || ProductType || ProductGroup || Colour || Pattern"`.

    Returns:
        e.g. `"a black solid jersey basic t-shirt, product photography, plain background"`.
    """
    return f"a {style_description(style_key)}, product photography, plain background"


def build_unrelated_anchor_prompt(style_key: str) -> str:
    """Deliberately-different-garment prompt for the `unrelated_anchor_gen` anchor.

    Args:
        style_key: `"Department || ProductType || ProductGroup || Colour || Pattern"`. Must be a
            key of `UNRELATED_GARMENT_DESCRIPTIONS`.

    Returns:
        e.g. `"a pair of denim trousers, product photography, plain background"` for the T-shirt
        style.

    Raises:
        KeyError: if `style_key` has no entry in `UNRELATED_GARMENT_DESCRIPTIONS`.
    """
    # UNRELATED_GARMENT_DESCRIPTIONS values already carry their own leading article ("a pair of
    # ...", "a chunky ...") -- no "a " prefix here, unlike build_copy_anchor_prompt above (whose
    # style_description() output has no article of its own). A prior version of this function
    # prepended "a " unconditionally, producing "a a pair of denim trousers, ..."; fixed before
    # any image was scored under the buggy prompt (the 2 SDXL runs generated under the buggy
    # prompt were discarded and regenerated).
    return f"{UNRELATED_GARMENT_DESCRIPTIONS[style_key]}, product photography, plain background"


@dataclass(frozen=True)
class AnchorImage:
    """One generated anchor image: its style, anchor type, seed, and saved path."""

    style_key: str
    anchor_type: str
    seed: int
    image_path: Path


def _slugify(style_key: str) -> str:
    """Filesystem-safe (Windows-safe: no `,`/`:`) slug for a `" || "`-separated `style_key`."""
    return (
        style_key.lower()
        .replace(" || ", "_")
        .replace(", ", "-")
        .replace(",", "-")
        .replace(" ", "-")
    )


def generate_anchor_images(
    style_to_references: dict[str, list[Path]],
    seeds: tuple[int, ...] = SEEDS,
    ip_adapter_scale: float = IP_ADAPTER_SCALE,
    output_dir: Path = OUTPUT_DIR,
) -> list[AnchorImage]:
    """Generate `copy_anchor_gen` + `unrelated_anchor_gen` candidates for every style x seed.

    Args:
        style_to_references: `style_key -> that style's real reference image Path`s (output of
            `nss.generate.final_concepts.load_final_three_references`). Only the first path per
            style is actually used as IP-Adapter conditioning -- see
            `nss.generate.backends` module docstring note 1.
        seeds: Seeds to generate one candidate per, for each (style, anchor_type) pair.
        ip_adapter_scale: IP-Adapter conditioning strength (full-strength, `1.0`, by default).
        output_dir: Directory to copy each labeled candidate image into (`generate_concept` itself
            saves under a seed-only filename that would be overwritten across styles/anchor types
            reusing the same seed).

    Returns:
        One `AnchorImage` per `(style_key, anchor_type, seed)` triple, `style_to_references`-major,
        `ANCHOR_TYPES`-then-`seeds`-minor order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_builders: dict[str, Callable[[str], str]] = {
        COPY_ANCHOR: build_copy_anchor_prompt,
        UNRELATED_ANCHOR: build_unrelated_anchor_prompt,
    }
    results: list[AnchorImage] = []
    for style_key, references in style_to_references.items():
        slug = _slugify(style_key)
        for anchor_type, build_prompt in prompt_builders.items():
            prompt = build_prompt(style_key)
            for seed in seeds:
                print(
                    f"[generate] style={style_key!r} anchor={anchor_type} seed={seed} "
                    f"prompt={prompt!r} ..."
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
                dest_path = output_dir / f"{slug}_{anchor_type}_seed{seed}.png"
                # Copy (not move) immediately -- source_path is fully written to disk by the time
                # generate_concept() returns, so this is safe before a later call overwrites it
                # (the same seed is reused across styles/anchor types).
                shutil.copy2(source_path, dest_path)
                source_meta = source_path.with_suffix(".json")
                if source_meta.exists():
                    shutil.copy2(source_meta, dest_path.with_suffix(".json"))
                results.append(
                    AnchorImage(
                        style_key=style_key,
                        anchor_type=anchor_type,
                        seed=seed,
                        image_path=dest_path,
                    )
                )
                print(f"  -> {dest_path}")
    return results


def score_anchor_images(
    anchor_images: list[AnchorImage],
    style_to_references: dict[str, list[Path]],
    control_images: list[Path],
) -> list[dict[str, Any]]:
    """Score every anchor image's margin against ITS OWN style's real references + control pool.

    Must be called only after `free_sdxl_pipeline()` -- see module docstring VRAM-safety note.
    Imports the CPU-only CLIP/DINOv2 scoring modules lazily, at call time, for the same reason.

    Args:
        anchor_images: Output of `generate_anchor_images`.
        style_to_references: `style_key -> real reference image Path`s (the `margin()`
            "style_reference_embeddings" term -- always that image's OWN style, for both anchor
            types, per module docstring points 1-2).
        control_images: Real images for the shared control pool (the `margin()`
            "control_pool_embeddings" term).

    Returns:
        One dict per (anchor image, embedding space): `style_key`, `anchor_type`, `seed`,
        `embedding_space` (`"clip"`/`"dinov2"`), `margin`, `image_path`.
    """
    from nss.generate import clip_scoring, dino_scoring

    embedders: dict[str, Callable[[Path], np.ndarray]] = {
        "clip": clip_scoring.embed_image,
        "dinov2": dino_scoring.embed_image,
    }
    rows: list[dict[str, Any]] = []
    for space, embed_fn in embedders.items():
        control_embeddings = [embed_fn(p) for p in control_images]
        style_ref_embeddings = {
            style_key: [embed_fn(p) for p in refs]
            for style_key, refs in style_to_references.items()
        }
        for anchor_image in anchor_images:
            print(
                f"[score:{space}] style={anchor_image.style_key!r} "
                f"anchor={anchor_image.anchor_type} seed={anchor_image.seed} ..."
            )
            concept_embedding = embed_fn(anchor_image.image_path)
            m = margin(
                concept_embedding,
                style_ref_embeddings[anchor_image.style_key],
                control_embeddings,
            )
            rows.append(
                {
                    "style_key": anchor_image.style_key,
                    "anchor_type": anchor_image.anchor_type,
                    "seed": anchor_image.seed,
                    "embedding_space": space,
                    "margin": m,
                    "image_path": str(anchor_image.image_path),
                }
            )
    return rows


def summarize_margins_by_style_and_pooled(
    scored_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize scored margins per (style, anchor_type, space), plus a pooled-across-styles row.

    Args:
        scored_rows: Output of `score_anchor_images` -- one dict per (anchor image, embedding
            space), each carrying `style_key`, `anchor_type`, `embedding_space`, `margin`.

    Returns:
        One dict per group: `style_key` (a real style key, or `ALL_STYLES_POOLED_KEY` for the row
        pooling every style's margins for that `(anchor_type, embedding_space)` together),
        `anchor_type`, `embedding_space`, plus `nss.generate.derive_margin_band.summarize`'s
        `n`/`mean`/`median`/`std`. Sorted by `(style_key, anchor_type, embedding_space)`, pooled
        rows last.
    """
    by_style: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    by_pooled: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in scored_rows:
        by_style[(row["style_key"], row["anchor_type"], row["embedding_space"])].append(
            row["margin"]
        )
        by_pooled[(row["anchor_type"], row["embedding_space"])].append(row["margin"])

    results: list[dict[str, Any]] = []
    for (style_key, anchor_type, space), margins in sorted(by_style.items()):
        results.append(
            {
                "style_key": style_key,
                "anchor_type": anchor_type,
                "embedding_space": space,
                **_summarize(margins),
            }
        )
    for (anchor_type, space), margins in sorted(by_pooled.items()):
        results.append(
            {
                "style_key": ALL_STYLES_POOLED_KEY,
                "anchor_type": anchor_type,
                "embedding_space": space,
                **_summarize(margins),
            }
        )
    return results


def load_realspace_anchor_summary(path: Path) -> dict[str, dict[str, float]]:
    """Load one embedding space's real-image anchor summary from a `margin_anchors_*.csv`.

    Args:
        path: Path to `reports/tables/margin_anchors_clip.csv` or `..._dinov2.csv` (output of
            `nss.generate.derive_margin_band`).

    Returns:
        Mapping `row_type ("upper_anchor"/"lower_anchor") -> {"mean": float, "std": float,
        "n": int}`.
    """
    df = pl.read_csv(path)
    return {
        row["row_type"]: {
            "mean": float(row["mean"]),
            "std": float(row["std"]),
            "n": int(row["n"]),
        }
        for row in df.iter_rows(named=True)
    }


def compute_gap(genspace_mean: float, realspace_mean: float) -> float:
    """The real-space-vs-generated-space measurement-error gap for one anchor.

    Args:
        genspace_mean: The anchor's mean margin computed directly in generated-image space.
        realspace_mean: The matching real-image anchor's mean margin.

    Returns:
        `genspace_mean - realspace_mean`. Negative means generated images score LOWER-margin than
        real ones for the same anchor construction (the direction the root-cause narrative above
        expects, if generated images are systematically harder to match to real references than
        real images are to each other).
    """
    return genspace_mean - realspace_mean


def build_gap_table(
    genspace_summaries: list[dict[str, Any]],
    realspace_summaries: dict[str, dict[str, dict[str, float]]],
) -> pl.DataFrame:
    """Join every generated-space anchor summary against its matching real-space anchor.

    Args:
        genspace_summaries: Output of `summarize_margins_by_style_and_pooled` -- one row per
            (style_key or `ALL_STYLES_POOLED_KEY`, anchor_type, embedding_space), each carrying
            `mean`/`median`/`std`/`n`.
        realspace_summaries: `embedding_space -> load_realspace_anchor_summary(...)` output, must
            have an entry for every `embedding_space` present in `genspace_summaries`.

    Returns:
        One row per input `genspace_summaries` entry, with `realspace_row_type`,
        `realspace_mean`, `realspace_std`, `realspace_n`, and `gap` (`compute_gap`'s output,
        `= genspace_mean - realspace_mean`) columns appended.
    """
    rows: list[dict[str, Any]] = []
    for summary in genspace_summaries:
        row_type = REALSPACE_ROW_TYPE[summary["anchor_type"]]
        realspace = realspace_summaries[summary["embedding_space"]][row_type]
        rows.append(
            {
                **summary,
                "realspace_row_type": row_type,
                "realspace_mean": realspace["mean"],
                "realspace_std": realspace["std"],
                "realspace_n": realspace["n"],
                "gap": compute_gap(summary["mean"], realspace["mean"]),
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    """Run the full pipeline: generate -> free VRAM -> score -> summarize -> gap -> write."""
    wall_start = time.perf_counter()

    style_to_references = load_final_three_references(FINAL_THREE_MANIFEST_PATH)
    control_images = load_control_pool(CONTROL_MANIFEST_PATH)
    print(f"Styles ({len(style_to_references)}): {list(style_to_references)}")
    print(f"Control pool ({len(control_images)}): {[str(p) for p in control_images]}")

    anchor_images = generate_anchor_images(style_to_references)
    print(
        f"\nGenerated {len(anchor_images)} anchor images ({len(style_to_references)} styles x "
        f"{len(ANCHOR_TYPES)} anchor types x {len(SEEDS)} seeds)."
    )

    vram_before, vram_after = free_sdxl_pipeline()
    print(f"VRAM before free: {vram_before:.3f} GB, after free: {vram_after:.3f} GB")
    if vram_after >= vram_before:
        print("WARNING: VRAM did not measurably drop after freeing the SDXL pipeline.")

    scored_rows = score_anchor_images(anchor_images, style_to_references, control_images)
    wall_seconds = time.perf_counter() - wall_start

    genspace_summaries = summarize_margins_by_style_and_pooled(scored_rows)
    genspace_df = pl.DataFrame(genspace_summaries).select(
        ["style_key", "anchor_type", "embedding_space", "n", "mean", "median", "std"]
    )
    OUT_ANCHORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    genspace_df.write_csv(OUT_ANCHORS_PATH)
    print(f"\nWrote {OUT_ANCHORS_PATH}")
    for row in genspace_df.sort(["embedding_space", "anchor_type", "style_key"]).iter_rows(
        named=True
    ):
        print(
            f"  {row['embedding_space']:6s} {row['anchor_type']:18s} {row['style_key']:70s} "
            f"n={row['n']} mean={row['mean']:.4f} std={row['std']:.4f}"
        )

    realspace_summaries = {
        space: load_realspace_anchor_summary(path) for space, path in REALSPACE_PATHS.items()
    }
    gap_df = build_gap_table(genspace_summaries, realspace_summaries)
    OUT_GAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    gap_df.write_csv(OUT_GAP_PATH)
    print(f"\nWrote {OUT_GAP_PATH}")

    print("\n=== Real-space vs. generated-space gap (the measurement error this corrects) ===")
    for row in gap_df.sort(["embedding_space", "anchor_type", "style_key"]).iter_rows(named=True):
        print(
            f"  {row['embedding_space']:6s} {row['anchor_type']:18s} {row['style_key']:70s} "
            f"genspace_mean={row['mean']:.4f} realspace_mean={row['realspace_mean']:.4f} "
            f"gap={row['gap']:+.4f}"
        )

    print(f"\nTotal wall-clock (generation + scoring): {wall_seconds:.1f}s")


if __name__ == "__main__":
    main()
