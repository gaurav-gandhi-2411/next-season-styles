"""IP-Adapter conditioning-strength (`ip_adapter_scale`) sweep -- novelty vs. fidelity (task B4).

Generates one image per `ip_adapter_scale` in `SCALES` (fixed prompt, fixed seed, fixed reference
set, `local_sdxl` backend) for one chosen winning style, then measures each generated image's CLIP
cosine similarity to that style's real reference images. This is the primary evidence for picking
an operating-point `ip_adapter_scale`: IP-Adapter's mechanism predicts higher scale -> stronger
conditioning on the reference image(s) -> higher similarity, and the empirically-derived in-band
similarity range (`reports/tables/clip_similarity_band.csv`, `[0.8906, 0.9401]`) marks where a
generation is "novel but recognizably related" rather than unrelated (below) or a near-copy
(above).

Two-phase process, run strictly sequentially within this one script (never interleaved) because
this machine has 8 GB VRAM and SDXL alone peaks at ~6.5 GB -- see `nss.generate.clip_scoring`'s
module docstring for why CLIP scoring must never share a process turn with SDXL generation:

1. **Generation phase**: all 8 `generate_concept(backend="local_sdxl", ...)` calls, one per scale.
   `generate_concept` saves every image under `data/generated/local_sdxl/seed{seed}_00.png` keyed
   only by seed+batch-index, not by scale -- since this sweep reuses one fixed seed across all 8
   scales, each call would overwrite the previous scale's file if left in place. Each generated
   image is therefore copied to a scale-labeled path under `SWEEP_OUTPUT_DIR` immediately after
   its `generate_concept` call returns (the file is already fully written to disk by then), before
   the next scale's call can overwrite the source.
2. **Free phase**: the cached SDXL pipeline (`nss.generate.backends._PIPELINE_CACHE`) is
   explicitly deleted and `torch.cuda.empty_cache()` is called. VRAM is measured
   (`torch.cuda.memory_allocated()`) immediately before and after to confirm it actually dropped,
   rather than assuming the free succeeded.
3. **Scoring phase**: `nss.generate.clip_scoring.clip_similarity` (CPU-only) is imported only now,
   after the free phase, and run against each of the 8 sweep images.

Usage:
    uv run python -m nss.generate.scale_sweep
"""

from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless -- this module only writes PNGs, never shows a window.
import matplotlib.pyplot as plt
import polars as pl
from PIL import Image

from nss.generate import backends

STYLE_KEY = "Ladieswear || T-shirt || Jersey Basic || Black || Solid"
# final_rank_1's manifest rows: 8 candidate article images attempted, 7 successfully fetched (only
# 610776002 failed -- see reports/tables/exemplar_images_final_three.csv). Chosen over
# final_rank_2/3 (7 candidates attempted, 7 fetched each) because it is closest to a full 8-image
# reference set, per the task's stated preference.
REFERENCE_IMAGES = [
    Path("data/images/0554598001.jpg"),
    Path("data/images/0767862001.jpg"),
    Path("data/images/0778064001.jpg"),
    Path("data/images/0800691008.jpg"),
    Path("data/images/0827968004.jpg"),
    Path("data/images/0843685001.jpg"),
    Path("data/images/0865076001.jpg"),
]

SCALES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
SEED = 42  # hardcoded per project determinism convention

SWEEP_OUTPUT_DIR = Path("data/generated/sweep")
FIGURES_DIR = Path("reports/figures")
TABLES_DIR = Path("reports/tables")
SWEEP_PLOT_PATH = FIGURES_DIR / "novelty_fidelity_sweep.png"
SWEEP_IMAGES_PATH = FIGURES_DIR / "novelty_fidelity_sweep_images.png"
SWEEP_TABLE_PATH = TABLES_DIR / "novelty_fidelity_sweep.csv"

# Reused verbatim from reports/tables/clip_similarity_band.csv (derive_similarity_band.py's
# within-style-median / across-style-p90 band) -- do not re-derive here.
BAND_LOWER = 0.8905517101287842
BAND_UPPER = 0.9401496052742004


def style_description(style_key: str) -> str:
    """Build a short natural-language style description from a `||`-separated `style_key`.

    Args:
        style_key: `"Department || ProductType || ProductGroup || Colour || Pattern"`.

    Returns:
        A lowercase, generation-prompt-friendly description, e.g.
        `"black solid jersey basic t-shirt"` for `STYLE_KEY`.

    Raises:
        ValueError: if `style_key` does not split into exactly 5 `" || "`-separated parts.
    """
    parts = [p.strip() for p in style_key.split(" || ")]
    if len(parts) != 5:
        raise ValueError(f"Expected 5 ' || '-separated parts in style_key, got {len(parts)}")
    _department, product_type, product_group, colour, pattern = parts
    return f"{colour.lower()} {pattern.lower()} {product_group.lower()} {product_type.lower()}"


def build_prompt(style_key: str) -> str:
    """Build the fixed generation prompt used across the whole sweep for one `style_key`.

    Args:
        style_key: `"Department || ProductType || ProductGroup || Colour || Pattern"`.

    Returns:
        A prompt of the form `"a new <description> fashion concept for <department>, product
        photography, plain background"`.
    """
    department = style_key.split(" || ")[0].lower()
    description = style_description(style_key)
    return (
        f"a new {description} fashion concept for {department}, product photography, "
        "plain background"
    )


@dataclass(frozen=True)
class SweepImage:
    """One generated sweep image: its `ip_adapter_scale`, saved path, and generation time."""

    scale: float
    image_path: Path
    generation_seconds: float


def run_generation_phase(
    prompt: str,
    reference_images: list[Path],
    scales: tuple[float, ...],
    seed: int,
    output_dir: Path,
) -> list[SweepImage]:
    """Generate one `local_sdxl` image per `ip_adapter_scale` in `scales`.

    Args:
        prompt: Fixed text prompt used for every scale.
        reference_images: IP-Adapter reference images (only the first is used per-call -- see
            `nss.generate.backends` module docstring note 1).
        scales: `ip_adapter_scale` values to sweep, in the order to generate them.
        seed: Fixed seed shared across every scale (so the only varying factor is scale).
        output_dir: Directory to copy each scale's generated image into (scale-labeled filename),
            since `generate_concept` itself would overwrite the same seed-keyed path each call.

    Returns:
        One `SweepImage` per scale, in `scales` order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[SweepImage] = []
    for scale in scales:
        print(f"[generate] ip_adapter_scale={scale} ...")
        paths = backends.generate_concept(
            prompt=prompt,
            reference_images=reference_images,
            backend=backends.LOCAL_SDXL,
            ip_adapter_scale=scale,
            seed=seed,
            n=1,
        )
        source_path = paths[0]
        dest_path = output_dir / f"scale_{scale:.1f}.png"
        # Copy (not move) immediately -- source_path is fully written to disk by the time
        # generate_concept() returns, so this is safe before the next scale's call overwrites it.
        shutil.copy2(source_path, dest_path)
        source_meta = source_path.with_suffix(".json")
        if source_meta.exists():
            shutil.copy2(source_meta, dest_path.with_suffix(".json"))
        results.append(SweepImage(scale=scale, image_path=dest_path, generation_seconds=0.0))
        print(f"  -> {dest_path}")
    return results


def free_sdxl_pipeline() -> tuple[float, float]:
    """Explicitly free the cached SDXL pipeline and report VRAM allocated before/after.

    Returns:
        `(vram_before_gb, vram_after_gb)` -- `torch.cuda.memory_allocated()` immediately before
        and after deleting the cached pipeline and calling `torch.cuda.empty_cache()`. Confirms
        the free actually happened rather than assuming it.
    """
    import gc

    import torch

    vram_before = torch.cuda.memory_allocated() / 1e9
    backends._PIPELINE_CACHE.pop(backends.LOCAL_SDXL, None)
    gc.collect()
    torch.cuda.empty_cache()
    vram_after = torch.cuda.memory_allocated() / 1e9
    return vram_before, vram_after


def run_scoring_phase(sweep_images: list[SweepImage], reference_images: list[Path]) -> pl.DataFrame:
    """Score every sweep image's CLIP similarity to `reference_images` (mean + max).

    Must be called only after `free_sdxl_pipeline()` -- see module docstring for the VRAM-safety
    rationale (imports `nss.generate.clip_scoring` lazily, at call time, for the same reason).

    Args:
        sweep_images: Output of `run_generation_phase`.
        reference_images: Real reference images to score similarity against.

    Returns:
        A `polars.DataFrame` with columns `ip_adapter_scale`, `clip_similarity_mean`,
        `clip_similarity_max`, `in_band`, sorted by `ip_adapter_scale` ascending.
    """
    from nss.generate.clip_scoring import clip_similarity

    rows: list[dict[str, float | bool]] = []
    for sweep_image in sweep_images:
        print(f"[score] ip_adapter_scale={sweep_image.scale} ...")
        scores = clip_similarity(sweep_image.image_path, reference_images)
        in_band = BAND_LOWER <= scores["mean"] <= BAND_UPPER
        rows.append(
            {
                "ip_adapter_scale": sweep_image.scale,
                "clip_similarity_mean": scores["mean"],
                "clip_similarity_max": scores["max"],
                "in_band": in_band,
            }
        )
    return pl.DataFrame(rows).sort("ip_adapter_scale")


def is_monotonic_increasing(values: list[float]) -> bool:
    """Whether `values` are non-decreasing in order (higher scale -> higher/equal similarity).

    Args:
        values: Sequence of floats in scale-ascending order.

    Returns:
        True iff every consecutive pair satisfies `values[i] <= values[i + 1]`.
    """
    return all(a <= b for a, b in zip(values, values[1:], strict=False))


def plot_sweep(results: pl.DataFrame, out_path: Path = SWEEP_PLOT_PATH) -> Path:
    """Plot mean CLIP similarity vs. `ip_adapter_scale`, shading the derived in-band region.

    Args:
        results: Output of `run_scoring_phase`.
        out_path: Destination PNG path; parent directories are created if missing.

    Returns:
        `out_path`, for convenience chaining.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhspan(
        BAND_LOWER,
        BAND_UPPER,
        color="tab:green",
        alpha=0.15,
        label=f"derived in-band region [{BAND_LOWER:.4f}, {BAND_UPPER:.4f}]",
    )
    ax.plot(
        results["ip_adapter_scale"],
        results["clip_similarity_mean"],
        marker="o",
        color="tab:blue",
        linewidth=1.5,
        label="mean CLIP similarity to reference",
    )
    ax.set_xlabel("ip_adapter_scale")
    ax.set_ylabel("mean CLIP cosine similarity to reference images")
    ax.set_title(f"Novelty/fidelity sweep -- {STYLE_KEY}")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_image_strip(
    sweep_images: list[SweepImage], results: pl.DataFrame, out_path: Path = SWEEP_IMAGES_PATH
) -> Path:
    """Save a labeled grid of all 8 sweep images (scale + mean similarity per tile).

    Args:
        sweep_images: Output of `run_generation_phase`, in scale-ascending order.
        results: Output of `run_scoring_phase` (for per-scale mean similarity labels).
        out_path: Destination PNG path; parent directories are created if missing.

    Returns:
        `out_path`, for convenience chaining.
    """
    scales = results["ip_adapter_scale"].to_list()
    means = results["clip_similarity_mean"].to_list()
    scale_to_mean = dict(zip(scales, means, strict=True))
    n = len(sweep_images)
    ncols = 4
    nrows = -(-n // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.5 * nrows))
    axes_flat = axes.flatten()
    for ax, sweep_image in zip(axes_flat, sweep_images, strict=False):
        img = Image.open(sweep_image.image_path)
        ax.imshow(img)
        mean_sim = scale_to_mean.get(sweep_image.scale)
        title = f"scale={sweep_image.scale:.1f}"
        if mean_sim is not None:
            title += f"\nmean sim={mean_sim:.4f}"
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes_flat[n:]:
        ax.axis("off")
    fig.suptitle(f"Generated images across ip_adapter_scale -- {STYLE_KEY}", fontsize=12)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    """Run the full B4 sweep: generate -> free VRAM -> score -> plot -> report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    for path in REFERENCE_IMAGES:
        if not path.exists():
            raise FileNotFoundError(f"Reference image not found: {path}")

    prompt = build_prompt(STYLE_KEY)
    print(f"Style: {STYLE_KEY}")
    print(f"Prompt: {prompt!r}")
    print(f"Seed: {args.seed}")
    print(f"Reference images ({len(REFERENCE_IMAGES)}): {[str(p) for p in REFERENCE_IMAGES]}")

    wall_start = time.perf_counter()

    sweep_images = run_generation_phase(
        prompt, REFERENCE_IMAGES, SCALES, args.seed, SWEEP_OUTPUT_DIR
    )

    vram_before, vram_after = free_sdxl_pipeline()
    print(f"\nVRAM before free: {vram_before:.3f} GB, after free: {vram_after:.3f} GB")
    if vram_after >= vram_before:
        print("WARNING: VRAM did not measurably drop after freeing the SDXL pipeline.")

    results = run_scoring_phase(sweep_images, REFERENCE_IMAGES)
    wall_seconds = time.perf_counter() - wall_start

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results.write_csv(SWEEP_TABLE_PATH)
    print(f"\nWrote {SWEEP_TABLE_PATH}")
    # Print rows as plain ASCII text rather than polars' default DataFrame repr -- the latter uses
    # box-drawing Unicode characters that raise UnicodeEncodeError on a cp1252 Windows console.
    for row in results.iter_rows(named=True):
        print(
            f"  scale={row['ip_adapter_scale']:.1f}  "
            f"mean={row['clip_similarity_mean']:.4f}  "
            f"max={row['clip_similarity_max']:.4f}  "
            f"in_band={row['in_band']}"
        )

    monotonic = is_monotonic_increasing(results["clip_similarity_mean"].to_list())
    in_band_scales = results.filter(pl.col("in_band"))["ip_adapter_scale"].to_list()
    print(f"\nMonotonic (higher scale -> higher/equal mean similarity): {monotonic}")
    print(f"In-band scales [{BAND_LOWER:.4f}, {BAND_UPPER:.4f}]: {in_band_scales}")

    plot_path = plot_sweep(results)
    strip_path = plot_image_strip(sweep_images, results)
    print(f"\nWrote {plot_path}")
    print(f"Wrote {strip_path}")

    print(f"\nTotal wall-clock (generation + scoring): {wall_seconds:.1f}s")


if __name__ == "__main__":
    main()
