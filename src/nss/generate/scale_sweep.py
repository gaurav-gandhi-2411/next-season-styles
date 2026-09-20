"""IP-Adapter conditioning-strength (`ip_adapter_scale`) sweep -- novelty vs. fidelity.

SUPERSEDES the earlier absolute-CLIP-cosine sweep (the old `clip_similarity`/`is_in_band` approach,
`nss.generate.derive_similarity_band`'s band). That approach was invalidated: its band
was derived from catalogue-photo-vs-catalogue-photo pairs but applied to
generated-image-vs-catalogue-photo pairs -- a different, confounded distribution (see
`nss.generate.margin_scoring`'s module docstring for the full mechanism). This module replaces it
with the margin-based approach (`nss.generate.margin_scoring.margin`), scored independently in TWO
embedding spaces (CLIP and DINOv2, `nss.generate.clip_scoring` / `nss.generate.dino_scoring`),
against the margin-based in-band ranges derived in `derive_margin_band`
(`reports/tables/margin_anchors_clip.csv`, `reports/tables/margin_anchors_dinov2.csv`).

Generates 3 images per `ip_adapter_scale` in `SCALES` (one per seed in `SEEDS`) -- fixed prompt,
fixed reference set, `local_sdxl` backend -- for one chosen winning style (`STYLE_KEY`, unchanged
from the earlier sweep for direct comparability), then scores each of the 24 images' margin against
that style's own real references, in both embedding spaces. Repeating each scale across 3 seeds
(rather than the earlier single seed per scale) surfaces per-scale sampling noise directly, instead
of silently baking one seed's luck into the only data point at each scale.

CONTROL POOL -- CORRECTED FROM THE ORIGINAL PREMISE, DOCUMENTED HERE: the control pool used as
the subtracted term in every `margin()` call in this module is `reports/tables/exemplar_images.csv`
`role == "control"` rows (1 style, 5 images -- confirmed on disk, see `n_control_images` in
`reports/tables/margin_anchors_clip.csv`), loaded via
`nss.generate.derive_margin_band.load_control_pool`. This is NOT the 20-style/160-image
`exemplar_images_margin_reference.csv` set. Re-reading `nss.generate.derive_margin_band`
confirms `exemplar_images_margin_reference.csv`'s 20 styles were the "non-control" TEST/anchor-
derivation set (supplying the leave-one-out own-style images for the upper anchor and the
cross-style images for the lower anchor) -- the second, SUBTRACTED term of every `margin()` call
throughout that module's anchor derivation was always the 5-image control-role set. The two derived
bands (`margin_anchors_clip.csv` / `margin_anchors_dinov2.csv`) are calibrated against that same
5-image control pool; scoring this sweep's images against a different, larger control pool would
produce margins that are not numerically comparable to those bands. Using the 5-image control pool
here is therefore required for the derived bands to mean anything for this sweep, not merely a
minor implementation choice.

Two-phase process, run strictly sequentially within this one script (never interleaved) because
this machine has 8 GB VRAM and SDXL alone peaks at ~6.5 GB -- see `nss.generate.clip_scoring`'s
module docstring for why embedding-model scoring must never share a process turn with SDXL
generation:

1. **Generation phase**: all 24 `generate_concept(backend="local_sdxl", ...)` calls, one per
   (scale, seed) pair. `generate_concept` saves every image under
   `data/generated/local_sdxl/seed{seed}_00.png` keyed only by seed+batch-index, not by scale --
   since a fixed seed is reused across multiple scales, each call would overwrite an
   already-produced image for the same seed if left in place. Each generated image is therefore
   copied to a scale+seed-labeled path under `SWEEP_OUTPUT_DIR` immediately after its
   `generate_concept` call returns (the file is already fully written to disk by then), before the
   next call can overwrite the source.
2. **Free phase**: the cached SDXL pipeline (`nss.generate.backends._PIPELINE_CACHE`) is
   explicitly deleted and `torch.cuda.empty_cache()` is called. VRAM is measured
   (`torch.cuda.memory_allocated()`) immediately before and after to confirm it actually dropped,
   rather than assuming the free succeeded.
3. **Scoring phase**: `nss.generate.clip_scoring.embed_image` and `nss.generate.dino_scoring.
   embed_image` (both CPU-only) are imported only now, after the free phase, and run against each
   of the 24 sweep images plus the style/control reference sets.

Usage:
    uv run python -m nss.generate.scale_sweep
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless -- this module only writes PNGs, never shows a window.
import matplotlib.pyplot as plt
import polars as pl
from PIL import Image

from nss.generate import backends
from nss.generate.derive_margin_band import CONTROL_MANIFEST_PATH, load_control_pool

STYLE_KEY = "Ladieswear || T-shirt || Jersey Basic || Black || Solid"
# final_rank_1's manifest rows: 8 candidate article images attempted, 7 successfully fetched (only
# 610776002 failed -- see reports/tables/exemplar_images_final_three.csv). Chosen over
# final_rank_2/3 (7 candidates attempted, 7 fetched each) because it is closest to a full 8-image
# reference set, as preferred. Unchanged from the earlier sweep for direct comparability.
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
SEEDS = (42, 43, 44)  # 3 fixed, documented seeds per scale -- surfaces per-scale sampling noise.

SWEEP_OUTPUT_DIR = Path("data/generated/sweep")
FIGURES_DIR = Path("reports/figures")
TABLES_DIR = Path("reports/tables")
SWEEP_PLOT_PATH = FIGURES_DIR / "novelty_fidelity_sweep.png"
SWEEP_IMAGES_PATH = FIGURES_DIR / "novelty_fidelity_sweep_images.png"
SWEEP_TABLE_PATH = TABLES_DIR / "novelty_fidelity_sweep.csv"
SWEEP_RAW_TABLE_PATH = TABLES_DIR / "novelty_fidelity_sweep_raw.csv"

CLIP_BAND_PATH = TABLES_DIR / "margin_anchors_clip.csv"
DINO_BAND_PATH = TABLES_DIR / "margin_anchors_dinov2.csv"


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

    Unchanged verbatim from the earlier sweep's `build_prompt` for a fair, direct comparison against
    the old (now-superseded) sweep results.

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
    """One generated sweep image: its `ip_adapter_scale`, seed, and saved path."""

    scale: float
    seed: int
    image_path: Path


def run_generation_phase(
    prompt: str,
    reference_images: list[Path],
    scales: tuple[float, ...],
    seeds: tuple[int, ...],
    output_dir: Path,
) -> list[SweepImage]:
    """Generate one `local_sdxl` image per `(ip_adapter_scale, seed)` pair in `scales` x `seeds`.

    Args:
        prompt: Fixed text prompt used for every (scale, seed) pair.
        reference_images: IP-Adapter reference images (only the first is used per-call -- see
            `nss.generate.backends` module docstring note 1).
        scales: `ip_adapter_scale` values to sweep, in the order to generate them.
        seeds: Seeds to repeat at every scale, in the order to generate them.
        output_dir: Directory to copy each (scale, seed)'s generated image into (labeled
            filename), since `generate_concept` itself would overwrite a same-seed path across
            different scales.

    Returns:
        One `SweepImage` per `(scale, seed)` pair, in `scales`-major, `seeds`-minor order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[SweepImage] = []
    for scale in scales:
        for seed in seeds:
            print(f"[generate] ip_adapter_scale={scale} seed={seed} ...")
            paths = backends.generate_concept(
                prompt=prompt,
                reference_images=reference_images,
                backend=backends.LOCAL_SDXL,
                ip_adapter_scale=scale,
                seed=seed,
                n=1,
            )
            source_path = paths[0]
            dest_path = output_dir / f"scale_{scale:.1f}_seed{seed}.png"
            # Copy (not move) immediately -- source_path is fully written to disk by the time
            # generate_concept() returns, so this is safe before a later call overwrites it.
            shutil.copy2(source_path, dest_path)
            source_meta = source_path.with_suffix(".json")
            if source_meta.exists():
                shutil.copy2(source_meta, dest_path.with_suffix(".json"))
            results.append(SweepImage(scale=scale, seed=seed, image_path=dest_path))
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


def load_margin_band(path: Path) -> tuple[float, float]:
    """Read the derived `(band_lower, band_upper)` margin thresholds from a `margin_anchors_*.csv`.

    Both rows of that CSV (`upper_anchor`/`lower_anchor`) carry the same `band_lower`/`band_upper`
    columns (the derived band is a single pair of numbers per embedding space), so either row
    works -- the first is used.

    Args:
        path: Path to `reports/tables/margin_anchors_clip.csv` or `..._dinov2.csv`.

    Returns:
        `(band_lower, band_upper)`.
    """
    df = pl.read_csv(path)
    row = df.row(0, named=True)
    return float(row["band_lower"]), float(row["band_upper"])


def run_scoring_phase(
    sweep_images: list[SweepImage],
    reference_images: list[Path],
    control_images: list[Path],
) -> pl.DataFrame:
    """Score every sweep image's margin against `reference_images`, in both CLIP and DINOv2 space.

    Must be called only after `free_sdxl_pipeline()` -- see module docstring for the VRAM-safety
    rationale (imports `nss.generate.clip_scoring`/`nss.generate.dino_scoring` lazily, at call
    time, for the same reason).

    Args:
        sweep_images: Output of `run_generation_phase`.
        reference_images: Real reference images for the sweep's target style (the `margin()`
            "style_reference_embeddings" term).
        control_images: Real images for the control pool (the `margin()`
            "control_pool_embeddings" term -- see module docstring CONTROL POOL note).

    Returns:
        A `polars.DataFrame` with columns `ip_adapter_scale`, `seed`, `clip_margin`, `dino_margin`,
        sorted by `ip_adapter_scale`, `seed` ascending.
    """
    from nss.generate import clip_scoring, dino_scoring
    from nss.generate.margin_scoring import margin

    clip_style_refs = [clip_scoring.embed_image(p) for p in reference_images]
    clip_control = [clip_scoring.embed_image(p) for p in control_images]
    dino_style_refs = [dino_scoring.embed_image(p) for p in reference_images]
    dino_control = [dino_scoring.embed_image(p) for p in control_images]

    rows: list[dict[str, float | int]] = []
    for sweep_image in sweep_images:
        print(f"[score] ip_adapter_scale={sweep_image.scale} seed={sweep_image.seed} ...")
        clip_concept = clip_scoring.embed_image(sweep_image.image_path)
        dino_concept = dino_scoring.embed_image(sweep_image.image_path)
        rows.append(
            {
                "ip_adapter_scale": sweep_image.scale,
                "seed": sweep_image.seed,
                "clip_margin": margin(clip_concept, clip_style_refs, clip_control),
                "dino_margin": margin(dino_concept, dino_style_refs, dino_control),
            }
        )
    return pl.DataFrame(rows).sort(["ip_adapter_scale", "seed"])


def aggregate_by_scale(
    raw: pl.DataFrame, clip_band: tuple[float, float], dino_band: tuple[float, float]
) -> pl.DataFrame:
    """Aggregate the per-(scale, seed) raw margins to per-scale mean + std (sample, n=len(seeds)).

    Spread is reported as sample standard deviation (`ddof=1`, matching
    `nss.generate.derive_margin_band.summarize`'s convention elsewhere in this project) across the
    seeds at each scale -- documented here explicitly so the spread statistic is stated.

    Args:
        raw: Output of `run_scoring_phase` (one row per `(scale, seed)` pair).
        clip_band: `(lower, upper)` CLIP margin band from `load_margin_band(CLIP_BAND_PATH)`.
        dino_band: `(lower, upper)` DINOv2 margin band from `load_margin_band(DINO_BAND_PATH)`.

    Returns:
        One row per `ip_adapter_scale`, columns: `ip_adapter_scale`, `clip_margin_mean`,
        `clip_margin_std`, `dino_margin_mean`, `dino_margin_std`, `clip_in_band` (mean within
        `clip_band`), `dino_in_band` (mean within `dino_band`), `n_seeds`. Sorted by
        `ip_adapter_scale` ascending.
    """
    clip_lower, clip_upper = clip_band
    dino_lower, dino_upper = dino_band
    aggregated = raw.group_by("ip_adapter_scale", maintain_order=True).agg(
        pl.col("clip_margin").mean().alias("clip_margin_mean"),
        pl.col("clip_margin").std(ddof=1).fill_null(0.0).alias("clip_margin_std"),
        pl.col("dino_margin").mean().alias("dino_margin_mean"),
        pl.col("dino_margin").std(ddof=1).fill_null(0.0).alias("dino_margin_std"),
        pl.len().alias("n_seeds"),
    )
    aggregated = aggregated.with_columns(
        pl.col("clip_margin_mean").is_between(clip_lower, clip_upper).alias("clip_in_band"),
        pl.col("dino_margin_mean").is_between(dino_lower, dino_upper).alias("dino_in_band"),
    )
    return aggregated.sort("ip_adapter_scale")


def is_monotonic_increasing(values: list[float]) -> bool:
    """Whether `values` are non-decreasing in order (higher scale -> higher/equal margin).

    Args:
        values: Sequence of floats in scale-ascending order.

    Returns:
        True iff every consecutive pair satisfies `values[i] <= values[i + 1]`.
    """
    return all(a <= b for a, b in zip(values, values[1:], strict=False))


def select_operating_scale(aggregated: pl.DataFrame) -> dict[str, Any]:
    """Report the `ip_adapter_scale`(s) recommended by CLIP, by DINOv2, and by both together.

    Does not force a single number when CLIP and DINOv2 disagree -- see module docstring:
    if no scale lands in-band for both metrics simultaneously, the honest answer is
    to report each metric's own in-band scales rather than picking one arbitrarily.

    Args:
        aggregated: Output of `aggregate_by_scale` (must have `ip_adapter_scale`, `clip_in_band`,
            `dino_in_band` columns).

    Returns:
        `{"clip_in_band_scales": list[float], "dino_in_band_scales": list[float],
        "both_in_band_scales": list[float], "recommended_scale": float | None}` --
        `recommended_scale` is the smallest scale in-band for both metrics (least aggressive
        conditioning that clears both bars), or `None` if no such scale exists.
    """
    clip_scales = aggregated.filter(pl.col("clip_in_band"))["ip_adapter_scale"].to_list()
    dino_scales = aggregated.filter(pl.col("dino_in_band"))["ip_adapter_scale"].to_list()
    both_scales = sorted(set(clip_scales) & set(dino_scales))
    return {
        "clip_in_band_scales": clip_scales,
        "dino_in_band_scales": dino_scales,
        "both_in_band_scales": both_scales,
        "recommended_scale": both_scales[0] if both_scales else None,
    }


def plot_sweep(
    aggregated: pl.DataFrame,
    clip_band: tuple[float, float],
    dino_band: tuple[float, float],
    out_path: Path = SWEEP_PLOT_PATH,
) -> Path:
    """Two-panel plot: mean margin (+/- 1 std error bars) vs. `ip_adapter_scale`, CLIP and DINOv2.

    Args:
        aggregated: Output of `aggregate_by_scale`.
        clip_band: `(lower, upper)` CLIP margin band, shaded on the left panel.
        dino_band: `(lower, upper)` DINOv2 margin band, shaded on the right panel.
        out_path: Destination PNG path; parent directories are created if missing.

    Returns:
        `out_path`, for convenience chaining.
    """
    scales = aggregated["ip_adapter_scale"].to_list()
    fig, (ax_clip, ax_dino) = plt.subplots(1, 2, figsize=(15, 6))

    panels: list[tuple[Any, str, str, str, str]] = [
        (ax_clip, "clip_margin_mean", "clip_margin_std", "CLIP", "tab:blue"),
        (ax_dino, "dino_margin_mean", "dino_margin_std", "DINOv2", "tab:orange"),
    ]
    bands = {"CLIP": clip_band, "DINOv2": dino_band}
    for ax, mean_col, std_col, label, color in panels:
        lower, upper = bands[label]
        ax.axhspan(
            lower,
            upper,
            color="tab:green",
            alpha=0.15,
            label=f"derived in-band region [{lower:.4f}, {upper:.4f}]",
        )
        ax.errorbar(
            scales,
            aggregated[mean_col].to_list(),
            yerr=aggregated[std_col].to_list(),
            marker="o",
            color=color,
            linewidth=1.5,
            capsize=4,
            label=f"mean {label} margin +/- 1 std (n=3 seeds)",
        )
        ax.set_xlabel("ip_adapter_scale")
        ax.set_ylabel(f"{label} margin (own-style - control)")
        ax.set_title(f"{label} margin vs. ip_adapter_scale")
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"Novelty/fidelity sweep -- {STYLE_KEY}", fontsize=12)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_image_strip(
    sweep_images: list[SweepImage], raw: pl.DataFrame, out_path: Path = SWEEP_IMAGES_PATH
) -> Path:
    """Save a labeled grid of all 24 sweep images (scale rows x seed columns).

    Args:
        sweep_images: Output of `run_generation_phase`, in scale-major/seed-minor order.
        raw: Output of `run_scoring_phase` (for per-image margin labels).
        out_path: Destination PNG path; parent directories are created if missing.

    Returns:
        `out_path`, for convenience chaining.
    """
    margin_lookup = {
        (row["ip_adapter_scale"], row["seed"]): (row["clip_margin"], row["dino_margin"])
        for row in raw.iter_rows(named=True)
    }
    scales = sorted({si.scale for si in sweep_images})
    seeds = sorted({si.seed for si in sweep_images})
    image_lookup = {(si.scale, si.seed): si.image_path for si in sweep_images}

    nrows, ncols = len(scales), len(seeds)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.5 * nrows))
    for r, scale in enumerate(scales):
        for c, seed in enumerate(seeds):
            ax = axes[r][c] if nrows > 1 else axes[c]
            image_path = image_lookup[(scale, seed)]
            img = Image.open(image_path)
            ax.imshow(img)
            clip_m, dino_m = margin_lookup[(scale, seed)]
            ax.set_title(
                f"scale={scale:.1f} seed={seed}\nclip={clip_m:.4f} dino={dino_m:.4f}", fontsize=9
            )
            ax.axis("off")
    fig.suptitle(f"Generated images across ip_adapter_scale x seed -- {STYLE_KEY}", fontsize=12)
    # rect leaves headroom for the suptitle above the top row's own per-tile titles -- without it,
    # tight_layout() lets the suptitle collide with row 1's "scale=... clip=... dino=..." labels on
    # this tall (nrows=8) grid.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    """Run the full sweep: generate -> free VRAM -> score -> aggregate -> plot -> report."""
    for path in REFERENCE_IMAGES:
        if not path.exists():
            raise FileNotFoundError(f"Reference image not found: {path}")
    control_images = load_control_pool(CONTROL_MANIFEST_PATH)

    prompt = build_prompt(STYLE_KEY)
    print(f"Style: {STYLE_KEY}")
    print(f"Prompt: {prompt!r}")
    print(f"Seeds: {SEEDS}")
    print(f"Reference images ({len(REFERENCE_IMAGES)}): {[str(p) for p in REFERENCE_IMAGES]}")
    print(f"Control pool ({len(control_images)}): {[str(p) for p in control_images]}")

    wall_start = time.perf_counter()

    sweep_images = run_generation_phase(prompt, REFERENCE_IMAGES, SCALES, SEEDS, SWEEP_OUTPUT_DIR)

    vram_before, vram_after = free_sdxl_pipeline()
    print(f"\nVRAM before free: {vram_before:.3f} GB, after free: {vram_after:.3f} GB")
    if vram_after >= vram_before:
        print("WARNING: VRAM did not measurably drop after freeing the SDXL pipeline.")

    raw = run_scoring_phase(sweep_images, REFERENCE_IMAGES, control_images)
    wall_seconds = time.perf_counter() - wall_start

    clip_band = load_margin_band(CLIP_BAND_PATH)
    dino_band = load_margin_band(DINO_BAND_PATH)
    print(f"\nCLIP band: {clip_band}")
    print(f"DINOv2 band: {dino_band}")

    aggregated = aggregate_by_scale(raw, clip_band, dino_band)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    raw.write_csv(SWEEP_RAW_TABLE_PATH)
    aggregated.write_csv(SWEEP_TABLE_PATH)
    print(f"\nWrote {SWEEP_RAW_TABLE_PATH}")
    print(f"Wrote {SWEEP_TABLE_PATH}")
    # Print rows as plain ASCII text rather than polars' default DataFrame repr -- the latter uses
    # box-drawing Unicode characters that raise UnicodeEncodeError on a cp1252 Windows console.
    for row in aggregated.iter_rows(named=True):
        print(
            f"  scale={row['ip_adapter_scale']:.1f}  "
            f"clip_mean={row['clip_margin_mean']:.4f} (std={row['clip_margin_std']:.4f})  "
            f"dino_mean={row['dino_margin_mean']:.4f} (std={row['dino_margin_std']:.4f})  "
            f"clip_in_band={row['clip_in_band']}  dino_in_band={row['dino_in_band']}"
        )

    clip_monotonic = is_monotonic_increasing(aggregated["clip_margin_mean"].to_list())
    dino_monotonic = is_monotonic_increasing(aggregated["dino_margin_mean"].to_list())
    print(
        f"\nCLIP monotonic (higher scale -> higher/equal mean margin, n=3 seeds/scale): "
        f"{clip_monotonic}"
    )
    print(f"DINOv2 monotonic (same, n=3 seeds/scale): {dino_monotonic}")

    operating = select_operating_scale(aggregated)
    print(f"\nCLIP in-band scales: {operating['clip_in_band_scales']}")
    print(f"DINOv2 in-band scales: {operating['dino_in_band_scales']}")
    print(f"Both in-band: {operating['both_in_band_scales']}")
    print(f"Recommended operating scale: {operating['recommended_scale']}")

    plot_path = plot_sweep(aggregated, clip_band, dino_band)
    strip_path = plot_image_strip(sweep_images, raw)
    print(f"\nWrote {plot_path}")
    print(f"Wrote {strip_path}")

    print(f"\nTotal wall-clock (generation + scoring): {wall_seconds:.1f}s")


if __name__ == "__main__":
    main()
