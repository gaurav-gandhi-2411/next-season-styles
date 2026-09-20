"""Empirically derive the CLIP cosine-similarity "in-band" thresholds used by `is_in_band`.

Uses every real, successfully-fetched exemplar image across the two manifests written by the
exemplar selection (`reports/tables/exemplar_images.csv`) and the final-three fetch
(`reports/tables/exemplar_images_final_three.csv`), combined and de-duplicated by `article_id` --
`final_rank_1`'s style_key is identical to `winner_rank_1`'s (see
`nss.data.select_final_three_exemplars`'s docstring), so the union gives ~6 distinct styles (3
original winners + 2 new emerging winners + 1 control) instead of just the 4 either manifest has
alone.

For every pair of images belonging to the SAME style_key, we compute CLIP cosine similarity --
that is the "within-style" distribution: how similar real images of one style are to each other.
For every pair of images belonging to DIFFERENT style_keys, we compute the "across-style"
distribution: how similar real images of two genuinely unrelated styles are to each other by
chance (same product-photography conventions, same CLIP embedding space, no shared style).

Band derivation:

- **Lower bound** = the across-style distribution's `LOWER_BOUND_PERCENTILE`th percentile (90th
  by default -- see `derive_band`'s docstring for why 90th over 95th was chosen here).
  Interpretation: above this similarity, a generated image is statistically indistinguishable
  from two images that are simply different, unrelated styles -- i.e. no longer explainable by
  "any two H&M product photos share some baseline visual similarity" alone, so it differentiates
  genuine own-style relatedness from that noise floor.
- **Upper bound** = the within-style distribution's median. Interpretation: above this, a
  generated image sits in the *upper half* of how similar two genuinely-the-same-style real
  images are to each other -- i.e. it is at least as similar to its references as those
  references are to each other, which is the copy/near-duplicate regime rather than a novel but
  related generation.

SMALL-SAMPLE CAVEAT (read before trusting these numbers): this project has only ~6 distinct
styles with a handful of images each (see the printed per-style counts and `n_pairs` columns in
the output CSV). This is a rough, revisable empirical estimate suitable for a first pass, NOT a
large-sample-validated statistical threshold -- percentiles computed from a few dozen pairs have
wide, unreported confidence intervals, and adding a 7th style could shift the band noticeably.
Revisit once more styles/images are available.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import polars as pl

from nss.generate.clip_scoring import _cosine, embed_image

EXEMPLAR_MANIFEST_PATHS = (
    Path("reports/tables/exemplar_images.csv"),
    Path("reports/tables/exemplar_images_final_three.csv"),
)
DISTRIBUTIONS_OUT_PATH = Path("reports/tables/clip_similarity_distributions.csv")
BAND_OUT_PATH = Path("reports/tables/clip_similarity_band.csv")

# 90th over 95th: with only a few dozen across-style pairs (see module docstring), the 95th
# percentile sits within interpolation distance of the single highest observed pair -- one
# unusual pair could swing it substantially. The 90th percentile is a mildly more stable choice
# for a sample this small while still representing a high (conservative) tail of the "different
# styles happen to look somewhat alike" distribution. Both are reported in the output CSV so a
# reader can substitute the 95th if more data later makes it the better choice.
LOWER_BOUND_PERCENTILE = 90.0


def load_exemplar_manifest(paths: tuple[Path, ...] = EXEMPLAR_MANIFEST_PATHS) -> pl.DataFrame:
    """Load, combine, and de-duplicate the exemplar-image manifests by `article_id`.

    Only rows with `fetch_success == True` are kept -- a manifest row can reference an image path
    that was never actually downloaded (see `exemplar_images.csv`'s `winner_rank_1`/`610776002`
    row), and building similarity distributions from a missing file is not possible.

    Args:
        paths: Manifest CSV paths to combine (the exemplar and final-three manifests by default).

    Returns:
        De-duplicated frame with columns `style_key`, `article_id`, `local_image_path`, sorted by
        `style_key`, `article_id` for determinism.
    """
    frames = [pl.read_csv(p) for p in paths]
    combined = pl.concat(
        [f.select(["style_key", "article_id", "local_image_path", "fetch_success"]) for f in frames]
    )
    deduped = (
        combined.filter(pl.col("fetch_success"))
        .unique(subset=["article_id"], keep="first")
        .sort(["style_key", "article_id"])
    )
    return deduped


def group_by_style(manifest: pl.DataFrame) -> dict[str, list[Path]]:
    """Group a de-duplicated exemplar manifest's image paths by `style_key`.

    Args:
        manifest: Output of `load_exemplar_manifest` (or any frame with `style_key` and
            `local_image_path` columns, already filtered to existing/fetched images).

    Returns:
        Mapping of `style_key -> sorted list of image Path`s. Only paths that actually exist on
        disk are included (defensive: `fetch_success == True` should already guarantee this).
    """
    grouped: dict[str, list[Path]] = {}
    for row in manifest.iter_rows(named=True):
        path = Path(row["local_image_path"])
        if not path.exists():
            continue
        grouped.setdefault(row["style_key"], []).append(path)
    for paths in grouped.values():
        paths.sort()
    return grouped


def embed_all(style_to_images: dict[str, list[Path]]) -> dict[Path, np.ndarray]:
    """Compute (and cache) the CLIP embedding for every image across all styles, once each.

    Args:
        style_to_images: Output of `group_by_style`.

    Returns:
        Mapping of image `Path -> L2-normalized CLIP embedding`.
    """
    embeddings: dict[Path, np.ndarray] = {}
    for paths in style_to_images.values():
        for path in paths:
            if path not in embeddings:
                embeddings[path] = embed_image(path)
    return embeddings


def within_style_similarities(
    style_to_images: dict[str, list[Path]], embeddings: dict[Path, np.ndarray]
) -> list[float]:
    """Pairwise CLIP cosine similarity between every pair of images of the SAME style.

    Styles with fewer than 2 images contribute no pairs (there is nothing to pair within them).

    Args:
        style_to_images: Output of `group_by_style`.
        embeddings: Output of `embed_all`.

    Returns:
        Flat list of pairwise cosine similarities across all styles with >= 2 images.
    """
    sims: list[float] = []
    for paths in style_to_images.values():
        for img_a, img_b in itertools.combinations(paths, 2):
            sims.append(_cosine(embeddings[img_a], embeddings[img_b]))
    return sims


def across_style_similarities(
    style_to_images: dict[str, list[Path]], embeddings: dict[Path, np.ndarray]
) -> list[float]:
    """Pairwise CLIP cosine similarity between every pair of images from DIFFERENT styles.

    Args:
        style_to_images: Output of `group_by_style`.
        embeddings: Output of `embed_all`.

    Returns:
        Flat list of pairwise cosine similarities across every distinct pair of styles.
    """
    sims: list[float] = []
    styles = sorted(style_to_images)
    for style_a, style_b in itertools.combinations(styles, 2):
        for img_a in style_to_images[style_a]:
            for img_b in style_to_images[style_b]:
                sims.append(_cosine(embeddings[img_a], embeddings[img_b]))
    return sims


def summarize(distribution: list[float]) -> dict[str, float]:
    """Summary statistics for a flat list of pairwise cosine similarities.

    Args:
        distribution: Non-empty list of cosine similarity values.

    Returns:
        Dict with `n_pairs`, `mean`, `median`, `std`, `min`, `max`, `p90`, `p95`.

    Raises:
        ValueError: if `distribution` is empty.
    """
    if not distribution:
        raise ValueError("distribution must be non-empty")
    arr = np.array(distribution, dtype=np.float64)
    return {
        "n_pairs": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def derive_band(
    within_summary: dict[str, float],
    across_summary: dict[str, float],
    lower_percentile: float = LOWER_BOUND_PERCENTILE,
) -> dict[str, float]:
    """Derive the `(lower, upper)` in-band thresholds from the two distributions' summary stats.

    Args:
        within_summary: Output of `summarize(within_style_similarities(...))`.
        across_summary: Output of `summarize(across_style_similarities(...))`.
        lower_percentile: Which across-style percentile to use as `lower` (90 or 95 -- see module
            docstring for the justification of the 90th-percentile default).

    Returns:
        `{"lower": float, "upper": float, "lower_percentile_used": float}`.
    """
    percentile_key = f"p{int(lower_percentile)}"
    return {
        "lower": across_summary[percentile_key],
        "upper": within_summary["median"],
        "lower_percentile_used": lower_percentile,
    }


def main() -> None:
    """CLI entry point: compute within/across-style CLIP similarity distributions + derive band.

    Writes `reports/tables/clip_similarity_distributions.csv` (per-distribution summary stats)
    and `reports/tables/clip_similarity_band.csv` (the derived `lower`/`upper` band), and prints a
    full report including per-style image counts and the small-sample caveat.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-paths", type=Path, nargs="*", default=list(EXEMPLAR_MANIFEST_PATHS)
    )
    parser.add_argument("--distributions-out-path", type=Path, default=DISTRIBUTIONS_OUT_PATH)
    parser.add_argument("--band-out-path", type=Path, default=BAND_OUT_PATH)
    parser.add_argument("--lower-percentile", type=float, default=LOWER_BOUND_PERCENTILE)
    args = parser.parse_args()

    manifest = load_exemplar_manifest(tuple(args.manifest_paths))
    style_to_images = group_by_style(manifest)

    print(f"{len(style_to_images)} distinct styles with >=1 fetched image:")
    for style_key, paths in sorted(style_to_images.items()):
        print(f"  {style_key}: {len(paths)} images")

    embeddings = embed_all(style_to_images)
    print(f"Embedded {len(embeddings)} distinct images (CLIP ViT-L/14, CPU).")

    within = within_style_similarities(style_to_images, embeddings)
    across = across_style_similarities(style_to_images, embeddings)
    within_summary = summarize(within)
    across_summary = summarize(across)

    print("\nWithin-style pairwise CLIP cosine similarity:")
    for key, value in within_summary.items():
        print(f"  {key}: {value}")
    print("\nAcross-style pairwise CLIP cosine similarity:")
    for key, value in across_summary.items():
        print(f"  {key}: {value}")

    band = derive_band(within_summary, across_summary, args.lower_percentile)
    print(
        f"\nDerived band: lower={band['lower']:.4f} "
        f"(across-style p{int(band['lower_percentile_used'])}), "
        f"upper={band['upper']:.4f} (within-style median)"
    )
    print(
        "SMALL-SAMPLE CAVEAT: "
        f"{len(style_to_images)} styles, {within_summary['n_pairs']} within-style pairs, "
        f"{across_summary['n_pairs']} across-style pairs -- a rough, revisable empirical "
        "estimate, not a large-sample-validated threshold. See module docstring."
    )

    distributions_df = pl.DataFrame(
        [
            {"distribution": "within_style", **within_summary},
            {"distribution": "across_style", **across_summary},
        ]
    )
    args.distributions_out_path.parent.mkdir(parents=True, exist_ok=True)
    distributions_df.write_csv(args.distributions_out_path)
    print(f"\nWrote distribution summary to {args.distributions_out_path}")

    band_df = pl.DataFrame(
        [
            {
                "lower": band["lower"],
                "upper": band["upper"],
                "lower_percentile_used": band["lower_percentile_used"],
                "n_styles": len(style_to_images),
                "n_within_style_pairs": within_summary["n_pairs"],
                "n_across_style_pairs": across_summary["n_pairs"],
            }
        ]
    )
    band_df.write_csv(args.band_out_path)
    print(f"Wrote derived band to {args.band_out_path}")


if __name__ == "__main__":
    main()
