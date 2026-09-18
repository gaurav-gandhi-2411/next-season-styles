"""Derive margin-based in-band thresholds for concept-vs-catalogue novelty scoring (task C2).

Replaces B3's `nss.generate.derive_similarity_band` (absolute-cosine band, derived from the wrong
distribution -- catalogue-vs-catalogue pairs applied to generated-vs-catalogue pairs; see
`nss.generate.margin_scoring`'s docstring for the full confound explanation) with a margin-based
band, cross-validated by construction, computed independently in TWO embedding spaces (CLIP and
DINOv2 -- see `nss.generate.clip_scoring` / `nss.generate.dino_scoring`).

REFERENCE BASE (task C2 step 1): 23 non-control styles -- the 3 final-three styles
(`reports/tables/exemplar_images_final_three.csv`, `role` starting with `final_rank_`) + 20 newly
selected random guard-passing styles (`reports/tables/exemplar_images_margin_reference.csv`,
`role` starting with `margin_ref_`) -- plus the existing control pool (`role == "control"` rows of
`reports/tables/exemplar_images.csv`, 1 style, Menswear/Scarf/Accessories/Grey/Melange). Only
successfully-fetched, on-disk images are used (`fetch_success == True` and the file actually
exists, same defensive convention as `derive_similarity_band.group_by_style`).

TWO ANCHORS (task C2 step 4):

1. UPPER ANCHOR ("what a perfect copy looks like"): for every real image belonging to one of the
   23 non-control styles, compute its margin against ITS OWN style's OTHER reference images
   (leave-one-out -- the query image is excluded from its own reference set, otherwise the query
   would be trivially most-similar to itself) vs. the SAME control pool used everywhere else in
   this module. Styles with fewer than 2 successfully-fetched images contribute no leave-one-out
   pairs (nothing to leave in as a reference) and are skipped, same convention as
   `derive_similarity_band.within_style_similarities` skipping singleton styles.

2. LOWER ANCHOR (validates the margin construction itself): margin of every real image belonging
   to ONE randomly chosen style (`style_a`) against a DIFFERENT randomly chosen style's
   (`style_b`'s) real reference images -- NOT `style_a`'s own references -- still subtracting the
   SAME control pool as the second term. `style_a`/`style_b` are chosen by a single seeded
   `random.Random(42).sample` draw over the 23 non-control styles.

   AMBIGUITY RESOLUTION (documented per project convention -- the task brief's exact phrasing
   admits more than one reading, so the interpretation actually implemented is stated explicitly
   rather than left implicit): "not the control pool this time" is read as describing what's
   DIFFERENT about the lower anchor relative to a naive baseline check, not as dropping `margin()`'s
   two-term structure. Reading implemented: `margin(image_from_style_a, style_b_references,
   control_pool)` -- i.e. the "own style" slot in `margin()`'s signature is filled by a genuinely
   different, non-control style rather than the image's true style, while the control-pool
   subtraction stays IDENTICAL to every other margin computation in this module. This validates a
   specific, useful claim: an unrelated real style is no more "related" to `style_a`'s images than
   the dedicated control pool is -- i.e. the control pool isn't a special/degenerate choice that
   the whole construction secretly depends on. Expected result: margin ~= 0, since both terms are
   now "some style `style_a`'s images have no real relationship to."

TARGET BAND DERIVATION (task C2 step 4, JUDGMENT CALL -- documented thoroughly per task
instructions, same kind of reasoning B3's `LOWER_BOUND_PERCENTILE` choice used, now margin-based):
the band is expressed as `[LOWER_MARGIN_FRACTION, UPPER_MARGIN_FRACTION] * upper_anchor_mean`,
i.e. a FRACTION of the "perfect copy" margin, not an absolute cosine value (which would not be
comparable across CLIP's and DINOv2's differently-scaled embedding spaces -- see step 4's explicit
instruction not to average the two spaces into one number).

- `LOWER_MARGIN_FRACTION = 0.25`: a generated concept's margin must reach at least a QUARTER of
  the perfect-copy margin to count as recognizably on-style. Chosen conservatively-low (rather
  than e.g. 0.5) because the reference base, while larger than B3's 6 styles, is still modest (23
  non-control styles) -- see SMALL-SAMPLE CAVEAT below -- and a generous floor avoids
  over-rejecting genuinely on-style but stylistically freer generations. Cross-validated by the
  lower anchor: as long as the lower anchor's mean sits well below `0.25 * upper_anchor_mean`
  (ideally close to 0, per its own construction), a 25% floor cleanly separates "genuine
  relationship" from "no relationship" rather than being lost in that noise -- this is checked
  and reported numerically by `main()`, not assumed.
- `UPPER_MARGIN_FRACTION = 0.75`: a generated concept's margin must stay BELOW three-quarters of
  the perfect-copy margin to count as "novel" rather than near-duplicate. Symmetric distance from
  the 0.5 midpoint as the lower fraction, deliberately leaving the widest defensible "in-band"
  window (50% of the upper anchor's span) given the still-modest sample -- a tighter band would
  imply a false precision the current sample size doesn't support.

Both fractions are METHODOLOGY choices, fixed BEFORE looking at the actual anchor numbers computed
by this run (same "choose the method, then apply it" discipline as B3's 90th-vs-95th-percentile
choice) -- not reverse-engineered from a target answer.

SMALL-SAMPLE CAVEAT: 23 non-control styles (up to 8 images each, ~180 candidate images before
fetch failures) for the upper anchor, and a SINGLE (style_a, style_b) pair (up to 8 images) for the
lower anchor. Better than B3's 6 styles, but still a rough, revisable empirical estimate -- NOT a
large-sample-validated statistical threshold. The lower anchor in particular, with at most ~8
images, has a wide, unreported confidence interval around its mean; treat its value as a sanity
check on construction validity (is it plausibly ~0?), not as a precise estimate of the true
cross-style margin. Revisit once more styles/images are available.
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl

from nss.generate import clip_scoring, dino_scoring
from nss.generate.derive_similarity_band import group_by_style
from nss.generate.margin_scoring import margin

CONTROL_MANIFEST_PATH = Path("reports/tables/exemplar_images.csv")
FINAL_THREE_MANIFEST_PATH = Path("reports/tables/exemplar_images_final_three.csv")
MARGIN_REFERENCE_MANIFEST_PATH = Path("reports/tables/exemplar_images_margin_reference.csv")

OUT_PATHS: dict[str, Path] = {
    "clip": Path("reports/tables/margin_anchors_clip.csv"),
    "dinov2": Path("reports/tables/margin_anchors_dinov2.csv"),
}

EMBEDDERS: dict[str, Callable[[Path], np.ndarray]] = {
    "clip": clip_scoring.embed_image,
    "dinov2": dino_scoring.embed_image,
}

RANDOM_SEED = 42

# See module docstring TARGET BAND DERIVATION.
LOWER_MARGIN_FRACTION = 0.25
UPPER_MARGIN_FRACTION = 0.75

_MANIFEST_COLS = ["role", "style_key", "article_id", "local_image_path", "fetch_success"]


def load_manifest(path: Path) -> pl.DataFrame:
    """Load one exemplar manifest CSV, keeping only successfully-fetched images, de-duplicated.

    Args:
        path: Manifest CSV path (`exemplar_images*.csv` schema -- must have `role`, `style_key`,
            `article_id`, `local_image_path`, `fetch_success` columns).

    Returns:
        De-duplicated (by `article_id`) frame of `fetch_success == True` rows, sorted by
        `style_key`, `article_id` for determinism.
    """
    df = pl.read_csv(path).select(_MANIFEST_COLS)
    return (
        df.filter(pl.col("fetch_success"))
        .unique(subset=["article_id"], keep="first")
        .sort(["style_key", "article_id"])
    )


def load_control_pool(control_manifest_path: Path = CONTROL_MANIFEST_PATH) -> list[Path]:
    """The existing control pool's on-disk image paths (`role == "control"` rows only).

    Args:
        control_manifest_path: Path to `exemplar_images.csv` (or an equivalent manifest carrying
            a `control`-role style).

    Returns:
        Sorted list of existing image `Path`s for the control style.

    Raises:
        ValueError: if no `control`-role rows with an existing, successfully-fetched image remain.
    """
    manifest = load_manifest(control_manifest_path)
    control_rows = manifest.filter(pl.col("role") == "control")
    grouped = group_by_style(control_rows)
    if not grouped:
        raise ValueError(f"no control-role images found in {control_manifest_path}")
    ((_, paths),) = grouped.items()
    return paths


def load_non_control_styles(
    final_three_path: Path = FINAL_THREE_MANIFEST_PATH,
    margin_reference_path: Path = MARGIN_REFERENCE_MANIFEST_PATH,
) -> dict[str, list[Path]]:
    """The 23 non-control test styles' on-disk image paths, grouped by `style_key`.

    Combines the 3 `final_rank_*` styles and the 20 `margin_ref_*` styles, de-duplicated by
    `article_id` across both manifests (defensive -- no overlap is expected by construction, since
    `select_margin_reference_styles.py` excludes the final-three style_keys, but a cheap safety
    net regardless).

    Args:
        final_three_path: Path to `exemplar_images_final_three.csv`.
        margin_reference_path: Path to `exemplar_images_margin_reference.csv`.

    Returns:
        Mapping of `style_key -> sorted list of image Path`s, for styles with >= 1 fetched image.
    """
    final_three = load_manifest(final_three_path).filter(
        pl.col("role").str.starts_with("final_rank_")
    )
    margin_reference = load_manifest(margin_reference_path).filter(
        pl.col("role").str.starts_with("margin_ref_")
    )
    combined = pl.concat([final_three, margin_reference]).unique(
        subset=["article_id"], keep="first"
    )
    return group_by_style(combined)


def embed_all(paths: list[Path], embed_fn: Callable[[Path], np.ndarray]) -> dict[Path, np.ndarray]:
    """Compute (and cache) an embedding for every path, once each, using the given embed function.

    Generic over embedding space (unlike `derive_similarity_band.embed_all`, which hardcodes
    `clip_scoring.embed_image`) so the same function serves both CLIP and DINOv2.

    Args:
        paths: Image paths to embed (duplicates embedded only once).
        embed_fn: `clip_scoring.embed_image` or `dino_scoring.embed_image` (or any
            `Path -> np.ndarray` embedder).

    Returns:
        Mapping of image `Path -> L2-normalized embedding`.
    """
    embeddings: dict[Path, np.ndarray] = {}
    for path in paths:
        if path not in embeddings:
            embeddings[path] = embed_fn(path)
    return embeddings


def upper_anchor_margins(
    style_to_images: dict[str, list[Path]],
    embeddings: dict[Path, np.ndarray],
    control_pool_embeddings: list[np.ndarray],
) -> list[float]:
    """Leave-one-out own-style margin for every image across every style with >= 2 images.

    See module docstring UPPER ANCHOR.

    Args:
        style_to_images: The 23 non-control test styles (output of `load_non_control_styles`).
        embeddings: Output of `embed_all` for (at least) every path in `style_to_images`.
        control_pool_embeddings: Embeddings for the control pool's images.

    Returns:
        Flat list of margins, one per qualifying image (styles with < 2 images contribute none).
    """
    margins: list[float] = []
    for paths in style_to_images.values():
        if len(paths) < 2:
            continue
        for i, query_path in enumerate(paths):
            references = [embeddings[p] for j, p in enumerate(paths) if j != i]
            margins.append(margin(embeddings[query_path], references, control_pool_embeddings))
    return margins


def lower_anchor_margins(
    style_to_images: dict[str, list[Path]],
    embeddings: dict[Path, np.ndarray],
    control_pool_embeddings: list[np.ndarray],
    seed: int = RANDOM_SEED,
) -> tuple[list[float], str, str]:
    """Margin of one random style's images against a DIFFERENT random style's references.

    See module docstring LOWER ANCHOR / AMBIGUITY RESOLUTION.

    Args:
        style_to_images: The 23 non-control test styles.
        embeddings: Output of `embed_all` for (at least) every path in `style_to_images`.
        control_pool_embeddings: Embeddings for the control pool's images (the subtracted term is
            unchanged from every other margin computation in this module).
        seed: Random seed for the single `(style_a, style_b)` draw.

    Returns:
        `(margins, style_a, style_b)` -- one margin per `style_a` image, plus which two styles
        were drawn (for provenance/reporting).

    Raises:
        ValueError: if fewer than 2 styles are available to draw from.
    """
    styles = sorted(style_to_images)
    if len(styles) < 2:
        raise ValueError("need at least 2 non-control styles to compute the lower anchor")
    rng = random.Random(seed)
    style_a, style_b = rng.sample(styles, 2)
    style_b_references = [embeddings[p] for p in style_to_images[style_b]]
    margins = [
        margin(embeddings[p], style_b_references, control_pool_embeddings)
        for p in style_to_images[style_a]
    ]
    return margins, style_a, style_b


def summarize(distribution: list[float]) -> dict[str, float]:
    """Summary statistics for a flat list of margin values.

    Args:
        distribution: Non-empty list of margin values.

    Returns:
        Dict with `n`, `mean`, `median`, `std`.

    Raises:
        ValueError: if `distribution` is empty.
    """
    if not distribution:
        raise ValueError("distribution must be non-empty")
    arr = np.array(distribution, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
    }


def derive_target_band(
    upper_summary: dict[str, float],
    lower_fraction: float = LOWER_MARGIN_FRACTION,
    upper_fraction: float = UPPER_MARGIN_FRACTION,
) -> dict[str, float]:
    """Derive `[lower, upper]` margin thresholds as fractions of the upper anchor's mean.

    See module docstring TARGET BAND DERIVATION.

    Args:
        upper_summary: Output of `summarize(upper_anchor_margins(...))`.
        lower_fraction: Fraction of the upper-anchor mean used as the lower bound.
        upper_fraction: Fraction of the upper-anchor mean used as the upper bound.

    Returns:
        `{"lower": float, "upper": float, "lower_fraction": float, "upper_fraction": float}`.
    """
    upper_anchor_mean = upper_summary["mean"]
    return {
        "lower": lower_fraction * upper_anchor_mean,
        "upper": upper_fraction * upper_anchor_mean,
        "lower_fraction": lower_fraction,
        "upper_fraction": upper_fraction,
    }


def main() -> None:
    """CLI entry point: compute upper/lower margin anchors + derived band, for CLIP and DINOv2.

    Writes `reports/tables/margin_anchors_clip.csv` and `reports/tables/margin_anchors_dinov2.csv`,
    and prints a full report including per-style image counts, both anchors' distributions, the
    derived band, and an explicit CLIP-vs-DINOv2 agreement/disagreement statement.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-manifest-path", type=Path, default=CONTROL_MANIFEST_PATH)
    parser.add_argument("--final-three-path", type=Path, default=FINAL_THREE_MANIFEST_PATH)
    parser.add_argument(
        "--margin-reference-path", type=Path, default=MARGIN_REFERENCE_MANIFEST_PATH
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--lower-fraction", type=float, default=LOWER_MARGIN_FRACTION)
    parser.add_argument("--upper-fraction", type=float, default=UPPER_MARGIN_FRACTION)
    args = parser.parse_args()

    control_paths = load_control_pool(args.control_manifest_path)
    style_to_images = load_non_control_styles(args.final_three_path, args.margin_reference_path)

    n_non_control_images = sum(len(p) for p in style_to_images.values())
    print(f"Control pool: {len(control_paths)} images")
    print(
        f"Non-control test set: {len(style_to_images)} styles, {n_non_control_images} images "
        "(final-three + 20 margin-reference styles)"
    )
    for style_key, paths in sorted(style_to_images.items()):
        print(f"  {style_key}: {len(paths)} images")

    band_summaries: dict[str, dict[str, float]] = {}
    for space, embed_fn in EMBEDDERS.items():
        print(f"\n=== {space.upper()} ===")
        all_paths = control_paths + [p for paths in style_to_images.values() for p in paths]
        embeddings = embed_all(all_paths, embed_fn)
        print(f"Embedded {len(embeddings)} distinct images ({space}, CPU).")
        control_embeddings = [embeddings[p] for p in control_paths]

        upper_margins = upper_anchor_margins(style_to_images, embeddings, control_embeddings)
        upper_summary = summarize(upper_margins)
        print(f"Upper anchor (perfect-copy, leave-one-out own-style margin): {upper_summary}")

        lower_margins, style_a, style_b = lower_anchor_margins(
            style_to_images, embeddings, control_embeddings, seed=args.seed
        )
        lower_summary = summarize(lower_margins)
        print(
            f"Lower anchor (cross-style validation, style_a={style_a!r} vs "
            f"style_b={style_b!r}): {lower_summary}"
        )

        band = derive_target_band(upper_summary, args.lower_fraction, args.upper_fraction)
        print(
            f"Derived band: lower={band['lower']:.4f} "
            f"({band['lower_fraction']:.0%} of upper-anchor mean), "
            f"upper={band['upper']:.4f} ({band['upper_fraction']:.0%} of upper-anchor mean)"
        )

        band_summaries[space] = {
            **band,
            **{f"upper_anchor_{k}": v for k, v in upper_summary.items()},
        }
        band_summaries[space].update({f"lower_anchor_{k}": v for k, v in lower_summary.items()})

        out_path = OUT_PATHS[space]
        out_df = pl.DataFrame(
            [
                {
                    "row_type": "upper_anchor",
                    "n": upper_summary["n"],
                    "mean": upper_summary["mean"],
                    "median": upper_summary["median"],
                    "std": upper_summary["std"],
                    "lower_anchor_style_a": None,
                    "lower_anchor_style_b": None,
                    "band_lower": band["lower"],
                    "band_upper": band["upper"],
                    "lower_fraction": band["lower_fraction"],
                    "upper_fraction": band["upper_fraction"],
                    "n_non_control_styles": len(style_to_images),
                    "n_control_images": len(control_paths),
                },
                {
                    "row_type": "lower_anchor",
                    "n": lower_summary["n"],
                    "mean": lower_summary["mean"],
                    "median": lower_summary["median"],
                    "std": lower_summary["std"],
                    "lower_anchor_style_a": style_a,
                    "lower_anchor_style_b": style_b,
                    "band_lower": band["lower"],
                    "band_upper": band["upper"],
                    "lower_fraction": band["lower_fraction"],
                    "upper_fraction": band["upper_fraction"],
                    "n_non_control_styles": len(style_to_images),
                    "n_control_images": len(control_paths),
                },
            ]
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.write_csv(out_path)
        print(f"Wrote {out_path}")

    print("\n=== CLIP vs DINOv2 agreement check ===")
    clip_lower_anchor_mean = band_summaries["clip"]["lower_anchor_mean"]
    dino_lower_anchor_mean = band_summaries["dinov2"]["lower_anchor_mean"]
    clip_upper_anchor_mean = band_summaries["clip"]["upper_anchor_mean"]
    dino_upper_anchor_mean = band_summaries["dinov2"]["upper_anchor_mean"]
    print(
        f"CLIP:   lower_anchor_mean={clip_lower_anchor_mean:.4f}, "
        f"upper_anchor_mean={clip_upper_anchor_mean:.4f}, "
        f"band=[{band_summaries['clip']['lower']:.4f}, {band_summaries['clip']['upper']:.4f}]"
    )
    print(
        f"DINOv2: lower_anchor_mean={dino_lower_anchor_mean:.4f}, "
        f"upper_anchor_mean={dino_upper_anchor_mean:.4f}, "
        f"band=[{band_summaries['dinov2']['lower']:.4f}, {band_summaries['dinov2']['upper']:.4f}]"
    )
    print(
        "NOTE: the two spaces' absolute margin values are NOT comparable (different embedding "
        "geometries) -- do not average them into one band. See printed numbers above / the two "
        "output CSVs for each space's own independently-derived operating point."
    )


if __name__ == "__main__":
    main()
