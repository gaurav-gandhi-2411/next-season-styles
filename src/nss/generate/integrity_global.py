"""Globally calibrated integrity floor (task P2).

DEFECT in the first floor (`gate3.integrity_floor`): garment coherence is a GLOBAL property ("is
this a real, well-formed garment?"), but the floor was calibrated WITHIN the concept's own style:
the 10th percentile of that style's real nearest-sibling similarity. For a style whose real articles
are near-identical (19 white jersey tops, p10 = 0.922) any genuine design change falls below it, so
the check was a similarity gate mislabelled as an integrity check.

FIX: pool the real nearest-sibling DINOv2 similarities of every style with at least
`MIN_STYLE_IMAGES` real photos on disk (all styles, not the concept's own), and take the 10th
percentile of THAT pool as one global floor. A concept passes iff its closest real reference (in
its own style's reference set, as before) is at least that similar. The pool is built from real
catalogue photos only (`data/images`, keyed to styles through `articles.csv`) plus the screened
reference bases; no generated image ever enters it.

RESULT (`integrity_global_validation.csv`): the global floor is 0.7786 (p10 over 154 real photos in
the 8 styles with at least 3 photos on disk). Against the labelled cases it catches 2 of the 3
malformed underwear images (misses seed 44, sheer mesh panels, closest reference 0.811) and 4 of 5
known-bad; it passes 9 of 9 known-good (per-style floor: 8 of 9). Coherence is not a property
of one style, so THIS floor gates (`n9_score`, task R2) and the per-style floor
(`gate3.integrity_floor`) is reported beside it as advisory. The one case the global floor misses is
seed 44 (sheer mesh, closest reference 0.811); the per-style floor catches it. Neither floor rescues
the white top (closest reference 0.733 is below both), so no concept's verdict changes: R2 corrects
mechanism and framing only. The pool is small (8 styles), so a larger pool is untested, not refuted.

Usage:
    uv run python -m nss.generate.integrity_global
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from nss.generate import dino_scoring
from nss.generate.clip_scoring import _cosine

ARTICLES = Path("data/raw/articles.csv")
IMAGES = Path("data/images")
OUT_POOL = Path("reports/tables/integrity_global_pool.csv")
OUT_FLOOR = Path("reports/tables/integrity_global_floor.csv")
MIN_STYLE_IMAGES = 3
PERCENTILE = 10
STYLE_COLS = (
    "index_group_name",
    "product_type_name",
    "garment_group_name",
    "perceived_colour_master_name",
    "graphical_appearance_name",
)


def real_photos_by_style() -> dict[str, list[Path]]:
    """Real catalogue photos on disk grouped by the style of their article."""
    arts = pl.read_csv(ARTICLES).with_columns(
        pl.concat_str(list(STYLE_COLS), separator=" || ").alias("style_key")
    )
    style_of = dict(zip(arts["article_id"].to_list(), arts["style_key"].to_list(), strict=True))
    out: dict[str, list[Path]] = {}
    for path in sorted(IMAGES.glob("*.jpg")):
        style = style_of.get(int(path.stem))
        if style:
            out.setdefault(style, []).append(path)
    return {k: v for k, v in out.items() if len(v) >= MIN_STYLE_IMAGES}


def nearest_sibling_pool(groups: dict[str, list[Path]]) -> pl.DataFrame:
    """Each real photo's nearest same-style sibling similarity (DINOv2), across all styles."""
    rows = []
    for style, paths in groups.items():
        emb = [dino_scoring.embed_image(p) for p in paths]
        for i, path in enumerate(paths):
            nn = max(_cosine(emb[i], emb[j]) for j in range(len(paths)) if j != i)
            rows.append({"style_id": style, "image_path": str(path), "nn_sibling_sim": float(nn)})
    return pl.DataFrame(rows)


def global_floor(pool: pl.DataFrame | None = None) -> float:
    """The global floor: the 10th percentile of the pooled real nearest-sibling similarities."""
    if pool is None:
        pool = pl.read_csv(OUT_POOL)
    return float(np.percentile(pool["nn_sibling_sim"].to_numpy(), PERCENTILE))


def main() -> None:
    """Build the pool, write it and the floor."""
    groups = real_photos_by_style()
    pool = nearest_sibling_pool(groups)
    pool.write_csv(OUT_POOL)
    floor = global_floor(pool)
    per_style = pool.group_by("style_id").agg(
        pl.len().alias("n_images"), pl.col("nn_sibling_sim").quantile(0.1).alias("style_p10")
    )
    pl.DataFrame(
        [
            {
                "global_floor_p10": floor,
                "n_styles": len(groups),
                "n_images": pool.height,
                "per_style_p10_min": float(per_style["style_p10"].min()),
                "per_style_p10_max": float(per_style["style_p10"].max()),
            }
        ]
    ).write_csv(OUT_FLOOR)
    print(f"global floor (p10 over {pool.height} photos in {len(groups)} styles): {floor:.4f}")
    lo, hi = per_style["style_p10"].min(), per_style["style_p10"].max()
    print(f"per-style p10 range: {lo:.3f} .. {hi:.3f}")


if __name__ == "__main__":
    main()
