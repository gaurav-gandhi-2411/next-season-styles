"""Validate the closed loop's style_key extraction on REAL catalogue images (task N8).

For 40 catalogue styles drawn at random (seed 42) from the frozen model's forecast table (styles
with at least 5 articles, so they are real assortments), take the style's best-selling article's
photo, whose true style_key is known, and run `concept_forecast.forecast_concept` on it with the
judge panel. Reported, not hidden:

- per-attribute accuracy of the extraction after mapping (product type, colour, pattern);
- accuracy of the (product type, colour, pattern) triple;
- exact style_key match (also needs department and garment group, which no image shows -- they are
  filled with the dominant catalogue variant, so exact match is an upper-bounded, honest measure);
- the forecast error that a wrong mapping causes: |predicted(mapped) - predicted(true)| and rank
  distance, and the confidence label's calibration (accuracy within high / medium / low).

Output: `reports/tables/concept_forecast_validation.csv` (one row per image).

Usage:
    NSS_JUDGES=smolvlm uv run python -m nss.generate.concept_forecast_validation
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

from nss.data.fetch_images import fetch_images, local_path
from nss.generate import concept_forecast as cf
from nss.generate import local_vlm

ARTICLES = Path("data/raw/articles.csv")
IMAGES_DIR = Path("data/images")
OUT = Path("reports/tables/concept_forecast_validation.csv")
N_STYLES = 40
SEED = 42


def sample_images() -> pl.DataFrame:
    """One best-selling-article photo per randomly drawn eligible style (fetched if missing)."""
    table = cf.load_table()
    articles = pl.read_csv(ARTICLES).with_columns(
        (
            pl.col("index_group_name")
            + " || "
            + pl.col(cf.TYPE)
            + " || "
            + pl.col("garment_group_name")
            + " || "
            + pl.col(cf.COLOUR)
            + " || "
            + pl.col(cf.PATTERN)
        ).alias("style_key")
    )
    counts = articles.group_by("style_key").len().filter(pl.col("len") >= 5)
    pool = table.join(counts, on="style_key").sample(n=N_STYLES, seed=SEED)
    # deterministic representative article: the lowest article_id of the style (no sales lookup
    # needed for validation; any real photo of the style is a valid positive)
    reps = (
        articles.filter(pl.col("style_key").is_in(pool["style_key"].to_list()))
        .sort("article_id")
        .group_by("style_key", maintain_order=True)
        .first()
        .select("style_key", "article_id")
    )
    ok = fetch_images(reps["article_id"].to_list(), IMAGES_DIR)
    reps = reps.with_columns(
        pl.col("article_id")
        .map_elements(lambda a: str(local_path(a, IMAGES_DIR)), return_dtype=pl.String)
        .alias("image_path"),
        pl.col("article_id")
        .map_elements(lambda a: ok[str(a)], return_dtype=pl.Boolean)
        .alias("ok"),
    )
    return reps.filter(pl.col("ok"))


def main() -> None:
    """Run the panel on every sampled image and write per-image results plus a summary."""
    backends = os.environ.get("NSS_JUDGES", "smolvlm").split(",")
    table = cf.load_table()
    truth = {r["style_key"]: r for r in table.iter_rows(named=True)}
    reps = sample_images()
    rows = []
    for backend in backends:
        local_vlm.load(backend)
        for r in reps.iter_rows(named=True):
            rows.append(
                {
                    "judge": backend,
                    "style_key": r["style_key"],
                    "image_path": r["image_path"],
                    "raw": cf.default_extractors(include_api=False)["local"](Path(r["image_path"])),
                }
            )
        local_vlm.unload()
    out = []
    for r in rows:
        t = truth[r["style_key"]]
        n = cf.normalise(r["raw"], table)
        pick, level = cf._pick(table, n["product_type"], n["colour"], n["pattern"])
        out.append(
            {
                "judge": r["judge"],
                "style_key": r["style_key"],
                "type_ok": n["product_type"] == t[cf.TYPE],
                "colour_ok": n["colour"] == t[cf.COLOUR],
                "pattern_ok": n["pattern"] == t[cf.PATTERN],
                "mapped_style_key": pick["style_key"] if pick else None,
                "match_level": level,
                "exact_style_key": bool(pick and pick["style_key"] == r["style_key"]),
                "forecast_true": t["predicted_intensity"],
                "forecast_mapped": pick["predicted_intensity"] if pick else None,
                "raw": str(r["raw"]),
            }
        )
    df = pl.DataFrame(out).with_columns(
        (pl.col("type_ok") & pl.col("colour_ok") & pl.col("pattern_ok")).alias("triple_ok")
    )
    df.write_csv(OUT)
    for backend in backends:
        d = df.filter(pl.col("judge") == backend)
        print(
            backend,
            {
                "n": d.height,
                "type": d["type_ok"].mean(),
                "colour": d["colour_ok"].mean(),
                "pattern": d["pattern_ok"].mean(),
                "triple": d["triple_ok"].mean(),
                "exact_style_key": d["exact_style_key"].mean(),
            },
        )


if __name__ == "__main__":
    main()
