"""Pattern-conformity screen for the underwear style's reference set (task H3).

DEFECT: the style's `graphical_appearance_name` is "Solid" but that is a colour-family tag, not a
fabric-texture one -- most articles carrying it are lace-constructed, and the F3 screen checked
FRAMING only, so IP-Adapter reproduced lace regardless of the negative prompt.

FIX: select only articles whose `detail_desc` contains no pattern/texture vocabulary (lace, mesh,
floral, embroidery, print, ...). A "lace trim" article is excluded too: strict screen, because a
lace trim at the top edge is exactly the texture IP-Adapter copies. The screen is on the
catalogue's own text; the survivors are then visually inspected before use (see the H3 report).
Best-selling first (last-26-weeks units), because `backends.py` conditions on `references[0]` only.

Usage:
    uv run python -m nss.generate.h3_underwear_refs
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

from nss.data.fetch_images import fetch_images

STYLE_ID = "Ladieswear || Underwear bottom || Under-, Nightwear || Red || Solid"
ARTICLES_PATH = Path("data/raw/articles.csv")
TRANSACTIONS_PATH = Path("data/interim/transactions_train_parquet")
IMAGES_DIR = Path("data/images")
MANIFEST_PATH = Path("reports/tables/underwear_solid_references.csv")
MIN_SOLID_REFERENCES = 3
# Pattern/texture vocabulary in `detail_desc` that disqualifies an article as a plain-solid ref.
PATTERN_RE = re.compile(
    r"lace|floral|flower|embroider|mesh|print|pattern|stripe|dot|jacquard|sheer|glitter|sequin",
    re.IGNORECASE,
)


def is_plain_description(detail_desc: str | None) -> bool:
    """True iff the description carries no pattern/texture vocabulary (missing text -> False)."""
    return detail_desc is not None and PATTERN_RE.search(detail_desc) is None


def main() -> None:
    """Screen the style's articles, fetch the survivors' images, write the manifest."""
    dept, ptype, group, colour, pattern = (p.strip() for p in STYLE_ID.split("||"))
    arts = pl.read_csv(ARTICLES_PATH).filter(
        (pl.col("index_group_name") == dept)
        & (pl.col("product_type_name") == ptype)
        & (pl.col("garment_group_name") == group)
        & (pl.col("perceived_colour_master_name") == colour)
        & (pl.col("graphical_appearance_name") == pattern)
    )
    plain = arts.filter(
        pl.col("detail_desc").map_elements(is_plain_description, return_dtype=pl.Boolean)
    )
    print(f"{arts.height} articles in style; {plain.height} pass the plain-description screen")
    ids = plain["article_id"].to_list()

    tx = pl.scan_parquet(str(TRANSACTIONS_PATH / "**" / "*.parquet"))
    units = (
        tx.filter(pl.col("article_id").is_in(ids))
        .group_by("article_id")
        .agg(pl.len().alias("units_all_time"))
        .collect()
    )
    fetched = fetch_images(ids, IMAGES_DIR)
    out = (
        plain.select("article_id", "prod_name", "detail_desc")
        .join(units, on="article_id", how="left")
        .with_columns(
            pl.col("units_all_time").fill_null(0),
            pl.col("article_id")
            .map_elements(lambda i: fetched[str(i)], return_dtype=pl.Boolean)
            .alias("fetch_success"),
            pl.col("article_id")
            .map_elements(lambda i: str(IMAGES_DIR / f"{i:010d}.jpg"), return_dtype=pl.String)
            .alias("image_path"),
        )
        .sort("units_all_time", descending=True)
    )
    out.write_csv(MANIFEST_PATH)
    print(out.select("article_id", "prod_name", "units_all_time", "fetch_success"))
    if out.filter(pl.col("fetch_success")).height < MIN_SOLID_REFERENCES:
        print("FEWER THAN 3 solid references available -- catalogue fact, stop (task H3).")


if __name__ == "__main__":
    main()
