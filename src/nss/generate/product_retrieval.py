"""L2: product type from image retrieval instead of free-text VLM reading.

The product type of a generated concept is that of the top-1 STYLE of `concept_forecast_index`
retrieval (CLIP and DINOv2, averaged cosine to the mean embedding of each style's real photos, the
headline view), over the union of the autumn and summer forecast tables. The check passes iff that
style's `product_type_name` equals the style's own product type, as an exact H&M string: no synonym
list is needed because both sides use the catalogue's vocabulary. Retrieval read product type at
82.5% on the 40 real photos, against 65% for free-text SmolVLM (`reports/SUBMISSION/WRITEUP.md`);
that figure is not re-measured here. Pre-registered in `reports/v3/PREREGISTRATION.md` (section L2).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import polars as pl

from nss.generate import concept_forecast as cf
from nss.generate import concept_forecast_index as cfi

SUMMER_TABLE = Path("reports/tables/forecast_all_styles_summer.csv")


@functools.lru_cache(maxsize=1)
def _index_and_styles() -> tuple[cfi.RetrievalIndex, list[str]]:
    keys = sorted(
        set(cf.load_table()["style_key"].to_list())
        | set(pl.read_csv(SUMMER_TABLE)["style_key"].to_list())
    )
    by_style = cfi.collect_index_images(keys, exclude_articles=(), max_per_style=cfi.PER_STYLE_CAP)
    return cfi.build_index(by_style), keys


def product_type_of(style_key: str) -> str:
    """The H&M product type of a `" || "`-joined style key."""
    return style_key.split(" || ")[1].strip()


def top1(image: Path | str) -> dict[str, Any]:
    """The retrieval top-1 style of an image and its similarity."""
    index, keys = _index_and_styles()
    query = cfi.embed_query(Path(image), index.embedders or None)
    result = cfi.retrieve(query, index, restrict=keys)
    style, sim = result.top(cfi.HEADLINE_VIEW, 1)[0]
    return {"style_key": style, "product_type": product_type_of(style), "similarity": float(sim)}


def check(image: Path | str, style_key: str) -> dict[str, Any]:
    """Retrieval product-type check of one concept against its style (exact string match)."""
    t = top1(image)
    want = product_type_of(style_key)
    return {
        "pass": t["product_type"] == want,
        "retrieved": t["product_type"],
        "expected": want,
        "top1_style": t["style_key"],
        "similarity": t["similarity"],
    }
