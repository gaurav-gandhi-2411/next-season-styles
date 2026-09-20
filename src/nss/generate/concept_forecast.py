"""Closed loop: score a generated concept through the same forecaster (task N8).

generated image -> blind VLM attribute extraction (product type, colour, pattern) -> map to the
nearest catalogue style_key -> look up the frozen model's forecast for that style.

    "maps to <style_key>; forecast N units/product/week; rank M of K; confidence <low|medium|high>"

WHY: predict -> generate -> score-the-generation-through-the-predictor closes the product loop with
components that already exist (VLM judges, the frozen forecaster's full ranking
`reports/tables/forecast_all_styles.csv`). It answers "would the style this image belongs to be
forecast to sell?" -- a statement about the ARCHETYPE the image reads as, not a demand forecast for
the new design (nothing here tests demand for the design itself).

MAPPING: the five style-key attributes are not all visible. `garment_group` ("Jersey Basic",
"Knitwear") is an internal merchandising label with no visual referent (task H2), and
`index_group` (department) is not in the picture either, so they are not asked of the judges; among
catalogue styles matching (product type, colour, pattern) the one with the most active articles
(the dominant variant) is taken. Matching backs off in order: exact -> pattern relaxed -> colour
relaxed -> product-type only; every back-off lowers confidence.

CONFIDENCE (from judge agreement, never from the forecast): `high` = at least two judges available
and all agree on (product type, colour, pattern) after normalisation, exact match; `medium` = judges
agree on product type and colour; otherwise `low`. The extraction error rate is measured on real
catalogue images with known style keys by `concept_forecast_validation` and reported, not hidden.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import polars as pl

FORECAST_TABLE = Path("reports/tables/forecast_all_styles.csv")
COLOUR = "perceived_colour_master_name"
TYPE = "product_type_name"
PATTERN = "graphical_appearance_name"

# Free-text words a judge may use -> H&M `perceived_colour_master_name` values.
COLOUR_SYNONYMS: dict[str, str] = {
    "gray": "Grey",
    "charcoal": "Grey",
    "silver": "Grey",
    "navy": "Blue",
    "denim": "Blue",
    "cream": "White",
    "ivory": "White",
    "off-white": "White",
    "offwhite": "White",
    "tan": "Beige",
    "camel": "Beige",
    "nude": "Beige",
    "taupe": "Beige",
    "sand": "Beige",
    "burgundy": "Red",
    "maroon": "Red",
    "coral": "Pink",
    "salmon": "Pink",
    "fuchsia": "Pink",
    "rose": "Pink",
    "olive": "Khaki green",
    "khaki": "Khaki green",
    "purple": "Lilac Purple",
    "lilac": "Lilac Purple",
    "lavender": "Lilac Purple",
    "violet": "Lilac Purple",
    "teal": "Turquoise",
    "mustard": "Yellow",
    "cognac": "Brown",
    "rust": "Orange",
}
PATTERN_SYNONYMS: dict[str, str] = {
    "plain": "Solid",
    "solid": "Solid",
    "striped": "Stripe",
    "stripes": "Stripe",
    "pinstripe": "Stripe",
    "checked": "Check",
    "plaid": "Check",
    "gingham": "Check",
    "floral": "All over pattern",
    "print": "All over pattern",
    "printed": "All over pattern",
    "patterned": "All over pattern",
    "heathered": "Melange",
    "marled": "Melange",
    "marl": "Melange",
    "speckled": "Melange",
    "colourblocking": "Colour blocking",
    "colorblock": "Colour blocking",
    "colorblocking": "Colour blocking",
    "color block": "Colour blocking",
    "polka dot": "Dot",
    "dotted": "Dot",
    "spotted": "Dot",
    "sheer": "Transparent",
    "see-through": "Transparent",
}
TYPE_SYNONYMS: dict[str, str] = {
    "jumper": "Sweater",
    "pullover": "Sweater",
    "knit": "Sweater",
    "knitwear": "Sweater",
    "tee": "T-shirt",
    "t shirt": "T-shirt",
    "tshirt": "T-shirt",
    "t-shirt": "T-shirt",
    "tank top": "Vest top",
    "camisole": "Vest top",
    "trouser": "Trousers",
    "pants": "Trousers",
    "jeans": "Trousers",
    "gown": "Dress",
    "shirt": "Shirt",
    "blouse": "Blouse",
    "top": "Top",
}

Extractor = Callable[[Path], dict[str, str]]


@dataclass(frozen=True)
class ConceptForecast:
    """The closed-loop result for one image."""

    style_key: str
    forecast: float
    rank: int
    n_styles: int
    match_level: str
    confidence: str
    normalised: dict[str, dict[str, str | None]]
    unavailable: dict[str, str] = field(default_factory=dict)

    def sentence(self) -> str:
        """Human-readable one-liner in the required format."""
        return (
            f"maps to {self.style_key}; forecast {self.forecast:.1f} units/product/week; "
            f"rank {self.rank} of {self.n_styles:,}; confidence {self.confidence}"
        )


@lru_cache(maxsize=1)
def load_table(path: Path = FORECAST_TABLE) -> pl.DataFrame:
    """The frozen model's forecast for every eligible style (see `models.forecast_all_styles`)."""
    return pl.read_csv(path)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9/ -]+", " ", text.lower()).strip()


def _map_to_vocab(text: str, vocab: Sequence[str], synonyms: Mapping[str, str]) -> str | None:
    """Free text -> one vocabulary value, or `None` if nothing plausible matches."""
    t = _norm(text)
    if not t:
        return None
    lookup = {v.lower(): v for v in vocab}
    hits = [v for low, v in lookup.items() if re.search(rf"\b{re.escape(low)}\b", t)]
    hits += [
        canon
        for word, canon in synonyms.items()
        if re.search(rf"\b{re.escape(word)}\b", t) and canon in vocab
    ]
    if hits:
        return max(hits, key=len)
    close = difflib.get_close_matches(t, list(lookup), n=1, cutoff=0.75)
    return lookup[close[0]] if close else None


def normalise(extraction: Mapping[str, str], table: pl.DataFrame) -> dict[str, str | None]:
    """One judge's raw extraction -> `{product_type, colour, pattern}` in catalogue vocabulary."""
    return {
        "product_type": _map_to_vocab(
            extraction.get("product_type", ""), table[TYPE].unique().to_list(), TYPE_SYNONYMS
        ),
        "colour": _map_to_vocab(
            extraction.get("colour_family", ""), table[COLOUR].unique().to_list(), COLOUR_SYNONYMS
        ),
        "pattern": _map_to_vocab(
            extraction.get("graphical_treatment", ""),
            table[PATTERN].unique().to_list(),
            PATTERN_SYNONYMS,
        ),
    }


def _pick(table: pl.DataFrame, product: str | None, colour: str | None, pattern: str | None):
    """Best-supported catalogue style for the (partial) attributes, plus the match level."""
    steps = (
        ("exact", {TYPE: product, COLOUR: colour, PATTERN: pattern}),
        ("pattern_relaxed", {TYPE: product, COLOUR: colour}),
        ("colour_relaxed", {TYPE: product, PATTERN: pattern}),
        ("type_only", {TYPE: product}),
    )
    for level, conds in steps:
        if any(v is None for v in conds.values()):
            continue
        sub = table.filter(pl.all_horizontal([pl.col(c) == v for c, v in conds.items()]))
        if not sub.is_empty():
            return sub.sort("guard1_n_active_articles_trailing_mean", descending=True).row(
                0, named=True
            ), level
    return None, "no_match"


def _consensus(values: Sequence[str | None]) -> str | None:
    """Majority value among judges (ties -> first judge listed); `None` if nobody produced one."""
    votes = [v for v in values if v]
    return max(dict.fromkeys(votes), key=votes.count) if votes else None


def forecast_concept(
    image: Path,
    extractors: Mapping[str, Extractor],
    table: pl.DataFrame | None = None,
) -> ConceptForecast:
    """Run every available judge on `image`, map to a style_key, look up the forecast.

    Raises `ValueError` if no catalogue style can be matched at all (never fabricates a forecast).
    """
    table = load_table() if table is None else table
    normalised: dict[str, dict[str, str | None]] = {}
    unavailable: dict[str, str] = {}
    for name, fn in extractors.items():
        try:
            normalised[name] = normalise(fn(image), table)
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise many types;
            # an unavailable judge is recorded and lowers confidence, never a silent pass
            unavailable[name] = f"{type(exc).__name__}: {str(exc)[:120]}"
    if not normalised:
        raise RuntimeError(f"no judge produced an extraction: {unavailable}")
    consensus = {
        dim: _consensus([n[dim] for n in normalised.values()])
        for dim in ("product_type", "colour", "pattern")
    }
    row, level = _pick(table, consensus["product_type"], consensus["colour"], consensus["pattern"])
    if row is None:
        raise ValueError(f"no catalogue style matches extraction {consensus} for {image}")
    rank = int(table.filter(pl.col("predicted_intensity") > row["predicted_intensity"]).height + 1)
    agree_all = len(normalised) >= 2 and all(
        len({n[d] for n in normalised.values()}) == 1 for d in ("product_type", "colour", "pattern")
    )
    agree_core = len(normalised) >= 2 and all(
        len({n[d] for n in normalised.values()}) == 1 for d in ("product_type", "colour")
    )
    confidence = "high" if agree_all and level == "exact" else "medium" if agree_core else "low"
    return ConceptForecast(
        style_key=row["style_key"],
        forecast=float(row["predicted_intensity"]),
        rank=rank,
        n_styles=table.height,
        match_level=level,
        confidence=confidence,
        normalised=normalised,
        unavailable=unavailable,
    )


def default_extractors(include_api: bool = True) -> dict[str, Extractor]:
    """The judge panel: the local model always, Groq and Gemini when `include_api`."""
    from nss.generate import local_vlm, vlm_judges

    panel: dict[str, Extractor] = {"local": local_vlm.extract_attributes_local}
    if include_api:
        panel["groq"] = vlm_judges.extract_attributes_groq
        panel["gemini"] = vlm_judges.extract_attributes_gemini
    return panel
