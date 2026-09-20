"""Closed loop for the final concepts: score each through the forecaster.

Each final concept image is embedded (CLIP ViT-L/14 + DINOv2), matched to the nearest catalogue
style by the mean-embedding retrieval of `concept_forecast_index`, and looked up in the frozen
model's full forecast table: the autumn/winter concepts against the forecast as of 2020-09-21, the
summer concept against the forecast as of 2020-06-01 (its own origin). The result is the concept's
ARCHETYPE forecast, not a demand forecast for the new design.

CAVEAT (circularity, stated not hidden): the index contains the screened reference photos of each
concept's intended style (they are also what the concept was generated from), so a concept mapping
to its intended style is expected by construction and is weak evidence of anything; the validation
on held-out real photos (`concept_forecast_validation`) is the evidence for the method. No photo is
excluded from the index here.

Output: `reports/tables/concept_forecast_retrieval.csv` (same columns as the earlier free-text
file `concept_forecast_final.csv`, which is left untouched, plus `top5`, `similarity`, `margin`,
`n_index_styles`, `n_index_images`).

Usage:
    python -m nss.generate.concept_forecast_final
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from nss.generate import concept_forecast as cf
from nss.generate import final_registry, final_selection_figures

OUT = Path("reports/tables/concept_forecast_retrieval.csv")
SUMMER_TABLE = Path("reports/tables/forecast_all_styles_summer.csv")
CONCEPTS_DIR = Path("reports/concepts")


def concept_images() -> dict[str, Path]:
    """The four committed final concept images, keyed by intended style id."""
    # same naming rule as `final_selection_figures.concept_filename` (which only covers the three
    # autumn/winter styles), extended to the summer concept
    return {
        style_id: CONCEPTS_DIR
        / f"{final_registry.PLAIN_NAMES[style_id].lower().replace(' ', '-')}_{path.stem}.png"
        for style_id, path in final_selection_figures.ALL_SELECTED.items()
    }


def main() -> None:
    """Forecast every final concept by retrieval and write the table."""
    summer_table = pl.read_csv(SUMMER_TABLE)
    rows = []
    for style_id, image in concept_images().items():
        is_summer = style_id == final_registry.SUMMER
        table = summer_table if is_summer else cf.load_table()
        index = cf.default_index(table)
        result = cf.forecast_concept(image, table, index)
        rows.append(
            {
                "style_id": style_id,
                "image_path": str(image),
                "forecast_origin": "2020-06-01" if is_summer else "2020-09-21",
                "mapped_style_key": result.style_key,
                "maps_to_intended_style": result.style_key == style_id,
                "forecast": result.forecast,
                "rank": result.rank,
                "n_styles": result.n_styles,
                "match_level": result.match_level,
                "confidence": result.confidence,
                "judges": ",".join(sorted(result.normalised)),
                "extraction": str(result.normalised),
                "top5": " | ".join(
                    f"{t['style_key']} (sim {t['similarity']:.3f}, rank {t['rank']})"
                    for t in result.top5
                ),
                "similarity": result.similarity,
                "margin": result.margin,
                "n_index_styles": len(index.styles),
                "n_index_images": sum(index.n_images.values()),
            }
        )
        print(result.sentence())
    pl.DataFrame(rows).write_csv(OUT)


if __name__ == "__main__":
    main()
