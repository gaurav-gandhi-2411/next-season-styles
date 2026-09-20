"""Closed loop for the final concepts (task N8): score each through the forecaster.

Each selected concept image is read by the local judge panel (blind attribute extraction), mapped to
the nearest catalogue style_key and looked up in the frozen model's full forecast table: the
autumn/winter concepts against the forecast as of 2020-09-21, the summer concept against the
forecast as of 2020-06-01 (its own origin). The result is the concept's ARCHETYPE forecast, not a
demand forecast for the new design.

Output: `reports/tables/concept_forecast_final.csv`.

Usage:
    NSS_JUDGES=smolvlm,florence2 uv run python -m nss.generate.concept_forecast_final
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

from nss.generate import concept_forecast as cf
from nss.generate import final_registry, local_vlm
from nss.generate.h4_deliverables import SELECTED, SUMMER_SELECTED

OUT = Path("reports/tables/concept_forecast_final.csv")
SUMMER_TABLE = Path("reports/tables/forecast_all_styles_summer.csv")


def raw_readings(backends: list[str], images: list[Path]) -> dict[str, dict[Path, dict[str, str]]]:
    """Blind extraction per (judge, image); each judge is loaded once."""
    out: dict[str, dict[Path, dict[str, str]]] = {}
    for backend in backends:
        local_vlm.load(backend)
        out[backend] = {p: local_vlm.extract_attributes_local(p) for p in images}
        local_vlm.unload()
    return out


def main() -> None:
    """Forecast every final concept through the panel and write the table."""
    backends = os.environ.get("NSS_JUDGES", "smolvlm,florence2").split(",")
    concepts = {**SELECTED, final_registry.SUMMER: SUMMER_SELECTED}
    raw = raw_readings(backends, list(concepts.values()))
    summer_table = pl.read_csv(SUMMER_TABLE)
    rows = []
    for style_id, image in concepts.items():
        table = summer_table if style_id == final_registry.SUMMER else cf.load_table()
        result = cf.forecast_concept(
            image, {b: (lambda p, b=b: raw[b][p]) for b in backends}, table
        )
        rows.append(
            {
                "style_id": style_id,
                "image_path": str(image),
                "forecast_origin": "2020-06-01"
                if style_id == final_registry.SUMMER
                else "2020-09-21",
                "mapped_style_key": result.style_key,
                "maps_to_intended_style": result.style_key == style_id,
                "forecast": result.forecast,
                "rank": result.rank,
                "n_styles": result.n_styles,
                "match_level": result.match_level,
                "confidence": result.confidence,
                "judges": ",".join(sorted(result.normalised)),
                "extraction": str(result.normalised),
            }
        )
        print(result.sentence())
    pl.DataFrame(rows).write_csv(OUT)


if __name__ == "__main__":
    main()
