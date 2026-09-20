"""Record a human's reference screening in the screened manifest (task M2).

`screen_references` classifies framing with VLM judges; when both are quota-blocked it fails closed
(everything excluded). This module records the SAME decision made by eye, in the same manifest
schema, with the judges marked unavailable and the reason stated, so it is auditable and
`load_screened_references` consumes it unchanged (full-garment survivors, best-selling first).

Criteria applied to each candidate photo: (1) FRAMING: a full-garment product shot, not a
texture crop or detail shot; (2) PATTERN/COLOUR CONFORMITY: the photo shows the style's own pattern
label (here plain solid) and a colour a viewer would call the style's colour master (a burgundy
photo of a "Red" style is excluded).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

SCREENED_PATH = Path("reports/tables/exemplar_images_screened.csv")
NOTE_PREFIX = "manual visual screening (task M2; VLM judges quota-blocked): "


def append_manual_screening(
    style_id: str,
    manifest_path: Path,
    verdicts: dict[int, tuple[bool, str]],
    path: Path = SCREENED_PATH,
) -> pl.DataFrame:
    """Append/replace `style_id`'s rows in the screened manifest from `verdicts`.

    Args:
        style_id: The style key.
        manifest_path: `exemplar_images_final_three.csv` (source of the candidate photos).
        verdicts: `{article_id: (keep, reason)}` for every fetched candidate.
        path: The screened manifest to update.
    """
    cand = pl.read_csv(manifest_path).filter(
        (pl.col("style_key") == style_id) & pl.col("fetch_success")
    )
    missing = set(cand["article_id"].to_list()) - set(verdicts)
    if missing:
        raise ValueError(f"no verdict for candidate articles {sorted(missing)}")
    rows = [
        {
            "style_id": style_id,
            "article_id": r["article_id"],
            "image_path": r["local_image_path"],
            "units_sold_last_26w": r["units_sold_last_26w"],
            "newly_fetched": True,
            "gemini_available": False,
            "gemini_label": None,
            "gemini_excluded_reason": "quota-blocked",
            "groq_available": False,
            "groq_label": None,
            "groq_excluded_reason": "quota-blocked",
            "is_full_garment": verdicts[r["article_id"]][0],
            "screening_note": NOTE_PREFIX + verdicts[r["article_id"]][1],
        }
        for r in cand.to_dicts()
    ]
    new = pl.DataFrame(rows)
    existing = pl.read_csv(path)
    existing = existing.filter(pl.col("style_id") != style_id)
    out = pl.concat([existing, new.select(existing.columns)], how="vertical_relaxed")
    out.write_csv(path)
    return out
