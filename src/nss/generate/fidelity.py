"""Gate-2 fidelity with the non-visual attribute exclusions applied (task L2).

H2 dropped `garment_group` from the judge checklist because a merchandising term ("Jersey Basic")
has no visual referent. The same defect remains in `graphical_appearance_name`: a handful of its
values are internal catch-alls the judge cannot name from a picture (the Summer concept scored 0 on
"Other structure" while the judge correctly described the image as "solid, ribbed"). Those values
are excluded from the checklist -- an EXPLICIT list, never a judgement per image -- and fidelity is
always reported BOTH ways (all three attributes, and visual-only) so nothing is hidden.

This completes H2's existing fix; it introduces no new threshold (the per-judge threshold is the
calibrated `0.75 x positive mean`, unchanged). A genuine judge error is NOT excluded: the sweater's
"Melange" scored 0 although melange is visible, and stays in the score and in the limitations.

Usage:
    uv run python -m nss.generate.fidelity
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from nss.generate.vlm_judges import ATTRIBUTE_DIMENSIONS

GRAPHICAL_DIM = "graphical_treatment"
# Values of H&M's `graphical_appearance_name` that are NOT a describable visual pattern. Checked
# against every value in `articles.csv` (30 distinct): "Other structure"/"Other pattern"/"Unknown"
# are catch-alls; "Treatment" names a fabric finishing process, not something visible. Everything
# else (Solid, Stripe, Lace, Melange, Mesh, Contrast, ...) is a visual description and is KEPT.
NON_VISUAL_GRAPHICAL_VALUES: frozenset[str] = frozenset(
    {"Other structure", "Other pattern", "Unknown", "Treatment"}
)
OUT_PATH = Path("reports/tables/fidelity_both_figures.csv")


def applicable_dimensions(graphical_value: str) -> tuple[str, ...]:
    """Checklist dimensions to score for a style whose pattern label is `graphical_value`."""
    if graphical_value.strip() in NON_VISUAL_GRAPHICAL_VALUES:
        return tuple(d for d in ATTRIBUTE_DIMENSIONS if d != GRAPHICAL_DIM)
    return tuple(ATTRIBUTE_DIMENSIONS)


def fidelity_both(scores: Mapping[str, float], graphical_value: str) -> dict[str, Any]:
    """One reading's per-attribute scores -> fidelity with and without the excluded attributes."""
    dims = applicable_dimensions(graphical_value)
    return {
        "all_attributes": sum(scores[d] for d in ATTRIBUTE_DIMENSIONS) / len(ATTRIBUTE_DIMENSIONS),
        "visual_only": sum(scores[d] for d in dims) / len(dims),
        "excluded": [d for d in ATTRIBUTE_DIMENSIONS if d not in dims],
    }


def summarise_readings(
    readings: Sequence[Mapping[str, float]], graphical_value: str
) -> dict[str, Any]:
    """Median across readings of both fidelity figures."""
    per = [fidelity_both(r, graphical_value) for r in readings]
    return {
        "n_readings": len(per),
        "median_all_attributes": statistics.median(p["all_attributes"] for p in per),
        "median_visual_only": statistics.median(p["visual_only"] for p in per),
        "excluded": per[0]["excluded"] if per else [],
    }


def _reading_scores(path: Path, image_key: str) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if (
            rec.get("judge", "groq") == "groq"
            and rec["available"]
            and rec["scores"]
            and rec["image_path"] == image_key
        ):
            out.append({d: rec["scores"][d] for d in ATTRIBUTE_DIMENSIONS})
    return out


def build_report() -> pl.DataFrame:
    """Both fidelity figures for the four final concepts, from the PERSISTED per-attribute scores.

    No judge is called: `judge_repeat_j4.jsonl` (three AW2020 finals) and
    `judge_repeat_summer.jsonl` (Summer) store every reading's per-attribute scores.
    """
    from nss.generate import judge_rescore
    from nss.generate.concept_qc_pipeline import parse_style_attributes
    from nss.generate.final_selection_figures import SELECTED, _image
    from nss.generate.seasonal_concept import SELECTED_SEED, candidate_path

    threshold = judge_rescore.recompute_calibration()[2]["groq"]
    j4 = Path("data/generated/judge_repeat_j4.jsonl")
    summer_log = Path("data/generated/judge_repeat_summer.jsonl")
    summer_style = pl.read_csv("reports/tables/seasonal_summer_screened.csv")["style_id"][0]
    items = [
        (sid, "AW2020", str(_image(sid, seed, src)), j4) for sid, (seed, src) in SELECTED.items()
    ]
    items.append((summer_style, "Summer", str(candidate_path(SELECTED_SEED)), summer_log))
    rows = []
    for style_id, season, image_key, log in items:
        graphical = parse_style_attributes(style_id)[GRAPHICAL_DIM]
        readings = _reading_scores(log, image_key)
        s = summarise_readings(readings, graphical)
        rows.append(
            {
                "style_id": style_id,
                "season": season,
                "pattern_label": graphical,
                "n_readings": s["n_readings"],
                "excluded_attributes": ",".join(s["excluded"]) or None,
                "fidelity_all_attributes": s["median_all_attributes"],
                "fidelity_visual_only": s["median_visual_only"],
                "threshold": threshold,
                "pass_all_attributes": s["median_all_attributes"] >= threshold,
                "pass_visual_only": s["median_visual_only"] >= threshold,
                "readings_json": json.dumps(readings),
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    """Write and print the both-figures table."""
    df = build_report()
    df.write_csv(OUT_PATH)
    with pl.Config(tbl_cols=-1, tbl_width_chars=220, fmt_str_lengths=28):
        print(df.drop("readings_json").with_columns(pl.col(pl.Float64).round(3)))


if __name__ == "__main__":
    main()
