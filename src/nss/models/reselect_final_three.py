"""Final-three re-selection: all emerging, editorial exclusions, colour/type diversity.

WHY (history): an earlier selection removed an intimates style (red underwear) and forced colour
diversity, but still kept an "incumbent" slot. That slot mechanically returns whatever sells most
per product, which in fast fashion is a black basic: the buying plan a retailer already has, not a
design brief. This selection drops it. Both this and the earlier rules are SELECTION-RULE decisions,
not model decisions: the frozen model ranks what the market rewards; a human judges what belongs in
a design deliverable. The model surfaces, a human judges -- made explicit, deterministic and
auditable here.

RULE (fixed and documented BEFORE the output was inspected; applied in this order, no tuning):

Input: the committed frozen-model emerging leaderboard `top_styles_emerging.csv` (guard-passing,
ranked by growth ratio). No model is retrained or re-run. NOTE ON PROVENANCE: the top-10 of that
table had already been seen in the earlier selection; the rules below are stated in terms of
garment categories, not of the styles that happen to fall out.

1. ALL THREE from the emerging (growth) table. No incumbent slot.
2. ABSOLUTE-SCALE FLOOR: predicted intensity >= the median predicted intensity among guard-passing
   styles. Satisfied BY CONSTRUCTION for every row of the emerging table (`diversity_forecast.
   t2_absolute_intensity_floor` is exactly that median and gates emerging eligibility); the exact
   floor value is not persisted, so it is recorded as "by construction", never re-derived by
   retraining.
3. GUARDS unchanged (mean active articles >= 10, price index >= 0.85, active >= 26 of 52 weeks);
   already applied in the emerging table.
4. CATEGORY EXCLUSION (unchanged from the earlier selection): intimates/underwear/nightwear/
   lingerie product types
   (>= 50% of articles in product group {Underwear, Nightwear, Underwear/nightwear}) or H&M garment
   group "Under-, Nightwear". Swimwear TOPS and sets stay eligible.
5. VISUAL-AMBIGUITY EXCLUSION (new; the same editorial judgment as 4): drop garments that cannot be
   identified from a flat-lay without a label. Documented list (`VISUAL_AMBIGUOUS_TYPES`): swimwear
   bottoms (a flat-lay swim bottom is indistinguishable from briefs) and hosiery / leg base layers
   (leggings, tights, socks, leg warmers). Applies to the seasonal (summer) view too.
6. DIVERSITY: no two chosen styles share `perceived_colour_master_name`; no two share
   `product_type_name`. Walk the emerging list in growth-rank order and accept the first eligible
   style that collides with nothing already chosen, until three are chosen. A shortfall is reported,
   never backfilled.

Every candidate considered is written to `final_three_selection_log.csv` with its disposition and
reason, so what was skipped and why is auditable.

Usage:
    uv run python -m nss.models.reselect_final_three
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

T2_PATH = Path("reports/tables/top_styles_emerging.csv")
ARTICLES_PATH = Path("data/raw/articles.csv")
FINAL_OUT = Path("reports/tables/top_styles_final_three.csv")
LOG_OUT = Path("reports/tables/final_three_selection_log.csv")
INTIMATE_PRODUCT_GROUPS: tuple[str, ...] = ("Underwear", "Nightwear", "Underwear/nightwear")
INTIMATE_GARMENT_GROUPS: tuple[str, ...] = ("Under-, Nightwear",)
INTIMATE_SHARE_THRESHOLD = 0.5
N_FINAL = 3
# Product types not identifiable from a flat-lay without a label (rule 5 above). Chosen from the
# H&M data dictionary's swimwear and hosiery groups; swimwear TOPS/sets are deliberately absent.
VISUAL_AMBIGUOUS_TYPES: frozenset[str] = frozenset(
    {"Swimwear bottom", "Leggings/Tights", "Underwear Tights", "Socks", "Leg warmers"}
)
COLOUR = "perceived_colour_master_name"
TYPE = "product_type_name"


def intimate_product_types(articles: pl.DataFrame) -> frozenset[str]:
    """Product types that are intimates/underwear/nightwear, derived from the data dictionary."""
    share = articles.group_by(TYPE).agg(
        pl.col("product_group_name").is_in(list(INTIMATE_PRODUCT_GROUPS)).mean().alias("share")
    )
    return frozenset(share.filter(pl.col("share") >= INTIMATE_SHARE_THRESHOLD)[TYPE].to_list())


def exclusion_reason(row: dict, intimate_types: frozenset[str]) -> str | None:
    """Why a style is excluded by the category constraint, or `None`."""
    if row[TYPE] in intimate_types:
        return f"category exclusion: product type '{row[TYPE]}' is intimates/underwear/nightwear"
    if row["garment_group_name"] in INTIMATE_GARMENT_GROUPS:
        return f"category exclusion: garment group '{row['garment_group_name']}'"
    if row[TYPE] in VISUAL_AMBIGUOUS_TYPES:
        return (
            f"visual-ambiguity exclusion: '{row[TYPE]}' is not identifiable from a flat-lay "
            "without a label"
        )
    return None


def reselect(t2: pl.DataFrame, intimate_types: frozenset[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Apply the selection rule to the emerging table; returns `(final_three, selection_log)`."""
    log: list[dict] = []
    chosen: list[dict] = []

    def record(rank: int, row: dict, disposition: str, reason: str) -> None:
        log.append(
            {
                "source_table": "T2_emerging",
                "rank_in_table": rank,
                "style_key": row["style_key"],
                "predicted_intensity": row["predicted_intensity"],
                "growth_ratio": row.get("growth_ratio"),
                "disposition": disposition,
                "reason": reason,
            }
        )

    for rank, row in enumerate(t2.to_dicts(), start=1):
        if len(chosen) >= N_FINAL:
            record(rank, row, "not needed", "three styles already chosen")
            continue
        why = exclusion_reason(row, intimate_types)
        used_colours = {c[COLOUR] for c in chosen}
        used_types = {c[TYPE] for c in chosen}
        if why:
            record(rank, row, "skipped", why)
        elif row[COLOUR] in used_colours:
            record(rank, row, "skipped", f"colour collision: {row[COLOUR]} already used")
        elif row[TYPE] in used_types:
            record(rank, row, "skipped", f"product-type collision: {row[TYPE]} already used")
        else:
            chosen.append({**row, "source_table": "T2_emerging", "rank_in_source_table": rank})
            record(rank, row, "CHOSEN", "floor met by construction; no collision")
    if len(chosen) < N_FINAL:
        print(f"SHORTFALL: chose {len(chosen)}/{N_FINAL} (not backfilled)")
    return pl.DataFrame(chosen, infer_schema_length=None), pl.DataFrame(
        log, infer_schema_length=None
    )


def main() -> None:
    """Run the rule on the committed leaderboards and write the final three + the skip log."""
    t2 = pl.read_csv(T2_PATH)
    intimate = intimate_product_types(pl.read_csv(ARTICLES_PATH))
    print(f"excluded product types ({len(intimate)}): {sorted(intimate)}")
    final, log = reselect(t2, intimate)
    final.write_csv(FINAL_OUT)
    log.write_csv(LOG_OUT)
    with pl.Config(tbl_rows=60, tbl_width_chars=220, fmt_str_lengths=70, tbl_cols=-1):
        print(log.select("source_table", "rank_in_table", "style_key", "disposition", "reason"))
        print(
            final.select(
                "source_table",
                "style_key",
                "predicted_intensity",
                "growth_ratio",
                "guard1_pass",
                "guard2_pass",
                "guard3_pass",
                "shap_driver_1_feature",
                "shap_driver_2_feature",
                "shap_driver_3_feature",
            )
        )


if __name__ == "__main__":
    main()
