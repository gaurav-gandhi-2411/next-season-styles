"""Final-three re-selection with an editorial category constraint and colour diversity (task M2).

WHY: the earlier rule (T1 rank 1 + T2 ranks 1-2, one-per-(product_type, colour)) surfaced an
intimates style (a red underwear bottom) and a plain black T-shirt that adds nothing over its
category. Both are SELECTION-RULE problems, not model problems: the frozen model ranked what the
market rewards; a merchandising judgment decides what belongs in this deliverable. The model
surfaces, a human judges -- this is that human judgment made explicit, deterministic and auditable.

RULE (fixed and documented BEFORE the output was inspected; applied in this order, no tuning):

Inputs: the committed frozen-model leaderboards (`top_styles_t1_incumbent.csv`: guard-passing,
ranked by predicted intensity; `top_styles_t2_emerging.csv`: guard-passing, ranked by growth ratio
above the median-intensity floor). All existing guards are already applied there (mean active
articles >= 10, price index >= 0.85, active >= 26 of the last 52 weeks) and so is the existing
(product_type, colour) one-per-list diversity. No model is retrained or re-run.

1. CATEGORY EXCLUSION (editorial, human-applied on top of model output): drop any style whose
   product type is intimates/underwear/nightwear/lingerie. Defined from the data dictionary, not by
   eye: a product type is excluded if >= 50% of its articles sit in `product_group_name` in
   {Underwear, Nightwear, Underwear/nightwear}, or the style's `garment_group_name` is H&M's own
   "Under-, Nightwear". Hosiery (tights, leggings, socks: garment group "Socks and Tights") and
   swimwear are NOT excluded (legitimate categories, rendered flat-lay).
2. INCUMBENT SLOT (at most one): the highest-ranked remaining T1 style.
3. EMERGING SLOTS (at least two): walk the remaining T2 list in rank order and accept a style iff
   (a) its `perceived_colour_master_name` is not already used by a chosen style, (b) its
   (product_type, colour) pair is not already used, (c) it is not already chosen. Accept until two.
4. COMPOSITION: 1 incumbent + 2 emerging. Never relaxed silently: if the lists are exhausted before
   the slots fill, the shortfall is reported, not backfilled.

Every candidate considered is written to `final_three_selection_log.csv` with its disposition and
reason (excluded category / colour collision / pair collision / chosen), so what was skipped and
why is auditable.

Usage:
    uv run python -m nss.models.reselect_final_three
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

T1_PATH = Path("reports/tables/top_styles_t1_incumbent.csv")
T2_PATH = Path("reports/tables/top_styles_t2_emerging.csv")
ARTICLES_PATH = Path("data/raw/articles.csv")
FINAL_OUT = Path("reports/tables/top_styles_final_three.csv")
LOG_OUT = Path("reports/tables/final_three_selection_log.csv")
INTIMATE_PRODUCT_GROUPS: tuple[str, ...] = ("Underwear", "Nightwear", "Underwear/nightwear")
INTIMATE_GARMENT_GROUPS: tuple[str, ...] = ("Under-, Nightwear",)
INTIMATE_SHARE_THRESHOLD = 0.5
N_INCUMBENT = 1
N_EMERGING = 2
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
    return None


def reselect(
    t1: pl.DataFrame, t2: pl.DataFrame, intimate_types: frozenset[str]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Apply the rule; returns `(final_three, selection_log)` (both in decision order)."""
    log: list[dict] = []
    chosen: list[dict] = []

    def record(table: str, rank: int, row: dict, disposition: str, reason: str) -> None:
        log.append(
            {
                "source_table": table,
                "rank_in_table": rank,
                "style_key": row["style_key"],
                "predicted_intensity": row["predicted_intensity"],
                "growth_ratio": row.get("growth_ratio"),
                "disposition": disposition,
                "reason": reason,
            }
        )

    def eligible(table: str, frame: pl.DataFrame) -> list[tuple[int, dict]]:
        out = []
        for rank, row in enumerate(frame.to_dicts(), start=1):
            why = exclusion_reason(row, intimate_types)
            if why:
                record(table, rank, row, "skipped", why)
            else:
                out.append((rank, row))
        return out

    t1_ok = eligible("T1_incumbent", t1)
    t2_ok = eligible("T2_emerging", t2)

    used_colours: set[str] = set()
    used_pairs: set[tuple[str, str]] = set()

    def take(table: str, rank: int, row: dict, note: str) -> None:
        used_colours.add(row[COLOUR])
        used_pairs.add((row[TYPE], row[COLOUR]))
        chosen.append({**row, "source_table": table, "rank_in_source_table": rank})
        record(table, rank, row, "CHOSEN", note)

    n_inc = 0
    for rank, row in t1_ok:
        if n_inc >= N_INCUMBENT:
            record("T1_incumbent", rank, row, "not needed", "incumbent slot already filled")
            continue
        take("T1_incumbent", rank, row, "incumbent slot: highest-ranked T1 style after exclusion")
        n_inc += 1

    n_em = 0
    for rank, row in t2_ok:
        if n_em >= N_EMERGING:
            record("T2_emerging", rank, row, "not needed", "emerging slots already filled")
            continue
        if row["style_key"] in {c["style_key"] for c in chosen}:
            record("T2_emerging", rank, row, "skipped", "already chosen from T1")
        elif row[COLOUR] in used_colours:
            record(
                "T2_emerging", rank, row, "skipped", f"colour collision: {row[COLOUR]} already used"
            )
        elif (row[TYPE], row[COLOUR]) in used_pairs:
            record("T2_emerging", rank, row, "skipped", "(product_type, colour) pair already used")
        else:
            take("T2_emerging", rank, row, "emerging slot: next colour-distinct T2 style")
            n_em += 1
    if n_inc < N_INCUMBENT or n_em < N_EMERGING:
        print(
            f"SHORTFALL: incumbent {n_inc}/{N_INCUMBENT}, "
            f"emerging {n_em}/{N_EMERGING} (not backfilled)"
        )
    return pl.DataFrame(chosen, infer_schema_length=None), pl.DataFrame(
        log, infer_schema_length=None
    )


def main() -> None:
    """Run the rule on the committed leaderboards and write the final three + the skip log."""
    t1 = pl.read_csv(T1_PATH)
    t2 = pl.read_csv(T2_PATH)
    intimate = intimate_product_types(pl.read_csv(ARTICLES_PATH))
    print(f"excluded product types ({len(intimate)}): {sorted(intimate)}")
    final, log = reselect(t1, t2, intimate)
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
