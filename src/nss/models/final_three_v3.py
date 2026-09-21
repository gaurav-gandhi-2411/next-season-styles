"""G2: re-select the final three under the outcome of G1 (pre-registered, no regeneration).

G1 (`reports/v3/PREREGISTRATION.md`, commit `e22a0ed`; result `emerging_redesign`) did not adopt the
redesigned emerging score, so the pre-stated FALLBACK applies: the final three come from the
validated INTENSITY table (guard-passing styles ranked by predicted intensity, diversity-constrained
on (product type, colour)), then through the unchanged `reselect_final_three` rules (intimates and
visual-ambiguity exclusions, no shared colour, no shared product type, first three that survive,
shortfall reported and never backfilled).

CONTROL GATES, before anything is selected: with the frozen final model retrained in this process
the existing code must reproduce the committed `top_styles_emerging.csv` (style order and predicted
intensity) and `top_styles_incumbent.csv`, and `reselect` on the committed emerging table must give
the committed final three. Otherwise the run stops.

    PYTHONHASHSEED=0 uv run python -m nss.models.final_three_v3
"""

from __future__ import annotations

import math

import polars as pl

from nss.models import final_forecast
from nss.models.diversity_forecast import select_t1_incumbent, select_t2_emerging
from nss.models.reselect_final_three import (
    ARTICLES_PATH,
    COLOUR,
    FINAL_OUT,
    N_FINAL,
    T2_PATH,
    TYPE,
    exclusion_reason,
    intimate_product_types,
    reselect,
)

OUT = "reports/tables"
T1_COMMITTED = "reports/tables/top_styles_incumbent.csv"
CURRENT_LABELS = {
    "Ladieswear || Sweater || Knitwear || Beige || Melange": "sweater",
    "Ladieswear || Dress || Dresses Ladies || Red || Solid": "dress",
    "Ladieswear || Top || Jersey Basic || White || Solid": "white top",
}


def reselect_from(
    table: pl.DataFrame, source: str, intimate_types: frozenset[str]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """`reselect_final_three.reselect`'s walk, with the source table named (same rules exactly)."""
    log: list[dict] = []
    chosen: list[dict] = []

    def record(rank: int, row: dict, disposition: str, reason: str) -> None:
        log.append(
            {
                "source_table": source,
                "rank_in_table": rank,
                "style_key": row["style_key"],
                "predicted_intensity": row["predicted_intensity"],
                "disposition": disposition,
                "reason": reason,
            }
        )

    for rank, row in enumerate(table.to_dicts(), start=1):
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
            chosen.append({**row, "source_table": source, "rank_in_source_table": rank})
            record(rank, row, "CHOSEN", "guards met; no collision")
    if len(chosen) < N_FINAL:
        print(f"SHORTFALL: chose {len(chosen)}/{N_FINAL} (not backfilled)")
    return pl.DataFrame(chosen, infer_schema_length=None), pl.DataFrame(
        log, infer_schema_length=None
    )


def parity(recomputed: pl.DataFrame, committed: pl.DataFrame, name: str) -> None:
    """Stop unless the recomputed leaderboard equals the committed one (order and intensity)."""
    a, b = recomputed["style_key"].to_list(), committed["style_key"].to_list()
    if a != b:
        raise SystemExit(f"parity FAILED for {name}: style order differs")
    x = recomputed["predicted_intensity"].to_list()
    y = committed["predicted_intensity"].to_list()
    if not all(math.isclose(p, q, rel_tol=0, abs_tol=1e-9) for p, q in zip(x, y, strict=True)):
        raise SystemExit(f"parity FAILED for {name}: predicted intensity differs")
    print(f"parity OK: {name} reproduces the committed table ({len(a)} styles)")


def main() -> None:
    """Gates, then the fallback selection, its log and the survivor table."""
    panel = pl.read_parquet(final_forecast.DEFAULT_PANEL_PATH)
    final_forecast.verify_forecast_origin(panel)
    model, _frame, columns = final_forecast.train_final_model(panel)
    ranking = final_forecast.build_ranking_frame(panel, model, columns)
    t1 = select_t1_incumbent(ranking)
    t2 = select_t2_emerging(panel, ranking)
    parity(t1, pl.read_csv(T1_COMMITTED), "incumbent (intensity) table")
    parity(t2, pl.read_csv(T2_PATH), "emerging (growth) table")

    intimate = intimate_product_types(pl.read_csv(ARTICLES_PATH))
    current, _ = reselect(pl.read_csv(T2_PATH), intimate)
    committed = pl.read_csv(FINAL_OUT)
    if current["style_key"].to_list() != committed["style_key"].to_list():
        raise SystemExit("gate FAILED: reselect on the committed emerging table != committed three")
    print("gate OK: the existing rules reproduce the committed final three")

    new_three, log = reselect_from(t1, "T1_intensity (fallback)", intimate)
    new_three.write_csv(f"{OUT}/v3_final_three_fallback.csv")
    log.write_csv(f"{OUT}/v3_final_three_fallback_log.csv")
    old = committed["style_key"].to_list()
    new = new_three["style_key"].to_list()
    rows = [
        {
            "current_style": k,
            "role": CURRENT_LABELS.get(k, ""),
            "survives_as_same_style": k in new,
            "same_product_type_in_new_three": k.split(" || ")[1]
            in {n.split(" || ")[1] for n in new},
        }
        for k in old
    ]
    pl.DataFrame(rows).write_csv(f"{OUT}/v3_final_three_survivors.csv")
    with pl.Config(
        tbl_rows=30, tbl_width_chars=230, fmt_str_lengths=80, tbl_formatting="ASCII_FULL"
    ):
        print(
            log.select("rank_in_table", "style_key", "predicted_intensity", "disposition", "reason")
        )
        print(new_three.select("rank_in_source_table", "style_key", "predicted_intensity"))
        print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
