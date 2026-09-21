"""Track 1c: yield at 24 seeds, with a cross-style false-pass floor. Rules: `reports/v3/`.

Pre-registered in `reports/v3/PREREGISTRATION.md` (Track 1c):

- Config unchanged (scale 0.35, compel weight 1.5, 8 concatenated references); seeds 42-49 already
  existed, seeds 50-65 are new: 24 per style, four styles.
- Pass = every automatic gate (`candidate_selection.passes_all_gates`).
- Reported: passes/24 pooled, and prior 8 (seeds 42-49) versus new 16 (seeds 50-65), each with a
  Wilson 95% interval. "Yield changed" only if those two intervals are disjoint.
- Floor: each style's 24 images are scored as if they were the NEXT style in registry order (wrong
  references, wrong attributes, wrong briefed changes) through the same gate stack; the
  false-pass rate is reported beside the yield.
- Selection: the 1a rule on each style's pool of 24 (a candidate only; the human check is the
  user's).

Also a free reproducibility check: seeds 42-49 are re-scored here and compared with the committed
`candidates_scored.csv` (deterministic judges, so gate results should be identical).

Usage (GPU for `score`; `report` is CPU and re-runnable):
    uv run python -m nss.generate.v3_yield score
    uv run python -m nss.generate.v3_yield report
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

from nss.generate import concept_generation, concept_scoring, final_registry
from nss.generate.candidate_selection import (
    GATE_COLUMNS,
    intervals_disjoint,
    pass_rate_table,
    passes_all_gates,
    select_candidate,
    wilson_interval,
)

STYLES: tuple[str, ...] = (*final_registry.STYLE_ORDER, final_registry.SUMMER)
SCALE = 0.35
SEEDS = range(42, 66)
SEED_GROUPS = {"prior_8": range(42, 50), "new_16": range(50, 66), "all_24": range(42, 66)}
BACKENDS = ["smolvlm", "florence2"]
SCORED_OUT = Path("reports/tables/v3_yield_scored.csv")
FLOOR_OUT = Path("reports/tables/v3_yield_floor_scored.csv")
RATES_OUT = Path("reports/tables/v3_yield_rates.csv")
FLOOR_RATES_OUT = Path("reports/tables/v3_yield_floor_rates.csv")
SELECTION_OUT = Path("reports/tables/v3_selection_24seed.csv")
REPRO_OUT = Path("reports/tables/v3_rescore_reproducibility.csv")
COMMITTED = Path("reports/tables/candidates_scored.csv")


def images_for(style_id: str) -> list[Path]:
    """The scale-0.35 candidates for seeds 42-65 that exist on disk."""
    paths = [concept_generation.image_path(style_id, SCALE, s) for s in SEEDS]
    return [p for p in paths if p.exists()]


def score() -> None:
    """Score every candidate against its own style (matched) and the next style (floor)."""
    thresholds = concept_scoring.judge_thresholds(BACKENDS)
    matched: list[dict] = []
    mismatched: list[dict] = []
    for i, style_id in enumerate(STYLES):
        imgs = images_for(style_id)
        matched += [
            {**r, "arm": "matched"} for r in concept_scoring.similarity_rows(style_id, imgs)
        ]
        target = STYLES[(i + 1) % len(STYLES)]
        wrong_changes = concept_generation.CHANGES[target]["applied_changes"]
        mismatched += [
            {**r, "arm": "mismatch", "source_style_id": style_id, "changes": wrong_changes}
            for r in concept_scoring.similarity_rows(target, imgs)
        ]
    rows = [*matched, *mismatched]
    for backend in BACKENDS:
        concept_scoring.judge_rows(backend, rows, thresholds)
    for row in rows:
        concept_scoring.apply_panel_rule(row)
    pl.DataFrame(matched, infer_schema_length=None).write_csv(SCORED_OUT)
    pl.DataFrame(
        [{k: v for k, v in r.items() if k != "changes"} for r in mismatched],
        infer_schema_length=None,
    ).write_csv(FLOOR_OUT)
    print(f"scored {len(matched)} matched and {len(mismatched)} mismatched images")


def reproducibility(new: pl.DataFrame) -> pl.DataFrame:
    """Compare re-scored seeds 42-49 with the committed table (gate booleans and fidelities)."""
    old = pl.read_csv(COMMITTED).filter(pl.col("scale") == SCALE)
    keys = ["style_id", "seed"]
    cols = [*GATE_COLUMNS, "smolvlm_fidelity", "florence2_fidelity", "smolvlm_gate3_answers"]
    j = new.join(old, on=keys, suffix="_old", how="inner")
    rows = []
    for c in cols:
        a, b = j[c], j[f"{c}_old"]
        same = (a == b) | (a.is_null() & b.is_null())
        if a.dtype.is_float():
            same = ((a - b).abs() < 1e-6) | same
        rows.append({"column": c, "n_compared": j.height, "n_identical": int(same.sum())})
    return pl.DataFrame(rows)


def report() -> None:
    """Pass-rate, floor, selection and reproducibility tables from the scored CSVs."""
    scored = pl.read_csv(SCORED_OUT)
    rates = pass_rate_table(scored, SEED_GROUPS)
    rates.write_csv(RATES_OUT)
    floor = pl.read_csv(FLOOR_OUT)
    frows = []
    for src in floor["source_style_id"].unique(maintain_order=True).to_list():
        sub = floor.filter(pl.col("source_style_id") == src).to_dicts()
        k = sum(passes_all_gates(r) for r in sub)
        lo, hi = wilson_interval(k, len(sub))
        frows.append(
            {
                "source_style_id": src,
                "scored_as": sub[0]["style_id"],
                "n": len(sub),
                "n_false_pass": k,
                "rate": k / len(sub),
                "wilson_lo": lo,
                "wilson_hi": hi,
            }
        )
    pl.DataFrame(frows).write_csv(FLOOR_RATES_OUT)
    picks = []
    for style_id in scored["style_id"].unique(maintain_order=True).to_list():
        pool = scored.filter(pl.col("style_id") == style_id).to_dicts()
        prior = [r for r in pool if r["seed"] < 50]
        pick, prior_pick = select_candidate(pool), select_candidate(prior)
        picks.append(
            {
                "style_id": style_id,
                "n_pool": len(pool),
                "n_pass": sum(passes_all_gates(r) for r in pool),
                "pick_seed_24": pick["seed"] if pick else None,
                "pick_seed_prior_8": prior_pick["seed"] if prior_pick else None,
                "pick_image": pick["image_path"] if pick else None,
                "pick_changes_visible": pick["smolvlm_gate3_answers"] if pick else None,
                "pick_smolvlm_fidelity": pick["smolvlm_fidelity"] if pick else None,
                "pick_closest_reference": pick["floor_max_sim"] if pick else None,
            }
        )
    pl.DataFrame(picks).write_csv(SELECTION_OUT)
    reproducibility(scored).write_csv(REPRO_OUT)
    with pl.Config(tbl_rows=40, tbl_width_chars=220, tbl_formatting="ASCII_FULL"):
        print(rates.with_columns(pl.col("style_id").str.slice(12, 14)))
        print(pl.DataFrame(frows).with_columns(pl.col("source_style_id").str.slice(12, 14)))
        print(pl.DataFrame(picks).drop("pick_image"))
        print(pl.read_csv(REPRO_OUT))
    for style_id in scored["style_id"].unique(maintain_order=True).to_list():
        a = rates.filter((pl.col("style_id") == style_id) & (pl.col("seed_group") == "prior_8"))
        b = rates.filter((pl.col("style_id") == style_id) & (pl.col("seed_group") == "new_16"))
        ia = (a["wilson_lo"][0], a["wilson_hi"][0])
        ib = (b["wilson_lo"][0], b["wilson_hi"][0])
        verdict = "CHANGED" if intervals_disjoint(ia, ib) else "consistent with prior"
        print(f"{style_id[:40]:40s} prior_8 vs new_16 intervals: {verdict}")


if __name__ == "__main__":
    {"score": score, "report": report}[sys.argv[1]]()
