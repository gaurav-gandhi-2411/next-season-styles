"""1e: negative/positive controls and the no-side-effect check for the "All over pattern" mapping.

Pre-registered in `reports/v3/PREREGISTRATION.md` (1e). Three parts:

1. NEGATIVE CONTROL (mandatory): 6 real catalogue photos of SOLID-colour bikini tops (lowest article
   id per colour among Orange, Black, White, Red, Blue, Pink). SmolVLM's short answer for each goes
   through `pattern_label.mapped_graphical_score(answer, "All over pattern")`. Every one must FAIL.
2. POSITIVE CONTROL (reported, not gating): every real "All over pattern" bikini top with a local
   photo.
3. NO-SIDE-EFFECT CHECK: all 96 candidates re-scored with the mapping from their stored (greedy,
   deterministic) extractions; every gate column of the non-bikini styles must be unchanged.

    uv run python -m nss.generate.v3_pattern_control fetch     # solid-bikini photos (Kaggle)
    uv run python -m nss.generate.v3_pattern_control controls  # SmolVLM on both control sets (GPU)
    uv run python -m nss.generate.v3_pattern_control rescore   # CPU
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

from nss.generate import concept_scoring, local_vlm, pattern_label
from nss.generate.candidate_selection import GATE_COLUMNS, passes_all_gates
from nss.generate.concept_qc_pipeline import parse_style_attributes
from nss.generate.fidelity import applicable_dimensions
from nss.generate.vlm_judges import SKILL

ARTICLES = Path("data/raw/articles.csv")
IMAGES = Path("data/images")
SOLID_COLOURS = ("Orange", "Black", "White", "Red", "Blue", "Pink")
CONTROL_OUT = Path("reports/tables/v3_pattern_control.csv")
RESCORE_OUT = Path("reports/tables/v3_pattern_rescore.csv")
RESCORE_SUMMARY_OUT = Path("reports/tables/v3_pattern_rescore_summary.csv")
SCORED = Path("reports/tables/v3_yield_scored.csv")


def _bikini_articles() -> pl.DataFrame:
    return pl.read_csv(ARTICLES).filter(pl.col("product_type_name") == "Bikini top")


def solid_control_ids() -> list[tuple[int, str]]:
    """(article_id, colour): lowest id per colour among solid bikini tops (pre-registered)."""
    solid = _bikini_articles().filter(pl.col("graphical_appearance_name") == "Solid")
    out = []
    for colour in SOLID_COLOURS:
        sub = solid.filter(pl.col("perceived_colour_master_name") == colour).sort("article_id")
        if sub.height:
            out.append((int(sub["article_id"][0]), colour))
    return out


def local_photo(article_id: int) -> Path | None:
    """A photo already on disk for this article, if any."""
    path = IMAGES / f"{article_id:010d}.jpg"
    if path.exists():
        return path
    hits = list(Path("data").rglob(f"{article_id:010d}.jpg"))
    return hits[0] if hits else None


def fetch() -> None:
    """Fetch the six solid-bikini photos with the existing one-file-at-a-time fetcher."""
    from nss.data import fetch_images

    ids = [i for i, _ in solid_control_ids() if local_photo(i) is None]
    print("fetching", ids)
    print(fetch_images.fetch_images(ids, IMAGES))


def controls() -> None:
    """SmolVLM's blind answer on the negative and positive control photos, through the rule."""
    rows: list[dict] = []
    negatives = [(i, c, local_photo(i)) for i, c in solid_control_ids()]
    pat = _bikini_articles().filter(pl.col("graphical_appearance_name") == "All over pattern")
    positives = [
        (int(r["article_id"]), r["perceived_colour_master_name"], local_photo(int(r["article_id"])))
        for r in pat.sort("article_id").to_dicts()
    ]
    local_vlm.load("smolvlm")
    for kind, group in (("negative_solid", negatives), ("positive_all_over_pattern", positives)):
        for article_id, colour, path in group:
            if path is None:
                continue
            answer = local_vlm.extract_attributes_local(path, ("graphical_treatment",))[
                "graphical_treatment"
            ]
            rows.append(
                {
                    "control": kind,
                    "article_id": article_id,
                    "colour": colour,
                    "smolvlm_answer": answer,
                    "satisfies_all_over_pattern": pattern_label.mapped_graphical_score(
                        answer, pattern_label.LABEL
                    )
                    == 1.0,
                }
            )
    local_vlm.unload()
    table = pl.DataFrame(rows)
    table.write_csv(CONTROL_OUT)
    neg = table.filter(pl.col("control") == "negative_solid")
    pos = table.filter(pl.col("control") == "positive_all_over_pattern")
    wrongly = int(neg["satisfies_all_over_pattern"].sum())
    with pl.Config(tbl_rows=40, tbl_width_chars=160, tbl_formatting="ASCII_FULL"):
        print(table)
    print(f"NEGATIVE CONTROL: {neg.height} solid bikinis, {wrongly} wrongly pass the mapped check")
    print(f"POSITIVE CONTROL: {int(pos['satisfies_all_over_pattern'].sum())}/{pos.height} pass")
    print("VERDICT:", "MAPPING BROKEN, REVERT" if wrongly else "negative control passed")


def rescore() -> None:
    """Re-score all 96 candidates with the mapping on (from stored extractions); diff vs before."""
    scored = pl.read_csv(SCORED)
    thresholds = concept_scoring.judge_thresholds(["smolvlm", "florence2"])
    rows = []
    for r in scored.to_dicts():
        truth = parse_style_attributes(r["style_id"])
        dims = applicable_dimensions(truth["graphical_treatment"])
        new = dict(r)
        for judge in ("smolvlm", "florence2"):
            extraction = json.loads(r[f"{judge}_extraction"])
            scores = SKILL.score_attributes(extraction, {d: truth[d] for d in dims})
            scores = pattern_label.apply_mapping(scores, extraction, truth)
            new[f"{judge}_fidelity"] = SKILL.mean_score(scores)
            new[f"{judge}_gate2_pass"] = new[f"{judge}_fidelity"] >= thresholds[judge]
        new["gate2_pass"] = new["smolvlm_gate2_pass"]  # SmolVLM gates; Florence-2 is advisory
        rows.append(new)
    after = pl.DataFrame(rows, infer_schema_length=None)
    after.write_csv(RESCORE_OUT)
    out = []
    for style_id in scored["style_id"].unique(maintain_order=True).to_list():
        b = scored.filter(pl.col("style_id") == style_id).sort("seed")
        a = after.filter(pl.col("style_id") == style_id).sort("seed")
        changed = sum(
            any(rb[c] != ra[c] for c in (*GATE_COLUMNS, "smolvlm_fidelity"))
            for rb, ra in zip(b.to_dicts(), a.to_dicts(), strict=True)
        )
        out.append(
            {
                "style_id": style_id,
                "n": b.height,
                "rows_with_any_gate_or_fidelity_change": changed,
                "gate2_pass_before": int(b["gate2_pass"].sum()),
                "gate2_pass_after": int(a["gate2_pass"].sum()),
                "all_gates_pass_before": sum(passes_all_gates(x) for x in b.to_dicts()),
                "all_gates_pass_after": sum(passes_all_gates(x) for x in a.to_dicts()),
            }
        )
    summary = pl.DataFrame(out)
    summary.write_csv(RESCORE_SUMMARY_OUT)
    with pl.Config(tbl_rows=10, tbl_width_chars=200, tbl_formatting="ASCII_FULL"):
        print(summary.with_columns(pl.col("style_id").str.slice(12, 16)))


if __name__ == "__main__":
    {"fetch": fetch, "controls": controls, "rescore": rescore}[sys.argv[1]]()
