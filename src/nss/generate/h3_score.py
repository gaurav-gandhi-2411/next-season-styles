"""Score H3's regenerated underwear candidates under H1's Gate 1 and H2's judge checklist.

Gate 1 benchmark = median within-style pairwise similarity of the H3 verified solid references
(4 articles -> only 6 pairs per space: a noisy benchmark, stated in the report). Visual QC verdicts
below come from reading all four images (`reports/figures/`-independent, recorded here so the
selection is auditable).

Usage:
    uv run python -m nss.generate.h3_score
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from nss.generate import clip_scoring, dino_scoring, h2_rejudge
from nss.generate.h3_generate import OUTPUT_DIR, SEEDS, reference_paths
from nss.generate.h3_underwear_refs import STYLE_ID
from nss.generate.within_style_benchmark import (
    concept_similarity,
    gate1_within_style,
    style_benchmark,
)

OUT_PATH = Path("reports/tables/h3_underwear_scored.csv")
# From reading every generated image (task H3 step: visual inspection of ALL candidates).
VISUAL_QC: dict[int, str] = {
    42: "reject: malformed halter/cut-out garment, not a brief",
    43: "keep: clean solid red brief, black piping, no lace/floral/mesh",
    44: "reject: sheer mesh panels (pattern drift, not solid)",
    45: "reject: folded/malformed object, no coherent garment",
}


def image_path(seed: int) -> Path:
    """Path H3 generation wrote for `seed` (matches `generate_candidates_for_style`'s slug)."""
    return OUTPUT_DIR / f"ladieswear_underwear-bottom_under--nightwear_red_solid_seed{seed}.png"


def main() -> None:
    """Score the 4 candidates: Gate 1 (new benchmark), judge panel (H2 checklist), visual QC."""
    refs = reference_paths()
    embedders = {"clip": clip_scoring.embed_image, "dinov2": dino_scoring.embed_image}
    ref_embs = {s: [fn(p) for p in refs] for s, fn in embedders.items()}
    bench = {s: style_benchmark(e) for s, e in ref_embs.items()}

    _, old_t, new_t = h2_rejudge.recompute_calibration()
    cache = h2_rejudge.judge_images([(STYLE_ID, sd, image_path(sd)) for sd in SEEDS])
    rescore = h2_rejudge.build_rescore_table(cache, old_t, new_t)
    rescore.write_csv(h2_rejudge.RESCORE_PATH)

    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        rec: dict[str, Any] = {"seed": seed, "visual_qc": VISUAL_QC[seed]}
        passes = []
        for space, fn in embedders.items():
            sim = concept_similarity(fn(image_path(seed)), ref_embs[space])
            ok = gate1_within_style(sim["mean"], bench[space]["pair_median"])
            passes.append(ok)
            rec |= {
                f"{space}_mean_sim": sim["mean"],
                f"{space}_benchmark": bench[space]["pair_median"],
                f"{space}_pass": ok,
                f"{space}_max_sim": sim["max"],
                f"{space}_nn_benchmark": bench[space]["nn_median"],
            }
        rec["gate1_pass"] = all(passes)
        judged = rescore.filter(
            (pl.col("image_path") == str(image_path(seed))) & pl.col("available")
        )
        for r in judged.to_dicts():
            rec[f"{r['judge']}_fidelity_3attr"] = r["fidelity_after_3attr"]
            rec[f"{r['judge']}_threshold"] = r["threshold_after"]
            rec[f"{r['judge']}_pass"] = r["pass_after"]
        rows.append(rec)
    out = pl.DataFrame(rows, infer_schema_length=None)
    out.write_csv(OUT_PATH)
    with pl.Config(tbl_cols=-1, tbl_rows=10, tbl_width_chars=250):
        print(out.select(pl.exclude("visual_qc")).with_columns(pl.col(pl.Float64).round(3)))


if __name__ == "__main__":
    main()
