"""Leave-one-out control for Gate 1 (task J1/J2): how often does a REAL article pass the gate?

WHY: Gate 1's H1 threshold was the MEDIAN pairwise similarity between real distinct articles of a
style. A median means roughly half of genuinely new real products fail by construction -- a coin
flip, not a copy detector. This control measures that directly instead of arguing it: each real
reference article is scored against the OTHER references of its own style through the EXACT code
path used for generated candidates (`concept_similarity` for the statistic,
`SKILL.within_style_novelty_pass` for the verdict) -- no parallel implementation.

TWO benchmark conventions are reported side by side:
* `full` -- the threshold candidates actually face (computed over ALL of the style's references,
  including the article being scored). Primary: this is the gate as shipped.
* `heldout` -- the threshold recomputed from the OTHER references only (n-1 articles), the
  stricter "genuinely new product" reading (the held-out article contributes to neither side).

Usage:
    uv run python -m nss.generate.leave_one_out_control
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nss.generate.vlm_judges import SKILL
from nss.generate.within_style_benchmark import concept_similarity, style_benchmark

OUT_PATH = Path("reports/tables/leave_one_out_control.csv")
CLONE_OUT_PATH = Path("reports/tables/clone_positive_control.csv")
SPACES = ("clip", "dinov2")
# Gate 1 statistic under test: name of the `style_benchmark` key used as the threshold.
RULES: dict[str, str] = {"median": "pair_median", "p90": "pair_p90"}


def loo_rows(
    style_id: str, embeddings: dict[str, Sequence[np.ndarray]], rule: str
) -> list[dict[str, Any]]:
    """One row per real article: its LOO similarity, thresholds and pass flags under `rule`."""
    stat = RULES[rule]
    n = len(embeddings[SPACES[0]])
    full = {s: style_benchmark(embeddings[s]) for s in SPACES}
    rows: list[dict[str, Any]] = []
    for i in range(n):
        rec: dict[str, Any] = {"style_id": style_id, "rule": rule, "article_index": i}
        sims: dict[str, float] = {}
        full_thr: dict[str, float] = {}
        held_thr: dict[str, float] = {}
        for s in SPACES:
            others = [e for j, e in enumerate(embeddings[s]) if j != i]
            sims[s] = concept_similarity(embeddings[s][i], others)["mean"]
            full_thr[s] = full[s][stat]
            held_thr[s] = style_benchmark(others)[stat]
            rec[f"{s}_loo_sim"] = sims[s]
            rec[f"{s}_threshold_full"] = full_thr[s]
            rec[f"{s}_pair_max"] = full[s]["pair_max"]  # reference only, never gated
            rec[f"{s}_pass_full"] = SKILL.within_style_novelty_pass({s: sims[s]}, {s: full_thr[s]})
            rec[f"{s}_threshold_heldout"] = held_thr[s]
            rec[f"{s}_pass_heldout"] = SKILL.within_style_novelty_pass(
                {s: sims[s]}, {s: held_thr[s]}
            )
        rec["joint_pass_full"] = SKILL.within_style_novelty_pass(sims, full_thr)
        rec["joint_pass_heldout"] = SKILL.within_style_novelty_pass(sims, held_thr)
        rows.append(rec)
    return rows


def summarise(rows: pl.DataFrame) -> pl.DataFrame:
    """Pass counts per style x rule, per metric and jointly, for both benchmark conventions."""
    cols = [f"{c}_pass_{v}" for v in ("full", "heldout") for c in (*SPACES, "joint")]
    return rows.group_by(["style_id", "rule"], maintain_order=True).agg(
        pl.len().alias("n_articles"), *[pl.col(c).sum().alias(c) for c in cols]
    )


def load_reference_embeddings() -> dict[str, dict[str, list[np.ndarray]]]:
    """Embeddings per style/space for the reference set each style's FINAL concept used.

    T-shirt and sweater: the screened references. Underwear: H3's verified plain-solid references
    (the screened set is the lace one the H3 defect fixed).
    """
    from nss.generate import (
        clip_scoring,
        dino_scoring,
        h3_generate,
        h3_underwear_refs,
        screen_references,
    )

    embedders = {"clip": clip_scoring.embed_image, "dinov2": dino_scoring.embed_image}
    refs = dict(screen_references.load_screened_references())
    refs[h3_underwear_refs.STYLE_ID] = h3_generate.reference_paths()
    return {
        sid: {s: [fn(p) for p in paths] for s, fn in embedders.items()}
        for sid, paths in refs.items()
    }


def clone_rows(style_id: str, embeddings: dict[str, Sequence[np.ndarray]]) -> list[dict[str, Any]]:
    """Positive control: an exact copy of reference 0 scored exactly like a generated candidate.

    A copy detector must FAIL this. Reports the gate statistic (mean cosine to ALL references,
    which include the copied one -- the same construction candidates face) under each rule, plus
    the nearest-neighbour statistic (max cosine to any reference) that the mean can dilute.
    """
    rec: dict[str, Any] = {"style_id": style_id}
    for rule, stat in RULES.items():
        sims = {s: concept_similarity(embeddings[s][0], embeddings[s])["mean"] for s in SPACES}
        thr = {s: style_benchmark(embeddings[s])[stat] for s in SPACES}
        rec[f"gate1_pass_{rule}"] = SKILL.within_style_novelty_pass(sims, thr)
    for s in SPACES:
        bench = style_benchmark(embeddings[s])
        sim = concept_similarity(embeddings[s][0], embeddings[s])
        rec[f"{s}_mean_sim"] = sim["mean"]
        rec[f"{s}_p90"] = bench["pair_p90"]
        rec[f"{s}_max_sim"] = sim["max"]
        rec[f"{s}_nn_median_benchmark"] = bench["nn_median"]
    return [rec]


def main(rules: Sequence[str] = tuple(RULES)) -> pl.DataFrame:
    """Run the control under each rule, write the per-article CSV, print the pass-count summary."""
    embs = load_reference_embeddings()
    rows = [r for sid, e in embs.items() for rule in rules for r in loo_rows(sid, e, rule)]
    df = pl.DataFrame(rows)
    df.write_csv(OUT_PATH)
    clones = pl.DataFrame([r for sid, e in embs.items() for r in clone_rows(sid, e)])
    clones.write_csv(CLONE_OUT_PATH)
    with pl.Config(tbl_cols=-1, tbl_width_chars=250, fmt_str_lengths=40):
        print(summarise(df))
        print(clones.drop("style_id"))
    return df


if __name__ == "__main__":
    main()
