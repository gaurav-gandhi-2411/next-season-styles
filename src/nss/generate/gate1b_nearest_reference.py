"""Gate 1b: nearest-reference novelty, calibrated on real articles.

WHY: Gate 1 compares a concept's MEAN similarity to its references with the p90 of real pairwise
similarity. The clone positive control (`clone_positive_control.csv`) showed that an exact copy of
one reference passes it, because a mean over n references dilutes a copy of one. A copy detector
must look at the NEAREST reference, not the average.

CALIBRATION (no raw cut, no tuning on candidates): for each real reference article, its nearest
sibling similarity is the max cosine over the OTHER references of its style -- the same
leave-one-out set as the leave-one-out control. The Gate 1b threshold is the p90 of that
real-article distribution, per style and per embedding space (never a raw similarity cut, which
would repeat the median error). A concept passes iff its max cosine over its references is at or
below that threshold in EVERY space (joint AND, via the same `SKILL.within_style_novelty_pass` used
for Gate 1).

DECLARED BEFORE SEEING CANDIDATE RESULTS: (1) the exact clone of reference 0 must FAIL, else the
check does not work and stays un-gated; (2) the threshold is not adjusted after scoring the finals.

KNOWN ASYMMETRY (stated, not corrected): a real article's nearest sibling is taken over n-1
references, a concept's over n, so the concept gets one extra chance to be close. That makes
Gate 1b slightly stricter for concepts than for real articles. Also, a concept is IP-Adapter-
conditioned on reference 0, so some closeness to it is structural.

Usage:
    uv run python -m nss.generate.gate1b_nearest_reference
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nss.generate.clip_scoring import _cosine
from nss.generate.leave_one_out_control import SPACES, load_reference_embeddings
from nss.generate.vlm_judges import SKILL
from nss.generate.within_style_benchmark import concept_similarity

OUT_PATH = Path("reports/tables/gate1b_nearest_reference.csv")
GATE1B_PERCENTILE = 90


def nearest_sibling_similarities(embeddings: Sequence[np.ndarray]) -> list[float]:
    """Per real article: max cosine to any OTHER article of the style (leave-one-out)."""
    if len(embeddings) < 2:
        raise ValueError("need >= 2 distinct articles to form a nearest-sibling benchmark")
    return [
        max(_cosine(a, b) for j, b in enumerate(embeddings) if j != i)
        for i, a in enumerate(embeddings)
    ]


def gate1b_threshold(embeddings: Sequence[np.ndarray]) -> float:
    """p90 of the real-article nearest-sibling similarity distribution (linear interpolation)."""
    return float(np.percentile(nearest_sibling_similarities(embeddings), GATE1B_PERCENTILE))


def gate1b_pass(
    concept_embeddings: dict[str, np.ndarray], reference_embeddings: dict[str, Sequence[np.ndarray]]
) -> dict[str, Any]:
    """Score one image (`{space: embedding}`) against one style's references; per-space + joint."""
    sims = {
        s: concept_similarity(concept_embeddings[s], reference_embeddings[s])["max"] for s in SPACES
    }
    thr = {s: gate1b_threshold(reference_embeddings[s]) for s in SPACES}
    out: dict[str, Any] = {"joint_pass": SKILL.within_style_novelty_pass(sims, thr)}
    for s in SPACES:
        out[f"{s}_max_sim"] = sims[s]
        out[f"{s}_threshold"] = thr[s]
        out[f"{s}_pass"] = SKILL.within_style_novelty_pass({s: sims[s]}, {s: thr[s]})
    return out


def final_concept_paths() -> dict[str, Path]:
    """The three final selected concept images, by style id."""
    from nss.generate.final_selection_figures import SELECTED

    return dict(SELECTED)


def main() -> None:
    """Calibrate Gate 1b, run the clone control, score the three finals, write `OUT_PATH`."""
    from nss.generate import clip_scoring, dino_scoring

    embedders = {"clip": clip_scoring.embed_image, "dinov2": dino_scoring.embed_image}
    refs = load_reference_embeddings()
    finals = final_concept_paths()
    rows: list[dict[str, Any]] = []
    for sid, ref_embs in refs.items():
        real = {s: nearest_sibling_similarities(ref_embs[s]) for s in SPACES}
        thr = {s: gate1b_threshold(ref_embs[s]) for s in SPACES}
        # Real-article sanity: pass rate of each article's own NN against the full-set threshold.
        real_pass = [
            SKILL.within_style_novelty_pass(
                {s: real[s][i] for s in SPACES}, {s: thr[s] for s in SPACES}
            )
            for i in range(len(ref_embs[SPACES[0]]))
        ]
        clone = gate1b_pass({s: ref_embs[s][0] for s in SPACES}, ref_embs)
        final_emb = {s: embedders[s](finals[sid]) for s in SPACES}
        final = gate1b_pass(final_emb, ref_embs)
        rows.append(
            {
                "style_id": sid,
                "n_real_articles": len(real_pass),
                "real_article_pass_rate": float(np.mean(real_pass)),
                **{f"{s}_real_nn_median": float(np.median(real[s])) for s in SPACES},
                **{f"{s}_real_nn_max": float(np.max(real[s])) for s in SPACES},
                **{f"{s}_threshold_p90": thr[s] for s in SPACES},
                "clone_joint_pass": clone["joint_pass"],
                **{f"clone_{s}_max_sim": clone[f"{s}_max_sim"] for s in SPACES},
                **{f"clone_{s}_pass": clone[f"{s}_pass"] for s in SPACES},
                "final_image": str(finals[sid]),
                "final_joint_pass": final["joint_pass"],
                **{f"final_{s}_max_sim": final[f"{s}_max_sim"] for s in SPACES},
                **{f"final_{s}_pass": final[f"{s}_pass"] for s in SPACES},
            }
        )
    df = pl.DataFrame(rows)
    df.write_csv(OUT_PATH)
    with pl.Config(tbl_cols=-1, tbl_width_chars=250, fmt_str_lengths=30):
        print(df.drop("final_image").with_columns(pl.col(pl.Float64).round(4)))
    print(f"clone fails Gate 1b in all styles: {not df['clone_joint_pass'].any()}")


if __name__ == "__main__":
    main()
