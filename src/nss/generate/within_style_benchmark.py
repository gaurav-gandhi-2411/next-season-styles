"""Gate 1 re-anchored on real within-style article similarity (task H1).

DEFECT BEING FIXED: F1's Gate 1 compared a concept's margin against 90% of `copy_anchor_gen`, an
image generated at `ip_adapter_scale=1.0`. That is not a copy -- it is SDXL's most reference-
faithful *rendering*, already a new image -- so the gate rejected anything less than 90% as
faithful as the most faithful generation possible. A fidelity ceiling mislabelled as a plagiarism
check.

NEW BENCHMARK, externally grounded: how similar are two genuinely different, commercially released
H&M articles of the SAME style to each other? A concept no more similar to its references than
two real distinct products in that style are to each other is, by construction, as novel as an
actual new product in that assortment. Derived per style from the actual screened reference
images (never the global B3 0.9401, which pooled 6 styles).

STATISTIC. Primary (the gate): the concept's MEAN cosine to its style's references must be <= the
MEDIAN of that style's pairwise cosines between distinct real articles, in BOTH CLIP and DINOv2
(same both-spaces convention as the old copy check). Two sensitivity variants are reported next to
it, not gated on: leave-one-out mean (real-article analogue of "mean over refs") and nearest-
neighbour (the strictest plagiarism reading: closest single reference vs. each real article's
closest sibling).

KNOWN LIMITS (recorded, not hidden): (1) n is small -- 6-8 screened references per style, so
15-28 pairs; the medians carry sampling noise, and T-shirt DINOv2 fails one seed by 0.0001, which
is a coin-flip, not a finding. (2) A concept's mean similarity INCLUDES the reference it was
IP-Adapter conditioned on, so it is structurally closer to its references than a real article is
to its siblings -- the gate is conservative on that axis (measured: T-shirt and sweater concepts
sit ABOVE the real-sibling DINOv2 median), not lenient. (3) Real-vs-real pairs share catalogue
photography while generated-vs-real pairs do not, which biases concept CLIP similarity low -- a
CLIP pass alone is weak evidence; DINOv2 (structure) and the nearest-neighbour columns are the
stricter checks.

Usage:
    uv run python -m nss.generate.within_style_benchmark
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nss.generate.clip_scoring import _cosine
from nss.generate.vlm_judges import SKILL

F5_TABLE_PATH = Path("reports/tables/final_concepts_v3.csv")
BENCHMARK_PATH = Path("reports/tables/within_style_benchmark.csv")
RESCORED_PATH = Path("reports/tables/gate1_rescored_within_style.csv")
SPACES = ("clip", "dinov2")


def pairwise_similarities(embeddings: Sequence[np.ndarray]) -> list[float]:
    """Cosine similarity of every unordered pair of distinct articles' embeddings."""
    return [_cosine(a, b) for a, b in itertools.combinations(embeddings, 2)]


def style_benchmark(embeddings: Sequence[np.ndarray]) -> dict[str, float]:
    """Within-style benchmark statistics for one style's real reference embeddings.

    Args:
        embeddings: >= 2 embeddings of DISTINCT real articles of one style.

    Returns:
        `n_articles`, `n_pairs`, `pair_median` (the gate threshold), `pair_mean`, `pair_p90`,
        `loo_mean_median` (median over articles of mean cosine to the other articles) and
        `nn_median` (median over articles of cosine to their closest sibling).

    Raises:
        ValueError: with fewer than 2 articles (no pair exists -- fail loud, never a default).
    """
    if len(embeddings) < 2:
        raise ValueError("need >= 2 distinct articles to form a within-style benchmark")
    pairs = pairwise_similarities(embeddings)
    sims = np.array([[_cosine(a, b) for b in embeddings] for a in embeddings])
    np.fill_diagonal(sims, np.nan)
    return {
        "n_articles": len(embeddings),
        "n_pairs": len(pairs),
        "pair_median": float(np.median(pairs)),
        "pair_mean": float(np.mean(pairs)),
        "pair_p90": float(np.percentile(pairs, 90)),
        "loo_mean_median": float(np.median(np.nanmean(sims, axis=1))),
        "nn_median": float(np.median(np.nanmax(sims, axis=1))),
    }


def concept_similarity(
    concept: np.ndarray, reference_embeddings: Sequence[np.ndarray]
) -> dict[str, float]:
    """A concept's `mean` and nearest-neighbour (`max`) cosine to its style's references."""
    sims = [_cosine(concept, r) for r in reference_embeddings]
    return {"mean": float(np.mean(sims)), "max": float(np.max(sims))}


def gate1_within_style(concept_mean_sim: float, benchmark_median: float) -> bool:
    """True iff the concept is no more similar to its references than real distinct articles are.

    Single-space form of `skills/concept-qc/run_qc.py`'s `within_style_novelty_pass`.
    """
    return bool(SKILL.within_style_novelty_pass({"x": concept_mean_sim}, {"x": benchmark_median}))


def main() -> None:
    """Compute per-style benchmarks and re-score F5's 12 candidates. No image generation."""
    from nss.generate import clip_scoring, dino_scoring, screen_references

    embedders = {"clip": clip_scoring.embed_image, "dinov2": dino_scoring.embed_image}
    refs = screen_references.load_screened_references()

    bench_rows: list[dict[str, Any]] = []
    ref_embs: dict[str, dict[str, list[np.ndarray]]] = {}
    for style_id, paths in refs.items():
        ref_embs[style_id] = {}
        for space, fn in embedders.items():
            embs = [fn(p) for p in paths]
            ref_embs[style_id][space] = embs
            bench_rows.append({"style_id": style_id, "space": space, **style_benchmark(embs)})
    bench = pl.DataFrame(bench_rows)
    bench.write_csv(BENCHMARK_PATH)
    print(bench)

    f5 = pl.read_csv(F5_TABLE_PATH)
    thresholds = {(r["style_id"], r["space"]): r["pair_median"] for r in bench.to_dicts()}
    nn = {(r["style_id"], r["space"]): r["nn_median"] for r in bench.to_dicts()}
    out: list[dict[str, Any]] = []
    for row in f5.to_dicts():
        sid = row["style_id"]
        rec: dict[str, Any] = {
            "style_id": sid,
            "seed": row["seed"],
            "is_selected": row["is_selected"],
            "old_copy_check_pass": row["copy_check_pass"],
        }
        passes = []
        for space, fn in embedders.items():
            sim = concept_similarity(fn(Path(row["image_path"])), ref_embs[sid][space])
            ok = gate1_within_style(sim["mean"], thresholds[(sid, space)])
            passes.append(ok)
            rec[f"{space}_mean_sim"] = sim["mean"]
            rec[f"{space}_benchmark"] = thresholds[(sid, space)]
            rec[f"{space}_pass"] = ok
            rec[f"{space}_max_sim"] = sim["max"]
            rec[f"{space}_nn_benchmark"] = nn[(sid, space)]
            rec[f"{space}_nn_pass"] = sim["max"] <= nn[(sid, space)]
        rec["gate1_new_pass"] = all(passes)
        out.append(rec)
    res = pl.DataFrame(out)
    res.write_csv(RESCORED_PATH)
    print(res.select(pl.exclude("style_id")).with_columns(pl.col(pl.Float64).round(4)))
    print(
        f"Gate 1 pass: old {int(res['old_copy_check_pass'].sum())}/{res.height}"
        f" -> new {int(res['gate1_new_pass'].sum())}/{res.height}"
    )


if __name__ == "__main__":
    main()
