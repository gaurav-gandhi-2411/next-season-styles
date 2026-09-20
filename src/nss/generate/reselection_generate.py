"""Regenerate the final three's concepts with the concrete briefs (task M3) -- pipeline unchanged.

Same primitives as every earlier generation: `final_concepts.build_generation_spec` (attribute-first
prompt, 77-token budget enforced, style-specific negative prompt), `build_prompt_2`,
`generate_candidates_for_style` at `final_concepts_v2.IP_ADAPTER_SCALE` (0.45), 4 seeds, references
from `screen_references.load_screened_references` (best-selling full-garment photo first). Output
goes to `data/generated/final_concepts_m3/attempt<N>/`.

Usage:
    uv run python -m nss.generate.reselection_generate generate <attempt>
    uv run python -m nss.generate.reselection_generate score <attempt>
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

OUT_ROOT = Path("data/generated/final_concepts_m3")
SCORED = "reports/tables/m3_candidates_attempt{attempt}_scored.csv"


def attempt_dir(attempt: int) -> Path:
    """Where attempt `attempt`'s images live."""
    return OUT_ROOT / f"attempt{attempt}"


def generate(attempt: int) -> None:
    """Generate 4 seeds for each of the three styles (GPU)."""
    from nss.generate import final_concepts, final_concepts_v2, screen_references
    from nss.generate.scale_sweep import free_sdxl_pipeline

    briefs = final_concepts.load_design_briefs()
    refs = screen_references.load_screened_references()
    out = attempt_dir(attempt)
    out.mkdir(parents=True, exist_ok=True)
    for style_id, brief in briefs.items():
        prompt, negative = final_concepts.build_generation_spec(style_id, brief)
        prompt_2 = final_concepts.build_prompt_2(style_id)
        print("PROMPT:", prompt)
        final_concepts.generate_candidates_for_style(
            style_id,
            prompt,
            negative,
            refs[style_id],
            seeds=tuple(final_concepts_v2.INITIAL_SEEDS),
            ip_adapter_scale=final_concepts_v2.IP_ADAPTER_SCALE,
            output_dir=out,
            prompt_2=prompt_2,
            negative_prompt_2=negative,
        )
    free_sdxl_pipeline()


def image_path(style_id: str, seed: int, attempt: int) -> Path:
    """Path `generate_candidates_for_style` wrote for this style/seed (its own slug rule)."""
    from nss.generate import final_concepts

    return attempt_dir(attempt) / f"{final_concepts._slugify(style_id)}_seed{seed}.png"


def score(attempt: int) -> pl.DataFrame:
    """Gate 1 + Gate 1b + clone check for every candidate of `attempt` (same code as the finals)."""
    from nss.generate import clip_scoring, dino_scoring, final_concepts, screen_references
    from nss.generate.gate1b_nearest_reference import gate1b_pass
    from nss.generate.vlm_judges import SKILL
    from nss.generate.within_style_benchmark import concept_similarity, style_benchmark

    spaces = ("clip", "dinov2")
    embedders = {"clip": clip_scoring.embed_image, "dinov2": dino_scoring.embed_image}
    briefs = final_concepts.load_design_briefs()
    refs = screen_references.load_screened_references()
    rows = []
    for style_id in briefs:
        ref_embs = {s: [embedders[s](p) for p in refs[style_id]] for s in spaces}
        bench = {s: style_benchmark(ref_embs[s]) for s in spaces}
        clone = gate1b_pass({s: ref_embs[s][0] for s in spaces}, ref_embs)
        for seed in (42, 43, 44, 45):
            path = image_path(style_id, seed, attempt)
            emb = {s: embedders[s](path) for s in spaces}
            sims = {s: concept_similarity(emb[s], ref_embs[s])["mean"] for s in spaces}
            thr = {s: bench[s]["pair_p90"] for s in spaces}
            b = gate1b_pass(emb, ref_embs)
            rows.append(
                {
                    "style_id": style_id,
                    "seed": seed,
                    "attempt": attempt,
                    "image_path": str(path),
                    **{f"{s}_mean_sim": sims[s] for s in spaces},
                    **{f"{s}_p90_threshold": thr[s] for s in spaces},
                    "gate1_pass": SKILL.within_style_novelty_pass(sims, thr),
                    **{f"{s}_max_sim": b[f"{s}_max_sim"] for s in spaces},
                    **{f"{s}_gate1b_threshold": b[f"{s}_threshold"] for s in spaces},
                    "gate1b_pass": b["joint_pass"],
                    "clone_gate1b_pass": clone["joint_pass"],
                }
            )
    df = pl.DataFrame(rows)
    df.write_csv(SCORED.format(attempt=attempt))
    return df


def main() -> None:
    """CLI dispatch."""
    cmd, attempt = sys.argv[1], int(sys.argv[2])
    if cmd == "generate":
        generate(attempt)
    elif cmd == "score":
        with pl.Config(tbl_cols=-1, tbl_width_chars=240, fmt_str_lengths=22):
            print(
                score(attempt)
                .drop("image_path", "attempt", "clone_gate1b_pass")
                .with_columns(pl.col(pl.Float64).round(3))
            )
    else:
        raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    main()
