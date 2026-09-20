"""Run the N1 lever experiments on one style and save every image with its config in the name.

Config tokens: ``<mode>:<scale>[~w<weight>]`` where mode is single|concat|mean and scale is a float
or ``style_only@<s>`` / ``style_layout@<s>``; ``~w1.5`` upweights the change clauses with compel.
Example: ``concat:0.25``  ``mean:style_only@1.0``  ``concat:0.35~w1.4``.

Usage:
    uv run python -m nss.generate.n1_experiment <style-keyword> <seeds,comma> <cfg> [<cfg> ...]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from nss.generate import final_concepts, levers, screen_references
from nss.generate.scale_sweep import free_sdxl_pipeline

OUT = levers.OUT_ROOT


def parse_scale(token: str) -> float | dict:
    """`0.25` -> 0.25; `style_only@1.0` -> the per-block dict."""
    if "@" in token:
        kind, strength = token.split("@")
        return levers.block_scale(kind, float(strength))
    return float(token)


def weight_change_clauses(prompt: str, weight: float) -> str:
    """Wrap each `Novel accent: ...` clause in compel weight syntax."""
    return re.sub(r"(Novel accent: [^.]+)\.", lambda m: f"({m.group(1)}){weight}.", prompt)


def natural_prompt(style_id: str, brief: dict, weight: float | None = None) -> str:
    """One natural sentence: colour + pattern + product, then the briefed changes (<= 77 tokens)."""
    _dept, product, _group, colour, pattern = (p.strip() for p in style_id.split(" || "))
    parts = [f"({c}){weight}" if weight else c for c in brief["applied_changes"]]
    changes = " and ".join(parts)
    return (
        f"flat-lay product photo of a {colour.lower()} {pattern.lower()} {product.lower()} "
        f"with {changes}, plain light background"
    )


def config_name(cfg: str) -> str:
    """Filesystem-safe config label."""
    return re.sub(r"[^A-Za-z0-9.]+", "-", cfg).strip("-")


def main(argv: list[str]) -> None:
    """Generate every (config, seed) image for the style matching `argv[0]`."""
    keyword, seeds = argv[0].lower(), [int(s) for s in argv[1].split(",")]
    briefs = final_concepts.load_design_briefs()
    style_id = next(s for s in briefs if keyword in s.lower())
    prompt, negative = final_concepts.build_generation_spec(style_id, briefs[style_id])
    prompt_2 = final_concepts.build_prompt_2(style_id)
    refs = screen_references.load_screened_references()[style_id]
    out_dir = OUT / final_concepts._slugify(style_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("PROMPT:", prompt, "| n_refs", len(refs))
    natural = natural_prompt(style_id, briefs[style_id])
    print("NATURAL:", natural)
    for cfg in argv[2:]:
        use_natural = cfg.endswith("~nat")
        cfg_body = cfg.removesuffix("~nat")
        body, _, weight = cfg_body.partition("~w")
        run_prompt = natural if use_natural else prompt
        if use_natural and weight:
            run_prompt = natural_prompt(style_id, briefs[style_id], float(weight))
        mode, _, scale_tok = body.partition(":")
        embeds = None
        if weight:
            weighted = (
                run_prompt if use_natural else weight_change_clauses(run_prompt, float(weight))
            )
            embeds = levers.compel_embeds(weighted, negative)
        for seed in seeds:
            path: Path = out_dir / f"{config_name(cfg)}_s{seed}.png"
            if path.exists():
                continue
            img, secs = levers.generate_variant(
                run_prompt,
                negative,
                None if use_natural else prompt_2,  # natural prompt drives BOTH encoders
                refs,
                mode=mode,
                scale=parse_scale(scale_tok),
                seed=seed,
                weighted_prompt_embeds=embeds,
            )
            img.save(path)
            print(f"{cfg} seed {seed}: {secs:.1f}s -> {path}")
    free_sdxl_pipeline()


if __name__ == "__main__":
    main(sys.argv[1:])
