"""Generation with the winning lever configuration, for the final three.

WINNING CONFIG (from the lever experiments, `reports/tables/prompt_lever_summary.md` /
`nss.generate.levers`):

1. NATURAL-LANGUAGE PROMPT ON BOTH TEXT ENCODERS. The earlier attribute-first prompt
   ("Sweater, knitwear construction, beige melange. Novel accent: ...") with an attribute-only
   second-encoder prompt made the change clauses invisible with an image reference, and with the
   reference off it collapsed into fabric swatches (the texture words dominated). One flat sentence
   naming the garment and its briefed changes, given to both encoders, made the changes appear at
   IP scales 0.25-0.45 even from a single reference.
2. MULTI-REFERENCE CONCAT (`ip_adapter_image=[[...]]`): the reference constrains the archetype
   rather than copying one garment.
3. COMPEL WEIGHTING (1.5) on the change clauses, which restores accents lost at the higher scales.
4. PER-STYLE IP-ADAPTER SCALE chosen by a sweep, not a global 0.45.

Usage:
    uv run python -m nss.generate.concept_generation generate <style-keyword> \
        <scales,csv> <seeds,csv>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from nss.generate import final_concepts, final_registry, levers, screen_references
from nss.generate.fidelity import NON_VISUAL_GRAPHICAL_VALUES
from nss.generate.scale_sweep import free_sdxl_pipeline

OUT_ROOT = Path("data/generated/n9")
N_REFS = 8  # best-selling screened references: enough archetype signal, cheap to encode
WEIGHT = 1.5
BRIEFS_PATH = Path("reports/tables/design_briefs.json")
NEGATIVE_BASE = (
    "blurry, distorted proportions, extra limbs, warped seams, low-resolution, watermark, text "
    "overlay, logo, duplicate garments, mismatched colourway, worn by a human model, face, skin"
)

# Concrete, visually checkable changes (human design decisions, colour anchor kept).
CHANGES: dict[str, dict[str, Any]] = {
    final_registry.SWEATER: {
        "applied_changes": ["a high funnel neck collar", "dark brown contrast rib cuffs and hem"],
        "negative_extra": "v-neck, plain beige cuffs",
    },
    final_registry.DRESS: {
        "applied_changes": [
            "a square neckline with puff sleeves",
            "a wide self belt tied at the waist",
        ],
        "negative_extra": "tie neck, cap sleeves, hanger, stripes, pink",
    },
    final_registry.SUMMER: {
        "applied_changes": [
            "a triangle halter neckline with long ties",
            "thick white contrast binding",
        ],
        "negative_extra": (
            "bikini bottom, swim briefs, two-piece set, plants, leaves, sunglasses, tassel, props"
        ),
    },
    final_registry.TOP: {
        "applied_changes": ["a square neckline", "long balloon sleeves with wide ribbed cuffs"],
        "negative_extra": "crew neck, short sleeves, print, stripes",
    },
}


def natural_prompt(style_id: str, changes: list[str], weight: float | None) -> str:
    """One flat sentence: garment first, then the briefed changes (weighted for compel)."""
    _dept, product, _group, colour, pattern = (p.strip() for p in style_id.split(" || "))
    pat = (
        ""
        if pattern == "Solid" or pattern in NON_VISUAL_GRAPHICAL_VALUES
        else f"{pattern.lower()} "
    )
    parts = [f"({c}){weight}" if weight else c for c in changes]
    return (
        f"flat-lay product photo of a {colour.lower()} {pat}{product.lower()} with "
        f"{' and '.join(parts)}, plain light background"
    )


def brief_for(style_id: str) -> dict[str, Any]:
    """A minimal brief in `design_briefs.json`'s shape (negative prompt via the rule table)."""
    spec = CHANGES[style_id]
    return {
        "style_id": style_id,
        "silhouette": "",
        "fabric_and_hand": "",
        "colour_direction": "",
        "detail_and_graphic_treatment": "",
        "applied_changes": spec["applied_changes"],
        "negative_prompt": f"{NEGATIVE_BASE}, {spec['negative_extra']}",
    }


def load_refs() -> dict[str, list[Path]]:
    """Screened references per style: the autumn/winter base plus the summer style's own base."""
    refs = screen_references.load_screened_references()
    summer = Path("reports/tables/reference_base_widened_summer.csv")
    if summer.exists():
        refs.update(screen_references.load_screened_references(summer))
    return refs


def style_id_for(keyword: str) -> str:
    """The final style whose key contains `keyword` (case-insensitive)."""
    return next(
        s
        for s in (*final_registry.STYLE_ORDER, final_registry.SUMMER)
        if keyword.lower() in s.lower()
    )


def image_path(style_id: str, scale: float, seed: int, out_root: Path = OUT_ROOT) -> Path:
    """Where `generate` / `generate_candidate` write one candidate."""
    slug = final_concepts._slugify(style_id)
    return out_root / slug / f"s{scale:.2f}_seed{seed}.png"


def generate_candidate(
    style_id: str,
    scale: float,
    seed: int,
    out_root: Path = OUT_ROOT,
    embeds: dict[str, Any] | None = None,
) -> tuple[Path, float]:
    """One candidate in the configuration that made the deliverables (GPU). Returns (path, secs).

    Concat mode over the style's `N_REFS` best screened references, the brief's negative prompt
    (generic rules + the style's extras) and the compel-weighted natural prompt. `embeds` may be
    passed to reuse one text encoding across candidates of the same style; the file is
    overwritten if it exists, so callers that must not overwrite check first (`generate` does).
    """
    brief = brief_for(style_id)
    _prompt, negative = final_concepts.build_generation_spec(style_id, {**_pad(brief)})
    refs = load_refs()[style_id][:N_REFS]
    if embeds is None:
        embeds = levers.compel_embeds(
            natural_prompt(style_id, brief["applied_changes"], WEIGHT), negative
        )
    path = image_path(style_id, scale, seed, out_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    img, secs = levers.generate_variant(
        "",
        negative,
        None,
        refs,
        mode="concat",
        scale=scale,
        seed=seed,
        weighted_prompt_embeds=embeds,
    )
    img.save(path)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "style_id": style_id,
                "scale": scale,
                "seed": seed,
                "weight": WEIGHT,
                "mode": "concat",
                "n_refs": len(refs),
                "refs": [str(p) for p in refs],
                "prompt": natural_prompt(style_id, brief["applied_changes"], WEIGHT),
                "negative": negative,
                "changes": brief["applied_changes"],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return path, secs


def generate(style_id: str, scales: list[float], seeds: list[int]) -> None:
    """Generate every (scale, seed) candidate that does not exist yet (GPU)."""
    brief = brief_for(style_id)
    _prompt, negative = final_concepts.build_generation_spec(style_id, {**_pad(brief)})
    refs = load_refs()[style_id][:N_REFS]
    embeds = levers.compel_embeds(
        natural_prompt(style_id, brief["applied_changes"], WEIGHT), negative
    )
    print("PROMPT:", natural_prompt(style_id, brief["applied_changes"], None))
    print("NEG:", negative)
    print("REFS:", [p.name for p in refs])
    for scale in scales:
        for seed in seeds:
            if image_path(style_id, scale, seed).exists():
                continue
            path, secs = generate_candidate(style_id, scale, seed, embeds=embeds)
            print(f"scale {scale} seed {seed}: {secs:.1f}s -> {path}")
    free_sdxl_pipeline()


def _pad(brief: dict[str, Any]) -> dict[str, Any]:
    """Fill the fields `build_generation_spec` requires (its prompt is not used here)."""
    filled = dict(brief)
    for key in (
        "silhouette",
        "fabric_and_hand",
        "colour_direction",
        "detail_and_graphic_treatment",
    ):
        filled[key] = filled[key] or "plain."
    return filled


def main(argv: list[str]) -> None:
    """CLI: `generate <keyword> <scales> <seeds>`."""
    if argv[0] != "generate":
        raise SystemExit(__doc__)
    generate(
        style_id_for(argv[1]),
        [float(x) for x in argv[2].split(",")],
        [int(x) for x in argv[3].split(",")],
    )


if __name__ == "__main__":
    main(sys.argv[1:])
