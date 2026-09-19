"""Regenerate the underwear style ONLY, with a pattern-verified solid reference set (task H3).

Everything except the reference set is F5's construction, unchanged: `ip_adapter_scale=0.45`, the
same `design_briefs.json` prompt via `build_generation_spec` (incl. the underwear negative-prompt
strengthening) and `build_prompt_2`, seeds 42-45, no retries.

REFERENCE SET (visually verified on a contact sheet -- the `detail_desc` screen in
`h3_underwear_refs` alone was NOT sufficient: 666651009/666651012 are plainly lace although their
descriptions say "microfibre"; 638521006's 4-pack includes lace briefs). Kept: the four visually
plain articles below. `references[0]` is the ONLY image IP-Adapter conditions on
(`backends.py` note 1); it is set to 592614002 -- the one single-garment, red, solid image --
deliberately OVERRIDING the project's best-selling-first convention (that would pick a
multi-colour 3-pack stack, which would import three garments and off-style colours into every
candidate).

Usage:
    uv run python -m nss.generate.h3_generate
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from nss.generate.final_concepts import (
    build_generation_spec,
    build_prompt_2,
    generate_candidates_for_style,
    load_design_briefs,
)
from nss.generate.h3_underwear_refs import IMAGES_DIR, MANIFEST_PATH, STYLE_ID
from nss.generate.scale_sweep import free_sdxl_pipeline

OUTPUT_DIR = Path("data/generated/final_concepts_h3")
SEEDS: tuple[int, ...] = (42, 43, 44, 45)
IP_ADAPTER_SCALE = 0.45  # unchanged from F5 -- only the reference set is the H3 variable
# (article_id, verdict) from visual inspection of the contact sheet, in conditioning order.
VISUAL_VERDICT: dict[int, str] = {
    592614002: "keep: single red garment, solid satin-look, no pattern (metal rings at sides)",
    634669009: "keep: three plain ribbed thongs, no lace/print (stack; off-colours)",
    562613009: "keep: three plain hipsters, one satin waistband (stack; off-colours)",
    803969001: "keep: 7-pack of plain basics (stack; off-colours)",
    638521006: "reject: 4-pack contains lace briefs",
    666651009: "reject: lace hipster despite plain description",
    666651012: "reject: lace hipster despite plain description",
}


def reference_paths() -> list[Path]:
    """The verified plain-solid references, conditioning image first."""
    return [IMAGES_DIR / f"{i:010d}.jpg" for i, v in VISUAL_VERDICT.items() if v.startswith("keep")]


def annotate_manifest() -> None:
    """Add the visual verdict + reference order to the H3 manifest CSV."""
    order = {
        i: n for n, i in enumerate(i for i, v in VISUAL_VERDICT.items() if v.startswith("keep"))
    }
    df = pl.read_csv(MANIFEST_PATH).with_columns(
        pl.col("article_id")
        .replace_strict(VISUAL_VERDICT, default="unreviewed")
        .alias("visual_verdict"),
        pl.col("article_id").replace_strict(order, default=None).alias("reference_order"),
    )
    df.write_csv(MANIFEST_PATH)


def main() -> None:
    """Generate 4 underwear candidates; free VRAM afterwards."""
    refs = reference_paths()
    assert len(refs) >= 3, "need >= 3 verified solid references"
    annotate_manifest()
    brief = load_design_briefs()[STYLE_ID]
    prompt, negative_prompt = build_generation_spec(STYLE_ID, brief)
    print(f"prompt: {prompt!r}\nnegative: {negative_prompt!r}\nrefs: {[p.name for p in refs]}")
    candidates = generate_candidates_for_style(
        STYLE_ID,
        prompt,
        negative_prompt,
        refs,
        seeds=SEEDS,
        ip_adapter_scale=IP_ADAPTER_SCALE,
        output_dir=OUTPUT_DIR,
        prompt_2=build_prompt_2(STYLE_ID),
        negative_prompt_2=negative_prompt,
    )
    print("VRAM before/after free (GB):", free_sdxl_pipeline())
    for c in candidates:
        print(c.seed, c.image_path)


if __name__ == "__main__":
    main()
