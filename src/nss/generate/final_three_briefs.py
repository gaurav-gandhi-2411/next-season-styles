"""Design briefs for the re-selected final three, with concrete, token-safe change clauses.

WHY: the earlier final prompts were attribute-only ("T-shirt, jersey basic construction, black
solid."). `prompt_budget.fit_prompt_to_token_budget` drops novelty clauses FIRST, and the long
descriptive clause (silhouette/fabric/colour/surface + photography boilerplate, ~100 tokens) never
fit beside them, so BOTH were dropped: the briefed design changes never reached the generator and
nothing visible changed. The pipeline mechanics are unchanged here; what changes is the brief DATA:
short, concrete fields so that every change clause survives the 77-token budget
(`assert_all_clauses_survive` checks that with the real CLIP tokenizer: zero clauses dropped).

Each attempt is a human design decision, recorded here, made per style: colour anchor kept (the
attribute that won), and two concrete, visually checkable changes drawn from the brief's own axes
(trim/construction detail, small proportion tweak within the category's normal range).

Usage:
    uv run python -m nss.generate.final_three_briefs <attempt>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import polars as pl

BRIEFS_PATH = Path("reports/tables/design_briefs.json")
TSHIRT = "Ladieswear || T-shirt || Jersey Basic || Black || Solid"
SWEATER = "Ladieswear || Sweater || Knitwear || Beige || Melange"
DRESS = "Ladieswear || Dress || Dresses Ladies || Red || Solid"

# attempt -> style -> concise fields. Colour anchor unchanged; changes concrete and visible.
ATTEMPTS: dict[int, dict[str, dict[str, Any]]] = {
    1: {
        TSHIRT: {
            "silhouette": "boxy.",
            "fabric_and_hand": "jersey.",
            "colour_direction": "black.",
            "detail_and_graphic_treatment": "plain.",
            "applied_changes": [
                "cropped dropped-shoulder fit",
                "light grey ribbed neckband",
            ],
        },
        SWEATER: {
            "silhouette": "relaxed.",
            "fabric_and_hand": "melange wool knit.",
            "colour_direction": "beige.",
            "detail_and_graphic_treatment": "flecked melange.",
            "applied_changes": [
                "funnel neck collar",
                "dark brown ribbed cuffs and hem",
            ],
        },
        DRESS: {
            "silhouette": "midi, fitted waist.",
            "fabric_and_hand": "crepe.",
            "colour_direction": "red.",
            "detail_and_graphic_treatment": "plain.",
            "applied_changes": [
                "puff sleeves, square neckline",
                "self tie belt at waist",
            ],
        },
    },
    # Attempt 2 (after attempt 1 showed NONE of the briefed changes in any of 12 images: the
    # IP-Adapter reference structure dominates a soft text hint): decisive wording, the unwanted
    # reference feature named in a style-specific negative prompt, colour anchor stated firmly.
    2: {
        TSHIRT: {
            "silhouette": "oversized, boxy.",
            "fabric_and_hand": "heavy jersey.",
            "colour_direction": "solid black body.",
            "detail_and_graphic_treatment": "plain.",
            "applied_changes": [
                "oversized cropped boxy fit, wide sleeves",
                "white contrast neckband and sleeve bands",
            ],
            "negative_extra": "grey body, fitted slim tee",
        },
        SWEATER: {
            "silhouette": "oversized.",
            "fabric_and_hand": "marled wool knit.",
            "colour_direction": "beige.",
            "detail_and_graphic_treatment": "marl flecks.",
            "applied_changes": [
                "high funnel neck collar",
                "dark brown contrast rib cuffs and hem",
            ],
            "negative_extra": "v-neck, plain beige cuffs",
        },
        DRESS: {
            "silhouette": "midi, fitted waist.",
            "fabric_and_hand": "crepe.",
            "colour_direction": "red.",
            "detail_and_graphic_treatment": "plain.",
            "applied_changes": [
                "square neckline with puff sleeves",
                "wide self belt with tie at waist",
            ],
            "negative_extra": "tie neck, cap sleeves, hanger, stripes, pink",
        },
    },
}
NEGATIVE_FALLBACK = (
    "blurry, distorted proportions, extra limbs, warped seams, low-resolution, watermark, text "
    "overlay, logo, duplicate garments, mismatched colourway, worn by a human model, face, skin"
)


def base_briefs() -> dict[str, dict[str, Any]]:
    """Existing curated briefs keyed by style (T-shirt, sweater) plus a generated dress brief."""
    from nss.generate import build_design_briefs

    existing = {b["style_id"]: b for b in json.loads(BRIEFS_PATH.read_text(encoding="utf-8"))}
    out = {k: existing[k] for k in (TSHIRT, SWEATER) if k in existing}
    if DRESS not in out:
        generated = build_design_briefs.build_all_design_briefs(
            top_styles_path=Path("reports/tables/top_styles_final_three.csv"),
            shap_verdict_path=Path("reports/tables/final_three_shap_verdict.csv"),
            exemplar_images_path=Path("reports/tables/exemplar_images_final_three.csv"),
        )
        out.update({b["style_id"]: b for b in generated if b["style_id"] == DRESS})
    return out


def build(attempt: int) -> list[dict[str, Any]]:
    """Briefs for `attempt`: base brief + that attempt's concise fields and concrete changes."""
    briefs = []
    for style, fields in ATTEMPTS[attempt].items():
        b = dict(base_briefs()[style])
        extra = fields.get("negative_extra")
        b.update({k: v for k, v in fields.items() if k != "negative_extra"})
        base = b.get("negative_prompt") or NEGATIVE_FALLBACK
        b["negative_prompt"] = f"{base}, {extra}" if extra else base
        b["rendered_prompt"] = ""  # unused: build_generation_spec re-derives from the fields
        b["design_attempt"] = attempt
        briefs.append(b)
    return briefs


def assert_all_clauses_survive(briefs: list[dict[str, Any]]) -> None:
    """Every change clause must reach the model: zero clauses dropped by the 77-token budget."""
    from nss.generate import final_concepts as fc
    from nss.generate import prompt_budget

    for b in briefs:
        attrs = fc._parse_generic_attributes(b["style_id"])
        mandatory = fc._build_attribute_clause(attrs)
        novelty = [f"Novel accent: {c}." for c in b["applied_changes"]]
        desc = [
            f"Silhouette: {b['silhouette']} Fabric: {b['fabric_and_hand']} Colour: "
            f"{b['colour_direction']} Surface treatment: {b['detail_and_graphic_treatment']} "
            "Product photography of the garment itself, clean studio background, even lighting, "
            "no styling props."
        ]
        _, n_tokens, dropped = prompt_budget.fit_prompt_to_token_budget(mandatory, novelty, desc)
        if dropped:
            raise ValueError(f"{b['style_id']}: budget dropped {dropped} ({n_tokens} tokens)")


def main() -> None:
    """Write `design_briefs.json` for the given attempt after verifying every clause survives."""
    attempt = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    briefs = build(attempt)
    assert_all_clauses_survive(briefs)
    BRIEFS_PATH.write_text(json.dumps(briefs, indent=2), encoding="utf-8")
    from nss.generate import final_concepts as fc

    for b in briefs:
        p, n = fc.build_generation_spec(b["style_id"], b)
        print(b["style_id"], "\n  PROMPT:", p, "\n  NEG:", n[:80], "...")
    _ = pl  # keep polars import explicit for callers extending this module


if __name__ == "__main__":
    main()
