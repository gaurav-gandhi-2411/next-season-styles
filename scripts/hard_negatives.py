"""J6: hard negatives for the gate ablation (right style, but not a valid concept for its brief).

Rule pre-registered in `reports/v3/PREREGISTRATION.md` (section J, commit bcd1357). Three classes,
all truth REJECT, for the four styles in `ALL_SELECTED`:

- H1 wrong attribute: the M3 coral-pink "red" dress (both M3 attempts, seed 45) plus one
  production-config generation per style with the colour/pattern replaced in the prompt;
- H2 briefed change absent: production config, the briefed changes left out of the prompt and added
  to the negative prompt;
- H3 averaged garment: the pixel mean of the style's 8 best screened references.

    uv run --no-sync python scripts/hard_negatives.py generate   # GPU, $0
    uv run --no-sync python scripts/hard_negatives.py score      # MCP score_concept, CPU
    uv run --no-sync python scripts/hard_negatives.py analyse    # ablation tables
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_demo import _call_tool, parse_allowlist  # noqa: E402
from agent_eval import build_cases  # noqa: E402

from nss.generate import agent_eval as ae  # noqa: E402
from nss.generate import concept_generation as cg  # noqa: E402
from nss.generate import final_concepts, levers  # noqa: E402
from nss.generate.final_selection_figures import ALL_SELECTED  # noqa: E402
from nss.generate.scale_sweep import free_sdxl_pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "reports" / "tables"
OUT = Path("data/generated/hard_negatives")
CACHE = TABLES / "v3_hard_negatives_scores.jsonl"
LABELS = TABLES / "v3_hard_negatives_labels.json"
SCALE, SEED = 0.35, 42
M3 = "data/generated/final_concepts_m3/attempt{a}/ladieswear_dress_dresses-ladies_red_solid_seed45.png"  # noqa: E501
SWAP = {  # style -> the wrong colour / pattern put in the prompt (the changes are kept)
    "Sweater": "navy blue",
    "Dress": "emerald green",
    "Top": "black",
    "Bikini top": "plain solid blue",
}
H1, H2, H3 = "H1_wrong_attribute", "H2_change_absent", "H3_averaged_garment"


def _short(style: str) -> str:
    return style.split(" || ")[1]


def _parts(style: str) -> tuple[str, str]:
    """(colour + pattern phrase as production writes it, lower-case product type)."""
    _dept, product, _group, colour, pattern = (p.strip() for p in style.split(" || "))
    pat = (
        ""
        if pattern == "Solid" or pattern in cg.NON_VISUAL_GRAPHICAL_VALUES
        else f"{pattern.lower()} "
    )
    return f"{colour.lower()} {pat}", product.lower()


def _prompts(style: str, cls: str) -> tuple[str, str]:
    """(weighted prompt, negative prompt) for a generated class."""
    changes = cg.CHANGES[style]["applied_changes"]
    brief = cg.brief_for(style)
    _p, negative = final_concepts.build_generation_spec(style, {**cg._pad(brief)})
    phrase, product = _parts(style)
    if cls == H1:
        weighted = " and ".join(f"({c}){cg.WEIGHT}" for c in changes)
        return (
            f"flat-lay product photo of a {SWAP[_short(style)]} {product} with {weighted}, "
            "plain light background",
            negative,
        )
    return (
        f"flat-lay product photo of a {phrase}{product}, plain light background",
        f"{negative}, {', '.join(changes)}",
    )


def generate() -> None:
    """Generate H1 (colour swap) and H2 (brief absent) images and build the H3 pixel means."""
    OUT.mkdir(parents=True, exist_ok=True)
    refs_all = cg.load_refs()
    for style in ALL_SELECTED:
        refs = refs_all[style][: cg.N_REFS]
        mean = np.mean(
            [np.asarray(Image.open(p).convert("RGB").resize((1024, 1024)), float) for p in refs],
            axis=0,
        )
        Image.fromarray(mean.astype(np.uint8)).save(OUT / f"{H3}_{_short(style)}.png")
        for cls in (H1, H2):
            path = OUT / f"{cls}_{_short(style)}.png"
            if path.exists():
                continue
            prompt, negative = _prompts(style, cls)
            # compel crashes (no `empty_z` on its multi-encoder provider) when the negative is the
            # longer one, as it is for H2: use the plain prompt there
            embeds = levers.compel_embeds(prompt, negative) if cls == H1 else None
            img, secs = levers.generate_variant(
                "" if embeds else prompt,
                negative,
                None,
                refs,
                mode="concat",
                scale=SCALE,
                seed=SEED,
                weighted_prompt_embeds=embeds,
            )
            img.save(path)
            path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "style": style,
                        "class": cls,
                        "prompt": prompt,
                        "negative": negative,
                        "scale": SCALE,
                        "seed": SEED,
                        "secs": secs,
                    },
                    indent=1,
                ),
                encoding="utf-8",
            )
            print(f"{path.name}: {secs:.1f}s", flush=True)
    free_sdxl_pipeline()


def hard_cases() -> list[dict[str, Any]]:
    """Every hard-negative case: name, class, style, image path."""
    cases = [
        {
            "name": f"m3_coral_dress_attempt{a}",
            "cls": H1,
            "style": cg.final_registry.DRESS,
            "path": M3.format(a=a),
        }
        for a in (1, 2)
    ]
    for style in ALL_SELECTED:
        for cls in (H1, H2, H3):
            cases.append(
                {
                    "name": f"{cls}_{_short(style)}",
                    "cls": cls,
                    "style": style,
                    "path": (OUT / f"{cls}_{_short(style)}.png").as_posix(),
                }
            )
    return cases


async def score() -> None:
    """Score every hard case through the MCP `score_concept` tool (CPU); resumable."""
    allow = {p.stem: parse_allowlist(p.read_text("utf-8")) for p in (ROOT / "agents").glob("*.md")}
    done = (
        {json.loads(x)["name"] for x in CACHE.read_text("utf-8").splitlines() if x}
        if CACHE.exists()
        else set()
    )
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "nss.mcp_server"],
        cwd=str(ROOT),
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"},
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        for case in hard_cases():
            if case["name"] in done:
                continue
            args = {
                "concept_path": case["path"],
                "style_key": case["style"],
                "include_fidelity": True,
                "changes": list(cg.CHANGES[case["style"]]["applied_changes"]),
            }
            result, _ = await _call_tool(session, allow, "critic", "score_concept", args)
            with CACHE.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({**case, "score": result}) + "\n")
            print(f"{case['name']}: {result['verdict'][:70]}", flush=True)


def analyse() -> None:
    """Per-case gate results and the ablation per class (hard negatives and the H3 strata)."""
    scored = [json.loads(x) for x in CACHE.read_text("utf-8").splitlines() if x]
    excluded = set(json.loads(LABELS.read_text("utf-8"))["excluded"]) if LABELS.exists() else set()
    hard = []
    for r in scored:
        passes, clone_ok = ae.passes_from_score(r["score"])
        hard.append(ae.Case(r["name"], r["cls"], False, passes, clone_ok))
    kept = [c for c in hard if c.name not in excluded]
    base, _d, _p = build_cases()
    rows = []
    for c in [*hard]:
        v = ae.decide(c.passes, c.clone_ok)
        rows.append(
            {
                "name": c.name,
                "class": c.stratum,
                "excluded": c.name in excluded,
                **{g: c.passes[g] for g in ae.GATES},
                "verdict": v,
                "failing": ",".join(ae.failing(c.passes, clone_ok=c.clone_ok)),
            }
        )
    pl.DataFrame(rows).write_csv(TABLES / "v3_hard_negatives_cases.csv")
    pl.DataFrame(ae.ablate(kept)).write_csv(TABLES / "v3_hard_negatives_ablation.csv")
    # H3 correction: the M3 dress was a P2 positive but is coral-pink (a wrong-attribute case)
    p1 = [c for c in base if c.stratum == "P1_approved"]
    p2 = [c for c in base if c.stratum == "P2_coherent"]
    neg = [c for c in base if c.stratum in ("N1_malformed", "N2_swatch", "N3_clone")]
    rec = []
    for label, pos in (
        ("P1", p1),
        ("P1+P2 (as counted in H3)", p1 + p2),
        ("P1+P2 without m3_dress", p1 + [c for c in p2 if c.name != "m3_dress"]),
    ):
        for extra_label, extra in (("N1-N3", []), ("N1-N3 + hard negatives", kept)):
            conf = ae.confusion([*pos, *neg, *extra])
            pr = ae.precision_recall(conf)
            rec.append(
                {
                    "positives": label,
                    "negatives": extra_label,
                    **{k: v for k, v in pr.items() if not k.endswith("_ci")},
                    "recall_lo": pr["recall_ci"][0],
                    "recall_hi": pr["recall_ci"][1],
                }
            )
    pl.DataFrame(rec).write_csv(TABLES / "v3_hard_negatives_recall.csv")
    print(json.dumps({"cases": len(hard), "excluded": sorted(excluded)}))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "generate":
        generate()
    elif mode == "score":
        asyncio.run(score())
    else:
        analyse()
