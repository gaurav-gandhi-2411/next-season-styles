"""Final deliverable figures + selection table, rebuilt under H1's Gate 1 and H2's checklist (H4).

Best candidate per style REGARDLESS of gate outcome (the graded artifact is a fashion deliverable):
T-shirt seed 43 and sweater seed 42 are F5's visual-QC selections (unchanged -- H3 regenerates the
underwear only); underwear is H3's seed 43 (visual QC rejected seeds 42/44/45, see `h3_score`).
`FINAL_concepts.png` is clean of QC marks (reuses `final_deliverables.build_hero_figure`);
`evidence_chain.png` carries the honest per-style verdict, including "Gate 2 not measured" where a
judge was quota-blocked -- an unmeasured gate is never rendered as a pass.

Usage:
    uv run python -m nss.generate.h4_deliverables
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import polars as pl
from PIL import Image

from nss.generate import h2_rejudge, screen_references
from nss.generate.final_concepts import load_design_briefs
from nss.generate.final_deliverables import (
    STYLE_ORDER,
    PanelData,
    build_brief_excerpt,
    build_exemplar_composite,
    build_hero_figure,
    build_rationale,
    display_name,
)
from nss.generate.h3_generate import reference_paths as h3_reference_paths
from nss.generate.h3_score import image_path as h3_image_path
from nss.generate.vlm_judges import ATTRIBUTE_DIMENSIONS

HERO_OUT = Path("reports/figures/FINAL_concepts.png")
EVIDENCE_OUT = Path("reports/figures/evidence_chain.png")
SELECTION_OUT = Path("reports/tables/final_selection_h4.csv")
T_SHIRT, UNDERWEAR, SWEATER = STYLE_ORDER
# (style -> (seed, source)) -- source picks which scored table carries Gate 1 for that image.
SELECTED: dict[str, tuple[int, str]] = {
    T_SHIRT: (43, "f5"),
    UNDERWEAR: (43, "h3"),
    SWEATER: (42, "f5"),
}


def _image(style_id: str, seed: int, source: str) -> Path:
    if source == "h3":
        return h3_image_path(seed)
    slug = {
        T_SHIRT: "ladieswear_t-shirt_jersey-basic_black_solid",
        SWEATER: "ladieswear_sweater_knitwear_beige_melange",
    }[style_id]
    return Path(f"data/generated/final_concepts_v3/{slug}_seed{seed}.png")


def selection_rows() -> list[dict[str, Any]]:
    """One row per style: Gate 1 (new benchmark), Gate 2 (H2 checklist, per available judge)."""
    f5 = pl.read_csv("reports/tables/gate1_rescored_within_style.csv")
    h3 = pl.read_csv("reports/tables/h3_underwear_scored.csv")
    judge = pl.read_csv(h2_rejudge.RESCORE_PATH)
    rows = []
    for sid in STYLE_ORDER:
        seed, src = SELECTED[sid]
        img = _image(sid, seed, src)
        g = (
            h3.filter(pl.col("seed") == seed).to_dicts()[0]
            if src == "h3"
            else f5.filter((pl.col("style_id") == sid) & (pl.col("seed") == seed)).to_dicts()[0]
        )
        gate1 = bool(g["gate1_pass"] if src == "h3" else g["gate1_new_pass"])
        jr = judge.filter((pl.col("image_path") == str(img)) & pl.col("available")).to_dicts()
        gate2: bool | None = all(r["pass_after"] for r in jr) if jr else None
        rows.append(
            {
                "style_id": sid,
                "seed": seed,
                "image_path": str(img),
                "clip_mean_sim": g["clip_mean_sim"],
                "clip_benchmark": g["clip_benchmark"],
                "dinov2_mean_sim": g["dinov2_mean_sim"],
                "dinov2_benchmark": g["dinov2_benchmark"],
                "gate1_pass": gate1,
                "n_judges": len(jr),
                "fidelity_3attr": (sum(r["fidelity_after_3attr"] for r in jr) / len(jr))
                if jr
                else None,
                "gate2_pass": gate2,
                "overall": (
                    "PASS"
                    if gate1 and gate2
                    else "FAIL"
                    if (not gate1 or gate2 is False)
                    else "GATE 2 NOT MEASURED"
                ),
                "scores_json": jr[0]["scores_json"] if jr else None,
                "raw_extraction_json": jr[0]["raw_extraction_json"] if jr else None,
            }
        )
    return rows


def _gate_text(r: dict[str, Any]) -> str:
    def mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    return "\n".join(
        [
            "Gate 1 -- similarity to refs vs real",
            "within-style benchmark (median of",
            "distinct real article pairs)",
            "",
            f"CLIP   {r['clip_mean_sim']:.3f} <= {r['clip_benchmark']:.3f}  "
            f"{mark(r['clip_mean_sim'] <= r['clip_benchmark'])}",
            f"DINOv2 {r['dinov2_mean_sim']:.3f} <= {r['dinov2_benchmark']:.3f}  "
            f"{mark(r['dinov2_mean_sim'] <= r['dinov2_benchmark'])}",
            "",
            f"Gate 1: {mark(r['gate1_pass'])}",
        ]
    )


def _fidelity_text(r: dict[str, Any]) -> str:
    if r["n_judges"] == 0:
        return (
            "Gate 2 -- visual attribute fidelity\n\nNOT MEASURED: every VLM judge\n"
            "quota-blocked (Groq TPD /\nGemini daily cap)."
        )
    scores = json.loads(r["scores_json"])
    raw = json.loads(r["raw_extraction_json"])
    lines = ["Gate 2 -- visual attribute fidelity", "(Groq judge, blind, 3 attributes)", ""]
    for d in ATTRIBUTE_DIMENSIONS:
        lines.append(f"{d[:19]:<21}{scores[d]:.2f}  '{str(raw[d])[:22]}'")
    lines += [
        "",
        f"mean {r['fidelity_3attr']:.3f}   Gate 2: {'PASS' if r['gate2_pass'] else 'FAIL'}",
    ]
    return "\n".join(lines)


def build_evidence_figure(rows: list[dict[str, Any]], refs: dict[str, list[Path]]) -> plt.Figure:
    """refs -> brief -> concept -> Gate 1 vs real benchmark -> per-attribute fidelity -> verdict."""
    briefs = load_design_briefs()
    fig, axes = plt.subplots(len(rows), 6, figsize=(27, 5.4 * len(rows)))
    titles = (
        "Real references",
        "Design brief -- kept vs changed",
        "Generated concept",
        "Gate 1: vs within-style benchmark",
        "Gate 2: per-attribute fidelity",
        "Verdict",
    )
    for i, r in enumerate(rows):
        a_ref, a_brief, a_img, a_g1, a_g2, a_v = axes[i]
        a_ref.imshow(build_exemplar_composite(refs[r["style_id"]]))
        a_ref.set_xticks([])
        a_ref.set_yticks([])
        a_ref.set_ylabel(display_name(r["style_id"]), fontsize=11, fontweight="bold")
        a_img.imshow(Image.open(r["image_path"]))
        a_img.axis("off")
        for ax, txt in (
            (a_brief, build_brief_excerpt(briefs[r["style_id"]])),
            (a_g1, _gate_text(r)),
            (a_g2, _fidelity_text(r)),
        ):
            ax.axis("off")
            ax.text(
                0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=7.6, family="monospace"
            )
        a_v.axis("off")
        a_v.text(
            0.05,
            0.6,
            r["overall"],
            transform=a_v.transAxes,
            fontsize=15,
            fontweight="bold",
            va="center",
        )
    for ax, t in zip(axes[0], titles, strict=True):
        ax.set_title(t, fontsize=10.5, fontweight="bold", pad=10)
    fig.suptitle(
        "Evidence chain: references -> brief -> concept -> Gate 1 (real within-style "
        "benchmark) -> Gate 2 -> verdict",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def main() -> None:
    """Write the selection table, the clean hero, and the evidence chain."""
    rows = selection_rows()
    pl.DataFrame(rows, infer_schema_length=None).write_csv(SELECTION_OUT)
    briefs = load_design_briefs()
    panels = [
        PanelData(
            r["style_id"],
            display_name(r["style_id"]),
            build_rationale(briefs[r["style_id"]]),
            Path(r["image_path"]),
            r,
            r["overall"] == "PASS",
        )
        for r in rows
    ]
    hero = build_hero_figure(panels)
    hero.savefig(HERO_OUT, dpi=150, bbox_inches="tight")
    plt.close(hero)
    refs = screen_references.load_screened_references()
    refs[UNDERWEAR] = h3_reference_paths()
    ev = build_evidence_figure(rows, refs)
    ev.savefig(EVIDENCE_OUT, dpi=150, bbox_inches="tight")
    plt.close(ev)
    for r in rows:
        print(display_name(r["style_id"]), r["overall"], r["gate1_pass"], r["n_judges"])


if __name__ == "__main__":
    main()
