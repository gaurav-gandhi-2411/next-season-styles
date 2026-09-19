"""Final deliverable figures + selection table (H4; rebuilt for J2 p90 Gate 1 and J3/J4 judges).

Best candidate per style REGARDLESS of gate outcome (the graded artifact is a fashion deliverable):
T-shirt seed 43 and sweater seed 42 are F5's visual-QC selections (unchanged -- H3 regenerates the
underwear only); underwear is H3's seed 43 (visual QC rejected seeds 42/44/45, see `h3_score`).

`FINAL_concepts.png` is clean of QC marks and captioned from LOOKING AT THE IMAGES
(`OBSERVED_CAPTIONS`), never from the brief: a brief's `applied_changes` are prompt inputs, not
verified outputs (the underwear shows dark piping, not the briefed burgundy picot edge).

`evidence_chain.png` carries the honest per-style verdict: Gate 1 against the real within-style p90
benchmark (plus the un-gated nearest-neighbour diagnostic), Gate 2 as the MEDIAN of repeated judge
calls with its noise bound, and a human visual check. An unmeasured gate is never rendered as a
pass (`judge_repeat` logs every raw call; missing repeats stay missing).

Usage:
    uv run python -m nss.generate.h4_deliverables
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import polars as pl
from PIL import Image

from nss.generate import judge_repeat, screen_references
from nss.generate.final_concepts import load_design_briefs
from nss.generate.final_deliverables import (
    STYLE_ORDER,
    PanelData,
    build_brief_excerpt,
    build_exemplar_composite,
    build_hero_figure,
    display_name,
)
from nss.generate.h3_generate import reference_paths as h3_reference_paths
from nss.generate.h3_score import VISUAL_QC
from nss.generate.h3_score import image_path as h3_image_path
from nss.generate.vlm_judges import ATTRIBUTE_DIMENSIONS, SKILL

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
# Written by looking at each image (task J5a) -- describes what IS visible, not what was briefed.
OBSERVED_CAPTIONS: dict[str, str] = {
    T_SHIRT: (
        "Black short-sleeve crew-neck tee in a smooth, fitted jersey. Visible details: topstitched "
        "neckline and shoulder seams, a white care-label tab at the inner neck. No print."
    ),
    UNDERWEAR: (
        "Solid red high-leg brief with contrasting black binding along the waist and leg "
        "openings, red topstitching and a small black woven label at the centre front. Lace-free."
    ),
    SWEATER: (
        "Camel-beige V-neck pullover in a chunky rib knit: relaxed drop-shoulder body, ribbed "
        "cuffs and a deep ribbed hem with a curved, uneven edge. Solid colour, no pattern."
    ),
}
# What a human saw when viewing the selected image (T-shirt/sweater: viewed in task J5a).
HUMAN_CHECK: dict[str, str] = {
    T_SHIRT: "clean, coherent garment",
    UNDERWEAR: VISUAL_QC[43].removeprefix("keep: "),
    SWEATER: "clean, coherent garment",
}
# Largest single-image score difference between two judge calls ever measured in this project
# (T-shirt seed 42: 0.6375 then 0.425); an upper bound on single-call noise, not a CI.
JUDGE_NOISE_BOUND = 0.21


def _image(style_id: str, seed: int, source: str) -> Path:
    if source == "h3":
        return h3_image_path(seed)
    slug = {
        T_SHIRT: "ladieswear_t-shirt_jersey-basic_black_solid",
        SWEATER: "ladieswear_sweater_knitwear_beige_melange",
    }[style_id]
    return Path(f"data/generated/final_concepts_v3/{slug}_seed{seed}.png")


def gate2_result(image: Path) -> dict[str, Any]:
    """Gate 2 for one image from `judge_repeat`'s logged calls: per-judge median + spread.

    Gate 2 is decided only when every judge with any successful call completed all `N_REPEATS`
    calls: it then passes iff every judge's median clears ITS OWN threshold (`SKILL` joint-AND).
    A judge with 1..N-1 calls (quota ran out mid-way) is reported with its `n_calls` but Gate 2 is
    `None` (inconclusive) -- given ~0.2 single-call noise, one call cannot decide it. No calls at
    all is `None` too: unmeasured, never a pass or a fail.
    """
    repeats = judge_repeat.repeat_records()
    thresholds = judge_repeat.judge_thresholds()
    medians: dict[str, float] = {}
    complete: dict[str, list[dict[str, Any]]] = {}
    for judge in judge_repeat.CALLERS:
        recs = repeats.get((str(image), judge), [])[: judge_repeat.N_REPEATS]
        if recs and judge in thresholds:
            medians[judge] = statistics.median(r["mean_score"] for r in recs)
            complete[judge] = recs
    if not medians:
        return {"judges": [], "gate2_pass": None}
    all_complete = all(len(r) == judge_repeat.N_REPEATS for r in complete.values())
    primary = sorted(medians)[0]
    recs = complete[primary]
    means = [r["mean_score"] for r in recs]
    return {
        "judges": sorted(medians),
        "gate2_pass": (
            SKILL.fidelity_pass_from_per_judge(medians, thresholds) if all_complete else None
        ),
        "n_calls": len(recs),
        "primary_judge": primary,
        "fidelity_median": medians[primary],
        "fidelity_spread": max(means) - min(means),
        "threshold": thresholds[primary],
        "attr_medians": {
            d: statistics.median(r["scores"][d] for r in recs) for d in ATTRIBUTE_DIMENSIONS
        },
        "raw_extraction": recs[0]["raw_extraction"],
    }


def selection_rows() -> list[dict[str, Any]]:
    """One row per style: Gate 1 (p90), the un-gated NN diagnostic, Gate 2 (median of repeats)."""
    f5 = pl.read_csv("reports/tables/gate1_rescored_within_style.csv")
    h3 = pl.read_csv("reports/tables/h3_underwear_scored.csv")
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
        g2 = gate2_result(img)
        gate2 = g2["gate2_pass"]
        rows.append(
            {
                "style_id": sid,
                "seed": seed,
                "image_path": str(img),
                "clip_mean_sim": g["clip_mean_sim"],
                "clip_benchmark_p90": g["clip_benchmark"],
                "dinov2_mean_sim": g["dinov2_mean_sim"],
                "dinov2_benchmark_p90": g["dinov2_benchmark"],
                "gate1_pass": gate1,
                "clip_max_sim": g["clip_max_sim"],
                "clip_nn_benchmark": g["clip_nn_benchmark"],
                "dinov2_max_sim": g["dinov2_max_sim"],
                "dinov2_nn_benchmark": g["dinov2_nn_benchmark"],
                "nn_diagnostic_flags_copy": bool(
                    g["clip_max_sim"] > g["clip_nn_benchmark"]
                    or g["dinov2_max_sim"] > g["dinov2_nn_benchmark"]
                ),
                "judges": ",".join(g2["judges"]) or None,
                "n_judge_calls": g2.get("n_calls"),
                "fidelity_median": g2.get("fidelity_median"),
                "fidelity_spread": g2.get("fidelity_spread"),
                "fidelity_threshold": g2.get("threshold"),
                "judge_noise_bound": JUDGE_NOISE_BOUND,
                "gate2_pass": gate2,
                "human_check": HUMAN_CHECK[sid],
                "overall": (
                    "PASS"
                    if gate1 and gate2
                    else "FAIL"
                    if (not gate1 or gate2 is False)
                    else "GATE 2 INCONCLUSIVE"
                    if g2["judges"]
                    else "GATE 2 NOT MEASURED"
                ),
                "attr_medians": g2.get("attr_medians"),
                "raw_extraction": g2.get("raw_extraction"),
            }
        )
    return rows


def _gate_text(r: dict[str, Any]) -> str:
    def mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    nn = "flags a copy" if r["nn_diagnostic_flags_copy"] else "clear"
    return "\n".join(
        [
            "Gate 1 -- range check: similarity to",
            "own references must be <= p90 of",
            "similarity between distinct REAL",
            "articles of this style",
            "",
            f"CLIP   {r['clip_mean_sim']:.3f} <= {r['clip_benchmark_p90']:.3f}  "
            f"{mark(r['clip_mean_sim'] <= r['clip_benchmark_p90'])}",
            f"DINOv2 {r['dinov2_mean_sim']:.3f} <= {r['dinov2_benchmark_p90']:.3f}  "
            f"{mark(r['dinov2_mean_sim'] <= r['dinov2_benchmark_p90'])}",
            "",
            f"Gate 1: {mark(r['gate1_pass'])}",
            "",
            "Not gated -- nearest-reference check:",
            f"closest ref CLIP {r['clip_max_sim']:.3f} vs {r['clip_nn_benchmark']:.3f}",
            f"DINOv2 {r['dinov2_max_sim']:.3f} vs {r['dinov2_nn_benchmark']:.3f}: {nn}",
        ]
    )


def _fidelity_text(r: dict[str, Any]) -> str:
    if r["judges"] is None:
        return (
            "Gate 2 -- visual attribute fidelity\n\nNOT MEASURED: no judge completed\n"
            "all repeat calls (quota)."
        )
    raw = r["raw_extraction"]
    lines = [
        "Gate 2 -- visual attribute fidelity",
        f"({r['judges']} judge, blind; median of {r['n_judge_calls']} call(s))",
        "",
    ]
    for d in ATTRIBUTE_DIMENSIONS:
        lines.append(f"{d[:19]:<21}{r['attr_medians'][d]:.2f}  '{str(raw[d])[:20]}'")
    lines += [
        "",
        f"median {r['fidelity_median']:.3f} vs threshold {r['fidelity_threshold']:.3f}",
        f"range over {r['n_judge_calls']} call(s): {r['fidelity_spread']:.3f}",
        f"cross-call noise seen: up to {r['judge_noise_bound']:.2f}",
        "Gate 2: "
        + (
            "INCONCLUSIVE (quota: <3 calls)"
            if r["gate2_pass"] is None
            else "PASS"
            if r["gate2_pass"]
            else "FAIL"
        ),
    ]
    return "\n".join(lines)


def build_evidence_figure(rows: list[dict[str, Any]], refs: dict[str, list[Path]]) -> plt.Figure:
    """refs -> brief -> concept -> Gate 1 (real p90) -> fidelity + noise bound -> verdict."""
    briefs = load_design_briefs()
    fig, axes = plt.subplots(len(rows), 6, figsize=(27, 5.4 * len(rows)))
    titles = (
        "Real references",
        "Design brief (inputs, not verified)",
        "Generated concept",
        "Gate 1: vs real within-style p90",
        "Gate 2: attribute fidelity + noise",
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
            0.62,
            r["overall"],
            transform=a_v.transAxes,
            fontsize=15,
            fontweight="bold",
            va="center",
        )
        a_v.text(
            0.05,
            0.40,
            f"Human visual check:\n{r['human_check']}",
            transform=a_v.transAxes,
            fontsize=8.5,
            va="top",
        )
    for ax, t in zip(axes[0], titles, strict=True):
        ax.set_title(t, fontsize=10.5, fontweight="bold", pad=10)
    fig.suptitle(
        "Evidence chain: references -> brief -> concept -> Gate 1 (real within-style p90) -> "
        "Gate 2 (median of repeated calls + noise bound) -> verdict",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def main() -> None:
    """Write the selection table, the clean hero, and the evidence chain."""
    rows = selection_rows()
    flat = [
        {k: v for k, v in r.items() if k not in ("attr_medians", "raw_extraction")} for r in rows
    ]
    pl.DataFrame(flat, infer_schema_length=None).write_csv(SELECTION_OUT)
    panels = [
        PanelData(
            r["style_id"],
            display_name(r["style_id"]),
            OBSERVED_CAPTIONS[r["style_id"]],
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
        print(display_name(r["style_id"]), r["overall"], r["gate1_pass"], r["judges"])


if __name__ == "__main__":
    main()
