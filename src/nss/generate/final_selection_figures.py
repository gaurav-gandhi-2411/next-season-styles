"""Final deliverable figures + selection table (rebuilt for the N9 concepts, Gate 3 and integrity).

Best candidate per style from task N9's 8-seed runs, chosen by: passes every automatic gate ->
most briefed changes visible -> highest fidelity, then confirmed by a human looking at the image
(`HUMAN_CHECK`). The graded artifact is a fashion deliverable, so a style whose every candidate
fails still gets its best candidate shown, with its real verdict.

`FINAL_concepts.png` is clean of QC marks and captioned from LOOKING AT THE IMAGES
(`OBSERVED_CAPTIONS`), never from the brief: a brief's `applied_changes` are prompt inputs, not
verified outputs.

`evidence_chain.png` carries the honest per-style verdict: Gate 1 and 1b against the real
within-style p90 limits, the integrity floor, Gate 2 (each local judge vs its calibrated threshold),
Gate 3 (are the briefed changes visible, per judge), the closed-loop forecast (task N8) and a human
visual check. An unmeasured gate is never rendered as a pass.

Usage:
    uv run python -m nss.generate.final_selection_figures
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import polars as pl
from PIL import Image

from nss.generate import concept_generation, final_registry
from nss.generate.concept_scoring import ADVISORY_JUDGES
from nss.generate.final_deliverables import (
    STYLE_ORDER,
    PanelData,
    build_exemplar_composite,
    build_hero_figure,
    display_name,
)

HERO_OUT = Path("reports/figures/FINAL_concepts.png")
EVIDENCE_OUT = Path("reports/figures/evidence_chain.png")
SELECTION_OUT = Path("reports/tables/final_selection.csv")
SCORED = Path("reports/tables/candidates_scored.csv")
FORECAST = Path("reports/tables/concept_forecast_retrieval.csv")  # retrieval closed loop (Q2)
SWEATER, DRESS, TOP = STYLE_ORDER
N9 = Path("data/generated/n9")
# style -> chosen candidate image (an N9 output at the per-style scale from the sweep)
SELECTED: dict[str, Path] = {
    SWEATER: N9 / "ladieswear_sweater_knitwear_beige_melange" / "s0.35_seed45.png",
    DRESS: N9 / "ladieswear_dress_dresses-ladies_red_solid" / "s0.35_seed44.png",
    TOP: N9 / "ladieswear_top_jersey-basic_white_solid" / "s0.35_seed42.png",
}
# The Summer (forecast origin 2020-06-01) concept, chosen from its own 8-seed run.
SUMMER_SELECTED = N9 / "ladieswear_bikini-top_swimwear_orange_all-over-pattern" / "s0.35_seed48.png"
# Written by looking at each image -- describes what IS visible, not what was briefed.
OBSERVED_CAPTIONS: dict[str, str] = {
    final_registry.SUMMER: (
        "Orange and white all-over print halter bikini top with a plunging front, thick white "
        "binding along the edges and a tie at the neck."
    ),
    SWEATER: "Ribbed beige knit sweater, dark-brown funnel neck, dark-brown cuffs and hem.",
    DRESS: "Red midi dress: square neckline, large puff sleeves, self belt tied in a bow.",
    TOP: "White long-sleeve top: square neckline, full balloon sleeves gathered at the cuff.",
}
# What a human saw, including whether each briefed change shows.
HUMAN_CHECK: dict[str, str] = {
    final_registry.SUMMER: (
        "Halter neckline with ties and thick white binding both visible. Orange all-over print, "
        "a single top on a plain background: no bottom, no props. One coherent garment."
    ),
    SWEATER: (
        "Both briefed changes visible (funnel neck; dark-brown rib cuffs and hem). Body is a light "
        "beige, only faintly heathered. One coherent garment."
    ),
    DRESS: (
        "All briefed changes visible (square neckline with puff sleeves; wide self belt tied at "
        "the waist). Solid red, plain flat-lay. One coherent garment."
    ),
    TOP: (
        "Square neckline and balloon sleeves visible; the cuffs are narrower than the briefed "
        "'wide ribbed' cuffs. Clean flat-lay, one coherent garment."
    ),
}
# Human verdict: is every briefed change visible?
HUMAN_BRIEF_MET: dict[str, bool] = {
    SWEATER: True,
    DRESS: True,
    TOP: True,
    final_registry.SUMMER: True,
}
ALL_SELECTED: dict[str, Path] = {**SELECTED, final_registry.SUMMER: SUMMER_SELECTED}


def selected_seed(style_id: str) -> int:
    """The seed of the selected candidate (parsed from its file name)."""
    return int(re.search(r"seed(\d+)", SELECTED[style_id].stem).group(1))


def concept_filename(style_id: str) -> str:
    """Name of the committed copy of the selected concept in `reports/concepts/`."""
    plain = final_registry.PLAIN_NAMES[style_id].lower().replace(" ", "-")
    return f"{plain}_{SELECTED[style_id].stem}.png"


def _forecast_by_style() -> dict[str, dict[str, Any]]:
    if not FORECAST.exists():
        return {}
    return {r["style_id"]: r for r in pl.read_csv(FORECAST).iter_rows(named=True)}


# Presentation label for every closed-loop panel (figure and DEMO). The numbers are the recorded
# 40-photo validation (retrieval_validation.csv `full_index_40` vs
# concept_forecast_validation.csv smolvlm: 11 retrieval-only vs 5 free-text-only exact matches,
# McNemar exact p=0.2101).
CLOSED_LOOP_LABEL = (
    "Prototype: nearest-neighbour retrieval over 1,980 styles (summer concept: 3,000). "
    "Exact-match 27.5% vs 12.5% free-text (n=40, McNemar p=0.21 — better but not "
    "established). Near-ties dominate: see top-5."
)
_TOP5_RE = re.compile(r"^(?P<key>.+) \(sim (?P<sim>[\d.]+), rank (?P<rank>\d+)\)$")
_ORDINALS = {2: "2nd", 3: "3rd", 4: "4th", 5: "5th"}


def closed_loop_view(style_id: str) -> dict[str, Any] | None:
    """Top-5 retrieval view of one concept's closed loop, read from the recorded Q2 table.

    Presentation only: parses the `top5` column (best-first, 3-decimal similarities) and marks where
    the intended style falls. Nothing is recomputed, re-ranked or re-thresholded.

    Returns:
        None if the concept has no forecast row; else a dict with `rows` (pos, key, sim, rank,
        intended), `intended_pos` (None if outside the top 5), `gap` (top-1 minus intended
        similarity when the intended style is 2nd-5th), `spread` (top-1 minus top-5 similarity),
        `summary` (one sentence) and the forecast/confidence of the top-1 style.
    """
    f = _forecast_by_style().get(style_id)
    if f is None or not f.get("top5"):
        return None
    rows = []
    for pos, part in enumerate(str(f["top5"]).split(" | "), start=1):
        m = _TOP5_RE.match(part)
        if m is None:
            raise ValueError(f"unparseable top5 entry: {part!r}")
        rows.append(
            {
                "pos": pos,
                "key": m["key"],
                "sim": float(m["sim"]),
                "rank": int(m["rank"]),
                "intended": m["key"] == style_id,
            }
        )
    hit = next((r for r in rows if r["intended"]), None)
    pos = hit["pos"] if hit else None
    gap = round(rows[0]["sim"] - hit["sim"], 3) if hit and pos != 1 else None
    if pos == 1:
        ahead = round(rows[0]["sim"] - rows[1]["sim"], 3)
        summary = f"Intended style is 1st, {ahead:.3f} ahead of 2nd"
    elif pos:
        summary = f"Intended style {_ORDINALS[pos]}, {gap:.3f} behind top-1"
    else:
        summary = "Intended style is outside the top 5"
    want = style_id.split(" || ")
    same_type = sum(r["key"].split(" || ")[1] == want[1] for r in rows)
    same_colour = sum(r["key"].split(" || ")[3] == want[3] for r in rows)
    return {
        "rows": rows,
        "same_type": same_type,
        "same_colour": same_colour,
        "intended_pos": pos,
        "gap": gap,
        "spread": round(rows[0]["sim"] - rows[-1]["sim"], 3),
        "summary": summary,
        "n_styles": int(f["n_styles"]),
        "units": f["forecast"],
        "confidence": f["confidence"],
    }


def selection_rows() -> list[dict[str, Any]]:
    """One row per style with every gate's result for the selected image."""
    scored = pl.read_csv(SCORED)
    forecasts = _forecast_by_style()
    rows = []
    for sid in ALL_SELECTED:
        img = ALL_SELECTED[sid]
        g = scored.filter(pl.col("image_path") == str(img)).to_dicts()[0]
        judges = sorted(c[: -len("_fidelity")] for c in g if c.endswith("_fidelity"))
        automatic = bool(
            g["gate1_pass"]
            and g["gate1b_pass"]
            and g["integrity_floor_pass"]
            and g["gate2_pass"]
            and g["gate3_pass"]
        )
        f = forecasts.get(sid, {})
        rows.append(
            {
                **{k: g[k] for k in g if not k.endswith("_extraction")},
                "style_id": sid,
                "image_path": str(img),
                "judges": ",".join(judges),
                "automatic_gates": "PASS" if automatic else "FAIL",
                "human_brief_met": HUMAN_BRIEF_MET[sid],
                "human_check": HUMAN_CHECK[sid],
                "forecast_style_key": f.get("mapped_style_key"),
                "forecast_units": f.get("forecast"),
                "forecast_rank": f.get("rank"),
                "forecast_confidence": f.get("confidence"),
            }
        )
    return rows


def _gate_text(r: dict[str, Any]) -> str:
    def mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    return "\n".join(
        [
            "Gate 1 (mean sim <= p90 of real pairs)",
            f"CLIP   {r['clip_mean_sim']:.3f} <= {r['clip_p90_limit']:.3f}",
            f"DINOv2 {r['dinov2_mean_sim']:.3f} <= {r['dinov2_p90_limit']:.3f}"
            f"  -> {mark(r['gate1_pass'])}",
            "",
            "Gate 1b (closest ref <= p90 of real nearest)",
            f"CLIP   {r['clip_max_sim']:.3f} <= {r['clip_gate1b_limit']:.3f}",
            f"DINOv2 {r['dinov2_max_sim']:.3f} <= {r['dinov2_gate1b_limit']:.3f}"
            f"  -> {mark(r['gate1b_pass'])}",
            "",
            "Integrity floor, GLOBAL (gates): closest ref",
            f"DINOv2 {r['floor_max_sim']:.3f} >= {r['global_floor_limit']:.3f}"
            f"  -> {mark(r['integrity_floor_pass'])}",
            f"Per-style floor (advisory) {r['floor_limit']:.3f}"
            f"  -> {mark(r['integrity_style_pass'])}",
            f"(refs: {r['n_refs']} screened real articles)",
        ]
    )


def _judge_text(r: dict[str, Any]) -> str:
    lines = ["Gate 2 -- attribute fidelity (local judges)"]
    for j in r["judges"].split(","):
        tag = " (advisory)" if j in ADVISORY_JUDGES else ""
        lines.append(
            f"{j}{tag}: {r[f'{j}_fidelity']:.2f}  {'PASS' if r[f'{j}_gate2_pass'] else 'FAIL'}"
        )
    lines += ["", "Gate 3 -- briefed changes visible?"]
    for j in r["judges"].split(","):
        if r.get(f"{j}_gate3_answers") is not None:
            lines.append(
                f"{j}: {r[f'{j}_gate3_answers']}  {'PASS' if r[f'{j}_gate3_pass'] else 'FAIL'}"
            )
    return "\n".join(lines)


def _closed_loop_text(r: dict[str, Any]) -> str:
    v = closed_loop_view(r["style_id"])
    if v is None:
        return "Closed loop: not run"
    out = ["\n".join(textwrap.wrap(CLOSED_LOOP_LABEL, 78)), ""]
    out.append(f"Top-5 matched styles of {v['n_styles']:,} (similarity, forecast rank):")
    for t in v["rows"]:
        mark = ">>" if t["intended"] else "  "
        key = t["key"].replace(" || ", "/")
        out.append(f"{mark} {t['pos']}. {key}  {t['sim']:.3f} #{t['rank']}")
    out.append(">> = the style this concept was designed from")
    out.append(f"{v['summary']}; top-5 similarity spread {v['spread']:.3f}.")
    out.append(
        f"Of the top 5: {v['same_type']}/5 share the intended product type, "
        f"{v['same_colour']}/5 the intended colour."
    )
    out.append(f"Top-1 forecast: {v['units']:.1f} units/product/wk ({v['confidence']} confidence)")
    return "\n".join(out)


def build_evidence_figure(rows: list[dict[str, Any]], refs: dict[str, list[Path]]) -> plt.Figure:
    """refs -> brief -> concept -> Gate 1/1b/integrity -> Gate 2/3/forecast -> verdict."""
    fig, axes = plt.subplots(
        len(rows),
        6,
        figsize=(30, 5.4 * len(rows)),
        gridspec_kw={"width_ratios": [1.0, 0.85, 1.0, 1.0, 1.9, 1.1]},
    )
    titles = (
        "Real references",
        "Briefed changes (inputs, not verified)",
        "Generated concept",
        "Gates 1, 1b, integrity",
        "Gates 2, 3, closed-loop forecast",
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
        brief = "Changes asked for:\n" + "\n".join(
            f"  - {c}" for c in concept_generation.CHANGES[r["style_id"]]["applied_changes"]
        )
        scale_line = f"IP scale {r['scale']:.2f}, seed {r['seed']}, weight 1.5, 8-ref concat"
        brief += f"\n\n{scale_line}"
        for ax, txt in ((a_brief, brief), (a_g1, _gate_text(r)), (a_g2, _judge_text(r))):
            ax.axis("off")
            ax.text(
                0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=7.8, family="monospace"
            )
        a_g2.text(
            0.02,
            0.70,
            _closed_loop_text(r),
            transform=a_g2.transAxes,
            va="top",
            fontsize=8.6,
            family="monospace",
        )
        a_v.axis("off")
        human = "briefed changes visible" if r["human_brief_met"] else "brief NOT met"
        a_v.text(
            0.05,
            0.70,
            f"Automatic gates: {r['automatic_gates']}\nHuman check: {human}",
            transform=a_v.transAxes,
            fontsize=13,
            fontweight="bold",
            va="center",
        )
        a_v.text(
            0.05,
            0.52,
            "Human visual check:\n" + textwrap.fill(r["human_check"], 52),
            transform=a_v.transAxes,
            fontsize=8.5,
            va="top",
        )
    for ax, t in zip(axes[0], titles, strict=True):
        ax.set_title(t, fontsize=10.5, fontweight="bold", pad=10)
    fig.suptitle(
        "Evidence chain: references -> briefed changes -> concept -> Gates 1 / 1b / integrity -> "
        "Gate 2 / Gate 3 / closed-loop forecast -> verdict",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def main() -> None:
    """Write the selection table, the clean hero, and the evidence chain."""
    rows = selection_rows()
    pl.DataFrame(rows, infer_schema_length=None).write_csv(SELECTION_OUT)
    panels = [
        PanelData(
            r["style_id"],
            display_name(r["style_id"]),
            OBSERVED_CAPTIONS[r["style_id"]],
            Path(r["image_path"]),
            r,
            r["automatic_gates"] == "PASS" and r["human_brief_met"],
        )
        for r in rows
        if r["style_id"] in STYLE_ORDER
    ]
    hero = build_hero_figure(panels)
    hero.savefig(HERO_OUT, dpi=150, bbox_inches="tight")
    plt.close(hero)
    refs = concept_generation.load_refs()
    ev = build_evidence_figure(rows, refs)
    ev.savefig(EVIDENCE_OUT, dpi=150, bbox_inches="tight")
    plt.close(ev)
    for r in rows:
        print(
            final_registry.DISPLAY_NAMES[r["style_id"]],
            r["automatic_gates"],
            "human brief met:",
            r["human_brief_met"],
        )


if __name__ == "__main__":
    main()
