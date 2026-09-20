"""The two figures a reviewer sees first: the concept sheet and the evidence chain.

Both are pure composition over recorded results (no scoring, no generation). The typography is a
serif for titles and names and a humanist sans for everything else, with pass and fail shown as
coloured labels rather than terminal-style text.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: this module only writes PNGs.
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.axes import Axes
from PIL import Image


def _first_available(candidates: list[str]) -> str:
    """The first installed font family among `candidates` (falls back to the last)."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    return next((c for c in candidates if c in installed), candidates[-1])


SERIF = _first_available(["Georgia", "Cambria", "Palatino Linotype", "DejaVu Serif"])
SANS = _first_available(["Segoe UI", "Calibri", "Corbel", "DejaVu Sans"])

INK = "#1F2933"
MUTED = "#6B7280"
RULE = "#D9D5CE"
PAPER = "#FBFAF8"
PASS_FG, PASS_BG = "#1E6B3A", "#E3F1E7"
FAIL_FG, FAIL_BG = "#9B2C2C", "#F8E3E1"
NOTE_FG, NOTE_BG = "#6B5B2A", "#F3EDDC"
ACCENT = "#8A5A2B"

HERO_TITLE = (
    "Next-season concepts — three new designs, each inspired by a style the model predicts "
    "will perform strongly"
)


def _t(ax: Axes, x: float, y: float, text: str, **kw: Any) -> Any:
    """Text in axes coordinates with the sans family by default."""
    kw.setdefault("family", SANS)
    kw.setdefault("color", INK)
    kw.setdefault("va", "top")
    return ax.text(x, y, text, transform=ax.transAxes, **kw)


def _pill(ax: Axes, x: float, y: float, ok: bool | None, text: str | None = None) -> None:
    """A right-aligned coloured status label (PASS / FAIL / not run)."""
    if ok is None:
        fg, bg, label = MUTED, "#ECEAE6", text or "not run"
    else:
        fg, bg = (PASS_FG, PASS_BG) if ok else (FAIL_FG, FAIL_BG)
        label = text or ("PASS" if ok else "FAIL")
    _t(
        ax,
        x,
        y,
        label,
        ha="right",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=fg,
        bbox={"boxstyle": "round,pad=0.28,rounding_size=0.5", "fc": bg, "ec": "none"},
    )


def build_hero(panels: list[Any]) -> plt.Figure:
    """The concept sheet: one clean panel per concept, headed by the style it was inspired by.

    Args:
        panels: Objects with `display_name`, `rationale` (the caption) and `image_path`.

    Returns:
        The figure, ready to save.

    Raises:
        ValueError: if `panels` is empty.
    """
    if not panels:
        raise ValueError("panels must be non-empty")
    n = len(panels)
    fig = plt.figure(figsize=(6.6 * n, 9.0), facecolor=PAPER)
    gs = fig.add_gridspec(1, n, left=0.03, right=0.97, top=0.74, bottom=0.17, wspace=0.09)
    fig.text(
        0.5,
        0.955,
        "\n".join(textwrap.wrap(HERO_TITLE, 62)),
        ha="center",
        va="top",
        family=SERIF,
        fontsize=25,
        fontweight="bold",
        color=INK,
        linespacing=1.3,
    )
    for i, panel in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(Image.open(panel.image_path))
        ax.axis("off")
        _t(ax, 0.5, 1.115, "Inspired by", ha="center", fontsize=11, color=MUTED)
        _t(
            ax,
            0.5,
            1.07,
            panel.display_name,
            ha="center",
            family=SERIF,
            fontsize=15.5,
            fontweight="bold",
        )
        _t(
            ax,
            0.5,
            -0.035,
            "\n".join(textwrap.wrap(panel.rationale, 46)),
            ha="center",
            fontsize=11.5,
            linespacing=1.4,
        )
    return fig


def _check_block(
    ax: Axes,
    y: float,
    title: str,
    plain: str,
    ok: bool | None,
    lines: list[str],
    lh: float,
) -> float:
    """One check: bold name and status on one line, a plain-language line, then its numbers."""
    _t(ax, 0.02, y, title, fontsize=13, fontweight="bold")
    _pill(ax, 0.97, y - lh * 0.55, ok)
    y -= lh * 1.35
    _t(ax, 0.02, y, plain, fontsize=10.5, color=MUTED)
    y -= lh * 1.25
    for line in lines:
        _t(ax, 0.04, y, line, fontsize=11)
        y -= lh * 1.15
    return y - lh * 0.7


def build_evidence(
    rows: list[dict[str, Any]],
    refs: dict[str, list[Path]],
    *,
    display_name: Callable[[str], str],
    changes_for: Callable[[str], list[str]],
    exemplar_composite: Callable[[list[Path]], Image.Image],
    closed_loop_view: Callable[[str], dict[str, Any] | None],
    closed_loop_label: str,
    advisory_judges: set[str] | frozenset[str],
) -> plt.Figure:
    """The evidence chain: references, briefed changes, concept, checks, closed loop, verdict.

    Args:
        rows: One selection row per concept (see `final_selection_figures.selection_rows`).
        refs: Real reference images per style id.
        display_name: Style id to the heading shown for the concept.
        changes_for: Style id to the list of briefed changes.
        exemplar_composite: Builds the reference thumbnail strip.
        closed_loop_view: Style id to the top-5 retrieval view (or None).
        closed_loop_label: The prototype note, shown once in the header.
        advisory_judges: Judges whose Gate 2 reading is advisory rather than deciding.

    Returns:
        The figure, ready to save.
    """
    n = len(rows)
    row_h = 5.9
    fig = plt.figure(figsize=(33, row_h * n + 3.0), facecolor=PAPER)
    top_frac = 1 - 2.7 / fig.get_size_inches()[1]
    gs = fig.add_gridspec(
        n,
        6,
        left=0.025,
        right=0.985,
        top=top_frac,
        bottom=0.012,
        wspace=0.2,
        hspace=0.16,
        width_ratios=[1.05, 0.9, 1.0, 1.25, 2.05, 1.1],
    )
    fig.text(
        0.025,
        0.99,
        "How each concept was built and checked",
        ha="left",
        va="top",
        family=SERIF,
        fontsize=27,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.025,
        0.99 - 0.55 / fig.get_size_inches()[1] * 1.0,
        "\n".join(
            [
                "From real reference photos and a brief of changes, to a generated concept, "
                "through the quality checks, to a forecast for the style it most resembles.",
                closed_loop_label,
                "A filled dot marks the style the concept was designed from. "
                "Gate 3 answers read Y (change visible) or N, one letter per briefed change.",
            ]
        ),
        ha="left",
        va="top",
        family=SANS,
        fontsize=13.5,
        color=MUTED,
        linespacing=1.55,
    )
    col_titles = (
        "Real references",
        "Briefed changes",
        "Generated concept",
        "Similarity and integrity checks",
        "Attribute checks and closed loop",
        "Verdict",
    )
    for i, r in enumerate(rows):
        axes = [fig.add_subplot(gs[i, c]) for c in range(6)]
        for ax in axes:
            ax.axis("off")
        a_ref, a_brief, a_img, a_sim, a_attr, a_v = axes
        height_in = a_sim.get_position().height * fig.get_size_inches()[1]

        def lh_for(fs: float, h: float = height_in) -> float:
            return fs * 1.0 / 72 / h

        lh = lh_for(11 * 1.4)
        # Fill the frame so the thumbnail strip keeps its aspect ratio and sits at the top.
        a_ref.axis("on")
        a_ref.imshow(exemplar_composite(refs[r["style_id"]]), aspect="equal")
        a_ref.set_anchor("NW")
        a_ref.set_xticks([])
        a_ref.set_yticks([])
        for spine in a_ref.spines.values():
            spine.set_color(RULE)
        _t(
            a_ref,
            0.0,
            -0.07,
            display_name(r["style_id"]),
            family=SERIF,
            fontsize=16,
            fontweight="bold",
        )

        y = 0.99
        _t(a_brief, 0.0, y, "Changes asked for", fontsize=12.5, fontweight="bold")
        _t(
            a_brief,
            0.0,
            y - lh * 1.3,
            "inputs to the generator, not results",
            fontsize=10.5,
            color=MUTED,
        )
        y -= lh * 3.2
        for change in changes_for(r["style_id"]):
            wrapped = textwrap.wrap(change, 30)
            _t(a_brief, 0.0, y, "•", fontsize=12, color=ACCENT)
            _t(a_brief, 0.09, y, "\n".join(wrapped), fontsize=12, linespacing=1.35)
            y -= lh * (1.35 * len(wrapped) + 0.6)

        a_img.imshow(Image.open(r["image_path"]))
        a_img.set_anchor("NW")
        _t(
            a_img,
            0.0,
            -0.02,
            f"IP-Adapter scale {r['scale']:.2f} · seed {r['seed']} · prompt weight 1.5\n"
            "8 references concatenated",
            fontsize=9.5,
            color=MUTED,
        )

        # --- similarity and integrity checks
        y = 0.99
        y = _check_block(
            a_sim,
            y,
            "Gate 1",
            "not too close to its references on average",
            bool(r["gate1_pass"]),
            [
                f"CLIP        {r['clip_mean_sim']:.3f}  ≤  {r['clip_p90_limit']:.3f}",
                f"DINOv2    {r['dinov2_mean_sim']:.3f}  ≤  {r['dinov2_p90_limit']:.3f}",
            ],
            lh,
        )
        y = _check_block(
            a_sim,
            y,
            "Gate 1b",
            "not a near-copy of any single reference",
            bool(r["gate1b_pass"]),
            [
                f"CLIP        {r['clip_max_sim']:.3f}  ≤  {r['clip_gate1b_limit']:.3f}",
                f"DINOv2    {r['dinov2_max_sim']:.3f}  ≤  {r['dinov2_gate1b_limit']:.3f}",
            ],
            lh,
        )
        y = _check_block(
            a_sim,
            y,
            "Integrity floor",
            "close enough to a real garment to look like one",
            bool(r["integrity_floor_pass"]),
            [
                f"closest reference  {r['floor_max_sim']:.3f}  ≥  {r['global_floor_limit']:.3f}",
                f"style-specific floor (advisory)  {r['floor_limit']:.3f}  "
                f"{'pass' if r['integrity_style_pass'] else 'fail'}",
            ],
            lh,
        )
        _t(
            a_sim,
            0.02,
            y,
            f"compared with {r['n_refs']} screened real articles",
            fontsize=10,
            color=MUTED,
        )

        # --- attribute checks (Gate 2, Gate 3)
        y = 0.99
        judges = r["judges"].split(",")
        gate2_ok = all(bool(r[f"{j}_gate2_pass"]) for j in judges if j not in advisory_judges)
        _t(a_attr, 0.02, y, "Gate 2", fontsize=13, fontweight="bold")
        _pill(a_attr, 0.97, y - lh * 0.55, gate2_ok)
        y -= lh * 1.35
        _t(
            a_attr,
            0.02,
            y,
            "reads as the right product type, colour and pattern",
            fontsize=10.5,
            color=MUTED,
        )
        y -= lh * 1.25
        for j in judges:
            tag = "  (advisory)" if j in advisory_judges else ""
            ok = bool(r[f"{j}_gate2_pass"])
            _t(a_attr, 0.04, y, f"{j}{tag}   fidelity {r[f'{j}_fidelity']:.2f}", fontsize=11)
            _pill(a_attr, 0.97, y - lh * 0.32, ok, "pass" if ok else "fail")
            y -= lh * 1.3
        y -= lh * 0.5
        _t(a_attr, 0.02, y, "Gate 3", fontsize=13, fontweight="bold")
        g3 = [j for j in judges if r.get(f"{j}_gate3_answers") is not None]
        gate3_ok = bool(r["gate3_pass"])
        _pill(a_attr, 0.97, y - lh * 0.55, gate3_ok)
        y -= lh * 1.35
        _t(a_attr, 0.02, y, "the briefed changes are visible", fontsize=10.5, color=MUTED)
        y -= lh * 1.25
        for j in g3:
            _t(a_attr, 0.04, y, f"{j}   answers {r[f'{j}_gate3_answers']}", fontsize=11)
            y -= lh * 1.3
        y -= lh * 0.9

        # --- closed loop
        view = closed_loop_view(r["style_id"])
        ax_rule = a_attr
        ax_rule.plot([0.0, 1.0], [y + lh * 0.35] * 2, transform=ax_rule.transAxes, color=RULE, lw=1)
        _t(a_attr, 0.02, y - lh * 0.2, "Closed loop", fontsize=13, fontweight="bold")
        _t(
            a_attr,
            0.2,
            y - lh * 0.28,
            "nearest real styles by image similarity",
            fontsize=10.5,
            color=MUTED,
        )
        y -= lh * 1.9
        if view is None:
            _t(a_attr, 0.04, y, "not run", fontsize=11, color=MUTED)
        else:
            _t(
                a_attr,
                0.04,
                y,
                f"top 5 of {view['n_styles']:,} styles",
                fontsize=10,
                color=MUTED,
            )
            _t(a_attr, 0.78, y, "similarity", fontsize=10, color=MUTED, ha="right")
            _t(a_attr, 0.97, y, "forecast rank", fontsize=10, color=MUTED, ha="right")
            y -= lh * 1.2
            for t in view["rows"]:
                yc = y - lh * 0.5
                if t["intended"]:
                    a_attr.plot(
                        [0.012],
                        [yc],
                        "o",
                        transform=a_attr.transAxes,
                        color=ACCENT,
                        ms=8,
                        clip_on=False,
                    )
                weight = "bold" if t["intended"] else "normal"
                key = t["key"].replace(" || ", " / ")
                _t(a_attr, 0.04, y, key, fontsize=11, fontweight=weight)
                _t(a_attr, 0.78, y, f"{t['sim']:.3f}", fontsize=11, fontweight=weight, ha="right")
                _t(a_attr, 0.97, y, f"#{t['rank']}", fontsize=11, fontweight=weight, ha="right")
                y -= lh * 1.25
            y -= lh * 0.3
            _t(
                a_attr,
                0.02,
                y,
                view["summary"] + f"; top-5 similarity spread {view['spread']:.3f}",
                fontsize=11,
                fontweight="bold",
            )
            y -= lh * 1.3
            _t(
                a_attr,
                0.02,
                y,
                f"{view['same_type']} of 5 share the intended product type, "
                f"{view['same_colour']} of 5 the intended colour",
                fontsize=10.5,
                color=MUTED,
            )
            y -= lh * 1.25
            _t(
                a_attr,
                0.02,
                y,
                f"top-1 forecast {view['units']:.1f} units per product per week "
                f"({view['confidence']} confidence)",
                fontsize=10.5,
                color=MUTED,
            )

        # --- verdict
        auto_ok = r["automatic_gates"] == "PASS"
        y = 0.99
        _t(a_v, 0.0, y, "Automatic checks", fontsize=11, color=MUTED)
        _t(
            a_v,
            0.0,
            y - lh * 1.15,
            "PASS" if auto_ok else "FAIL",
            fontsize=24,
            fontweight="bold",
            family=SERIF,
            color=PASS_FG if auto_ok else FAIL_FG,
        )
        y -= lh * 4.2
        human = "briefed changes visible" if r["human_brief_met"] else "brief not met"
        _t(a_v, 0.0, y, "Human check", fontsize=11, color=MUTED)
        _t(a_v, 0.0, y - lh * 1.15, human, fontsize=14, fontweight="bold")
        y -= lh * 3.2
        _t(
            a_v,
            0.0,
            y,
            "\n".join(textwrap.wrap(r["human_check"], 34)),
            fontsize=11,
            linespacing=1.4,
        )

        if i == 0:
            y_hdr = gs[0, 0].get_position(fig).y1 + 0.006
            for c, title in enumerate(col_titles):
                fig.text(
                    gs[0, c].get_position(fig).x0,
                    y_hdr,
                    title,
                    fontsize=13,
                    fontweight="bold",
                    family=SANS,
                    color=ACCENT,
                    va="bottom",
                )
        if i < n - 1:
            ypos = gs[i, 0].get_position(fig).y0 - 0.55 * (gs.hspace or 0.16) * 0.1
            fig.add_artist(
                plt.Line2D(
                    [0.025, 0.985],
                    [ypos - 0.006, ypos - 0.006],
                    color=RULE,
                    lw=1.2,
                    transform=fig.transFigure,
                )
            )
    return fig
