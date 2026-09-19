"""Build the final deliverable figures from E5's real, current, already-generated evidence.

Pure image composition -- no GPU, no generation, no scoring. Consumes artifacts already on disk:
`reports/tables/design_briefs.json` (C5, updated by E5 with `applied_changes`),
`reports/tables/exemplar_images_final_three.csv` (A8, real catalogue reference images),
`reports/tables/final_concepts_v2.csv` (E5's full retry history under the corrected prompts: every
attempt, every style, real CLIP/DINOv2 margins against the E2 sign-safe copy-anchor gate and the
Groq VLM attribute-fidelity judge, plus the visual-QC-overridden final selection per style), and
`reports/tables/margin_anchors_clip.csv`/`..._dinov2.csv` (C2's original two-sided real-space band,
shown in `evidence_chain.png` as a diagnostic only -- see that figure's docstring).

SUPERSEDES task C8's `final_concepts.csv`/`concept_qc_results.csv`-based version of this module (see
git history) -- E1/E2/E5 replaced the margin-anchor derivation, the QC gate, and the concept
generation itself, so the deliverable figures must be rebuilt from the new artifacts, not patched
around the old ones.

HONEST RESULT, NOT SUPPRESSED: E5 found 0 of the 3 winning styles pass BOTH new E2 gates (Gate 1:
sign-safe copy-anchor check on CLIP+DINOv2; Gate 2: VLM attribute fidelity >= 0.75) within the
2-retry cap (`final_concepts_v2.csv`'s `selection_passed` column is `False` for all 3 `is_selected`
rows). This module does not fabricate a passing result -- both figures it produces show the real
generated images and the real scores, including the failures. A QC gate that correctly rejects real
defects (colour/pattern drift, a texture-close-up reference image dominating conditioning) is a
working gate, not a broken deliverable; that is the story these figures are built to tell.

FIGURE SPLIT: `FINAL_concepts.png` (the hero) is a clean fashion deliverable -- one panel per
winning style, the selected concept image, and a one-line rationale naming the concrete changes
actually applied (drawn from each brief's `applied_changes` field). It carries NO QC status, no
pass/fail stamps, no scores -- that belongs entirely in `evidence_chain.png`, which traces
reference -> brief -> concept -> real margins/scores per style, including the honest 0/3 status.

Usage:
    uv run python -m nss.generate.final_deliverables
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless -- this module only writes PNGs, never shows a window.
import matplotlib.pyplot as plt
import polars as pl
from PIL import Image

from nss.generate.final_concepts import load_design_briefs, load_final_three_references
from nss.generate.scale_sweep import CLIP_BAND_PATH, DINO_BAND_PATH, load_margin_band

FINAL_CONCEPTS_V2_PATH = Path("reports/tables/final_concepts_v2.csv")
HERO_OUT_PATH = Path("reports/figures/FINAL_concepts.png")
EVIDENCE_OUT_PATH = Path("reports/figures/evidence_chain.png")

ATTRIBUTE_FIDELITY_THRESHOLD = 0.75  # must match concept_qc_pipeline.ATTRIBUTE_FIDELITY_THRESHOLD.

# Display order for both figures, matching `final_concepts_v2.csv`'s style_id values.
STYLE_ORDER: tuple[str, ...] = (
    "Ladieswear || T-shirt || Jersey Basic || Black || Solid",
    "Ladieswear || Underwear bottom || Under-, Nightwear || Red || Solid",
    "Ladieswear || Sweater || Knitwear || Beige || Melange",
)

# Human-readable panel titles. Hardcoded (not derived via string manipulation) since there are
# only 3 styles and exact wording matters for a headline/hero image -- see module docstring.
STYLE_DISPLAY_NAMES: dict[str, str] = {
    STYLE_ORDER[0]: "Black Jersey Basic T-Shirt",
    STYLE_ORDER[1]: "Red Under-, Nightwear Underwear Bottom",
    STYLE_ORDER[2]: "Beige Melange Knitwear Sweater",
}

# How many real catalogue reference images to show per style in the evidence-chain figure.
N_EXEMPLAR_THUMBNAILS = 3
_EXEMPLAR_THUMB_HEIGHT_PX = 260
_EXEMPLAR_THUMB_GAP_PX = 8

_PRESERVE_PREFIXES: dict[str, str] = {
    "category": "garment category:",
    "construction": "construction/fabrication family:",
    "colour": "anchor colour:",
    "surface": "surface treatment:",
}


def load_final_concepts_v2(path: Path = FINAL_CONCEPTS_V2_PATH) -> pl.DataFrame:
    """Load `final_concepts_v2.csv` (E5's full retry-history + visual-QC-applied selection).

    Args:
        path: Path to `final_concepts_v2.csv`.

    Returns:
        The raw results frame, one row per `(style_id, seed)`.
    """
    return pl.read_csv(path)


def display_name(style_id: str) -> str:
    """Human-readable panel title for one `style_id` (see `STYLE_DISPLAY_NAMES`).

    Args:
        style_id: A `design_briefs.json`/`final_concepts_v2.csv` `style_id`.

    Returns:
        The registered display name.

    Raises:
        ValueError: if no display name is registered for `style_id`.
    """
    try:
        return STYLE_DISPLAY_NAMES[style_id]
    except KeyError as exc:
        raise ValueError(f"No display name registered for style_id={style_id!r}") from exc


def select_final_row(df: pl.DataFrame, style_id: str) -> dict[str, Any]:
    """Return the E5-selected (`is_selected == True`) row for one style.

    Selection itself (including the manual visual-QC veto) already happened upstream in
    `nss.generate.final_concepts_v2.apply_visual_qc_and_rewrite` -- this module only reads the
    result, it never re-ranks candidates.

    Args:
        df: Output of `load_final_concepts_v2`.
        style_id: The style to look up.

    Returns:
        The selected row as a dict.

    Raises:
        ValueError: if `style_id` has zero or more than one `is_selected` row.
    """
    selected = df.filter((pl.col("style_id") == style_id) & pl.col("is_selected"))
    if selected.height != 1:
        raise ValueError(
            f"Expected exactly 1 is_selected row for style_id={style_id!r}, found "
            f"{selected.height}"
        )
    return selected.row(0, named=True)


def _preserve_value(preserve: list[str], prefix: str) -> str:
    """Extract the value after `"<prefix>: "` from a design brief's `preserve` list.

    Args:
        preserve: A `design_briefs.json` entry's `preserve` list.
        prefix: The leading label to match (e.g. `"anchor colour:"`).

    Returns:
        The trimmed value text after the matching item's colon.

    Raises:
        ValueError: if no item in `preserve` starts with `prefix`.
    """
    for item in preserve:
        if item.startswith(prefix):
            return item.split(":", 1)[1].strip()
    raise ValueError(f"No preserve item starting with {prefix!r} in {preserve!r}")


def _strip_change_annotation(item: str) -> str:
    """Drop an `applied_changes` item's trailing `" -- <category label>"` annotation, if present.

    Args:
        item: One `design_briefs.json` `applied_changes` entry (e.g.
            `"ribbed funnel neckline in place of the plain crew neckline -- neckline variant"`).

    Returns:
        The change description alone, with the trailing category label removed.
    """
    return item.split(" -- ", 1)[0].strip()


def build_rationale(brief: dict[str, Any]) -> str:
    """One-line 'what was intentionally kept vs. changed' rationale for one style.

    Drawn directly from the design brief's own `preserve`/`applied_changes` fields (never
    invented): the preserved side names the exact winning combination (garment category,
    construction family, anchor colour, surface treatment) that must NOT change, because it is
    what is driving demand; the changed side names the CONCRETE changes actually baked into this
    round's prompt (`applied_changes`) -- not the earlier brainstormed axis list, the real ones
    this concept was generated with.

    Args:
        brief: One `design_briefs.json` entry (must have `preserve` with the 4 dimensions in
            `_PRESERVE_PREFIXES`, and `applied_changes`).

    Returns:
        A single-sentence rationale string.
    """
    preserve = brief["preserve"]
    category = _preserve_value(preserve, _PRESERVE_PREFIXES["category"])
    construction = _preserve_value(preserve, _PRESERVE_PREFIXES["construction"])
    colour = _preserve_value(preserve, _PRESERVE_PREFIXES["colour"])
    changes = "; ".join(_strip_change_annotation(item) for item in brief["applied_changes"])
    return (
        f"Preserves the winning {colour} {construction} {category} combination exactly -- "
        f"demand tracks this exact combination, not a seasonal cue. Applied changes: {changes}."
    )


@dataclass(frozen=True)
class PanelData:
    """Everything one hero/evidence-chain panel needs for one winning style."""

    style_id: str
    display_name: str
    rationale: str
    image_path: Path
    row: dict[str, Any]
    selection_passed: bool


def build_panel_data(
    df: pl.DataFrame, briefs: dict[str, dict[str, Any]], style_id: str
) -> PanelData:
    """Assemble one style's `PanelData`: E5-selected row + brief-derived rationale.

    Args:
        df: Output of `load_final_concepts_v2`.
        briefs: Output of `nss.generate.final_concepts.load_design_briefs`.
        style_id: The style to build panel data for.

    Returns:
        The assembled `PanelData`.
    """
    row = select_final_row(df, style_id)
    brief = briefs[style_id]
    return PanelData(
        style_id=style_id,
        display_name=display_name(style_id),
        rationale=build_rationale(brief),
        image_path=Path(row["image_path"]),
        row=row,
        selection_passed=bool(row["selection_passed"]),
    )


def build_hero_figure(panels: list[PanelData]) -> plt.Figure:
    """Build the `FINAL_concepts.png` hero figure: one clean panel per winning style.

    Each panel shows the E5-selected generated concept image (`select_final_row`), captioned with
    the style's human-readable name and a one-line design rationale naming the concrete changes
    actually applied. This is deliberately a clean fashion deliverable, NOT a QC dashboard -- no
    pass/fail stamps, no QC badges, no scores anywhere on this image. The full, honest QC trace
    (including the real 0/3 gate status) lives entirely in `evidence_chain.png`.

    Args:
        panels: One `PanelData` per winning style, in display order.

    Returns:
        The constructed `Figure`, ready to save.

    Raises:
        ValueError: if `panels` is empty.
    """
    if not panels:
        raise ValueError("panels must be non-empty")

    fig, axes_raw = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 7.4))
    axes = [axes_raw] if len(panels) == 1 else list(axes_raw)

    for ax, panel in zip(axes, panels, strict=True):
        img = Image.open(panel.image_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(panel.display_name, fontsize=13, fontweight="bold", pad=10)

        caption = "\n".join(textwrap.wrap(panel.rationale, width=42))
        ax.text(
            0.5,
            -0.05,
            caption,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
        )

    fig.suptitle(
        "next-season-styles -- 3 Winning Styles, Generated Concepts",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.94))
    return fig


def build_exemplar_composite(paths: list[Path], n: int = N_EXEMPLAR_THUMBNAILS) -> Image.Image:
    """Concatenate up to `n` real reference images side-by-side into one composite thumbnail strip.

    Each source image is resized to a common height (preserving aspect ratio) before
    concatenation, since H&M product photos vary in native resolution/aspect ratio.

    Args:
        paths: Reference image paths for one style (already existence-checked by
            `nss.generate.final_concepts.load_final_three_references`), sorted.
        n: Max number of thumbnails to include (the first `n` of `paths`).

    Returns:
        One composite `PIL.Image.Image`.

    Raises:
        ValueError: if `paths` is empty.
    """
    if not paths:
        raise ValueError("paths must be non-empty")

    chosen = paths[:n]
    thumbs: list[Image.Image] = []
    for p in chosen:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        new_w = max(1, round(w * _EXEMPLAR_THUMB_HEIGHT_PX / h))
        thumbs.append(img.resize((new_w, _EXEMPLAR_THUMB_HEIGHT_PX)))

    total_w = sum(t.width for t in thumbs) + _EXEMPLAR_THUMB_GAP_PX * (len(thumbs) - 1)
    composite = Image.new("RGB", (total_w, _EXEMPLAR_THUMB_HEIGHT_PX), color="white")
    x = 0
    for t in thumbs:
        composite.paste(t, (x, 0))
        x += t.width + _EXEMPLAR_THUMB_GAP_PX
    return composite


def _wrap_bullet(item: str, width: int = 44) -> str:
    """Wrap one bullet item to a fixed width, continuation lines indented.

    Some `design_briefs.json` bullet items run past 100 characters unwrapped (e.g. the "winning
    combination as a whole" `preserve` item) -- rendered as a single unwrapped line inside a narrow
    evidence-chain column, a line that long forces `tight_layout()` to collapse ALL columns to
    near-zero width trying to accommodate it. Every bullet is wrapped here, unconditionally, so no
    rendered line can ever be wide enough to cause that.

    Args:
        item: One bullet's raw text.
        width: Max characters per wrapped line.

    Returns:
        A `"  - "`-prefixed, possibly multi-line, indented bullet string.
    """
    return textwrap.fill(item, width=width, initial_indent="  - ", subsequent_indent="    ")


def build_brief_excerpt(brief: dict[str, Any]) -> str:
    """Readable multi-line excerpt of one design brief's silhouette/colour/preserve/applied fields.

    Shows what was intentionally KEPT (`preserve`) alongside what was intentionally CHANGED
    (`applied_changes` -- the concrete, brief-grounded changes actually baked into this round's
    prompt, not the earlier brainstormed axis list).

    Args:
        brief: One `design_briefs.json` entry.

    Returns:
        A newline-joined text block, ready for direct rendering via `ax.text`.
    """
    preserve_lines = "\n".join(_wrap_bullet(item) for item in brief["preserve"][:4])
    applied_lines = "\n".join(_wrap_bullet(item) for item in brief["applied_changes"])
    return (
        f"Silhouette:\n  {textwrap.fill(brief['silhouette'], width=46)}\n\n"
        f"Colour direction:\n  {textwrap.fill(brief['colour_direction'], width=46)}\n\n"
        f"PRESERVED (kept intentionally):\n{preserve_lines}\n\n"
        f"CHANGED (applied this round):\n{applied_lines}"
    )


def build_margin_text(
    row: dict[str, Any],
    real_space_clip_band: tuple[float, float],
    real_space_dino_band: tuple[float, float],
) -> str:
    """Readable multi-line block of one concept's real margins vs. BOTH anchor systems.

    Shows the concept's real CLIP/DINOv2 margins against (1) the E2 Gate-1 sign-safe copy-anchor
    threshold -- the LIVE gate this concept was actually selected/rejected against -- and (2), as a
    diagnostic only, the OLD two-sided real-space band from C2, which no longer gates selection.
    Both are clearly labeled so a reviewer never confuses the two.

    Args:
        row: One `final_concepts_v2.csv` row (the E5-selected candidate for a style).
        real_space_clip_band: `(lower, upper)` C2 real-space CLIP band (diagnostic only).
        real_space_dino_band: `(lower, upper)` C2 real-space DINOv2 band (diagnostic only).

    Returns:
        A newline-joined text block with real numbers, never just a checkmark/x.
    """
    clip_lo, clip_hi = real_space_clip_band
    dino_lo, dino_hi = real_space_dino_band
    clip_diag_in_band = clip_lo <= row["clip_margin"] <= clip_hi
    dino_diag_in_band = dino_lo <= row["dino_margin"] <= dino_hi

    return (
        f"Seed {row['seed']} (retry round {row['retry_round']})\n\n"
        f"GATE 1 -- LIVE sign-safe copy-anchor check (E2):\n"
        f"  CLIP margin: {row['clip_margin']:.4f}\n"
        f"    copy-anchor threshold: {row['clip_copy_anchor_threshold']:.4f}\n"
        f"    below copy-anchor (too close to source): {row['clip_below_copy_anchor']}\n"
        f"  DINOv2 margin: {row['dino_margin']:.4f}\n"
        f"    copy-anchor threshold: {row['dino_copy_anchor_threshold']:.4f}\n"
        f"    below copy-anchor (too close to source): {row['dino_below_copy_anchor']}\n"
        f"  Gate 1 copy_check_pass: {row['copy_check_pass']}\n\n"
        f"DIAGNOSTIC ONLY -- old two-sided real-space band (C2, NOT a live gate):\n"
        f"  CLIP band: [{clip_lo:.4f}, {clip_hi:.4f}] -- "
        f"{'IN' if clip_diag_in_band else 'OUT of'} band\n"
        f"  DINOv2 band: [{dino_lo:.4f}, {dino_hi:.4f}] -- "
        f"{'IN' if dino_diag_in_band else 'OUT of'} band"
    )


def build_judge_scores_text(row: dict[str, Any]) -> str:
    """Readable multi-line block of the selected candidate's per-judge Gate-2 attribute scores.

    States plainly when Gemini was unavailable this round (free-tier quota exhaustion) rather than
    hiding the resulting single-judge (Groq-only) limitation.

    Args:
        row: One `final_concepts_v2.csv` row (the E5-selected candidate for a style).

    Returns:
        A newline-joined text block: each judge's availability/score, consensus fidelity, and the
        final Gate 2 + overall verdicts, with the actual numbers -- never just a checkmark/x.
    """
    lines = [f"GATE 2 -- VLM attribute fidelity (threshold >= {ATTRIBUTE_FIDELITY_THRESHOLD}):", ""]

    if row["groq_available"]:
        lines.append(f"  Groq: {row['groq_mean_score']:.3f}")
    else:
        lines.append(f"  Groq: unavailable ({row['groq_excluded_reason']})")

    if row["gemini_available"]:
        lines.append(f"  Gemini: {row['gemini_mean_score']:.3f}")
    else:
        lines.append("  Gemini: UNAVAILABLE this round -- free-tier quota exhausted")
        lines.append("    (429 RESOURCE_EXHAUSTED). Single-judge (Groq-only) result below --")
        lines.append("    NOT a 2-judge consensus.")

    lines.append("")
    lines.append(
        f"  Mean attribute fidelity (n_judges={row['n_contributing_judges']}): "
        f"{row['mean_attribute_fidelity']:.3f}"
    )
    fidelity_status = "PASS" if row["fidelity_pass"] else "FAIL"
    lines.append(f"  Gate 2 fidelity_pass: {fidelity_status}")
    lines.append("")
    overall_status = "PASS" if row["overall_pass"] else "FAIL"
    lines.append(f"OVERALL (Gate 1 AND Gate 2), this candidate: {overall_status}")
    lines.append("")
    style_status = "PASSED" if row["selection_passed"] else "DID NOT PASS"
    lines.append(
        f"Style QC status ({row['n_retry_rounds_used']} retry round(s) used, "
        f"selection_mode={row['selection_mode']}):"
    )
    lines.append(f"  {style_status}")
    return "\n".join(lines)


def build_evidence_chain_figure(
    panels: list[PanelData],
    references: dict[str, list[Path]],
    briefs: dict[str, dict[str, Any]],
    real_space_clip_band: tuple[float, float],
    real_space_dino_band: tuple[float, float],
) -> plt.Figure:
    """Build `evidence_chain.png`: refs -> brief -> concept -> margins-vs-anchors -> judge scores.

    One row per winning style (3 rows), 5 columns: (1) a composite of real catalogue reference
    images for that style, (2) the design brief's silhouette/colour/preserve/applied-changes
    excerpt as readable text (what was kept vs. changed), (3) the SAME generated concept image
    selected for the hero figure (`select_final_row`), (4) the concept's real CLIP/DINOv2 margins
    against BOTH the live E2 Gate-1 copy-anchor threshold and, as a labeled diagnostic only, the
    old C2 two-sided real-space band, (5) the real, printed Gate-2 per-judge attribute-fidelity
    scores, explicitly noting when Gemini was unavailable (quota exhaustion) this round, plus the
    final overall/style QC verdict. Lets a reviewer trace, end to end, exactly how each winning
    style's real reference photos and design brief led to its generated concept, and see the real
    evidence (including the honest 0/3 gate status) for that concept's quality.

    Args:
        panels: One `PanelData` per winning style, in display order.
        references: Output of `nss.generate.final_concepts.load_final_three_references`.
        briefs: Output of `nss.generate.final_concepts.load_design_briefs`.
        real_space_clip_band: `(lower, upper)` C2 real-space CLIP band (diagnostic only).
        real_space_dino_band: `(lower, upper)` C2 real-space DINOv2 band (diagnostic only).

    Returns:
        The constructed `Figure`, ready to save.

    Raises:
        ValueError: if `panels` is empty.
    """
    if not panels:
        raise ValueError("panels must be non-empty")

    col_titles = (
        "Real catalogue references",
        "Design brief -- kept vs. changed",
        "Generated concept",
        "Margins vs. Gate 1 + C2 diagnostic band",
        "Gate 2 judge scores + QC verdict",
    )
    fig, axes = plt.subplots(len(panels), 5, figsize=(24.0, 5.6 * len(panels)))
    if len(panels) == 1:
        axes = axes.reshape(1, 5)

    for row_idx, panel in enumerate(panels):
        ax_ref, ax_brief, ax_img, ax_margin, ax_scores = axes[row_idx]

        composite = build_exemplar_composite(references[panel.style_id])
        ax_ref.imshow(composite)
        # Ticks/spines hidden manually (NOT `ax.axis("off")`) so the native Matplotlib `ylabel`
        # below is kept as the per-row style-name label -- a real Axes decoration that
        # `tight_layout()` accounts for correctly, unlike a manually positioned out-of-bounds
        # `ax.text()` (which previously made `tight_layout()` warn that it couldn't find a width
        # fitting every decoration).
        ax_ref.set_xticks([])
        ax_ref.set_yticks([])
        for spine in ax_ref.spines.values():
            spine.set_visible(False)
        ax_ref.set_ylabel(panel.display_name, fontsize=11, fontweight="bold")

        ax_brief.axis("off")
        ax_brief.text(
            0.02,
            0.98,
            build_brief_excerpt(briefs[panel.style_id]),
            transform=ax_brief.transAxes,
            ha="left",
            va="top",
            fontsize=7.4,
            family="monospace",
        )

        img = Image.open(panel.image_path)
        ax_img.imshow(img)
        ax_img.axis("off")

        ax_margin.axis("off")
        ax_margin.text(
            0.02,
            0.98,
            build_margin_text(panel.row, real_space_clip_band, real_space_dino_band),
            transform=ax_margin.transAxes,
            ha="left",
            va="top",
            fontsize=7.6,
            family="monospace",
        )

        ax_scores.axis("off")
        ax_scores.text(
            0.02,
            0.98,
            build_judge_scores_text(panel.row),
            transform=ax_scores.transAxes,
            ha="left",
            va="top",
            fontsize=7.6,
            family="monospace",
        )

    for ax, title in zip(axes[0], col_titles, strict=True):
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=10)

    fig.suptitle(
        "next-season-styles -- Evidence Chain: References -> Brief -> Concept -> Gate 1/Gate 2 "
        "Scores",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return fig


def main(
    final_concepts_v2_path: Path = FINAL_CONCEPTS_V2_PATH,
    hero_out_path: Path = HERO_OUT_PATH,
    evidence_out_path: Path = EVIDENCE_OUT_PATH,
) -> tuple[Path, Path]:
    """Run the full pipeline: load E5's selected candidates, build + save both figures.

    Args:
        final_concepts_v2_path: Path to `final_concepts_v2.csv`.
        hero_out_path: Destination for the hero figure; parent directories are created if missing.
        evidence_out_path: Destination for the evidence-chain figure; parent directories are
            created if missing.

    Returns:
        `(hero_out_path, evidence_out_path)`.
    """
    df = load_final_concepts_v2(final_concepts_v2_path)
    briefs = load_design_briefs()
    references = load_final_three_references()
    real_space_clip_band = load_margin_band(CLIP_BAND_PATH)
    real_space_dino_band = load_margin_band(DINO_BAND_PATH)

    panels = [build_panel_data(df, briefs, style_id) for style_id in STYLE_ORDER]

    for panel in panels:
        status = "PASSED" if panel.selection_passed else "DID NOT PASS"
        print(
            f"[select] {panel.display_name}: seed={panel.row['seed']} "
            f"retry_round={panel.row['retry_round']} "
            f"fidelity={panel.row['mean_attribute_fidelity']:.3f} "
            f"selection_mode={panel.row['selection_mode']} style_qc={status}"
        )

    hero_fig = build_hero_figure(panels)
    hero_out_path.parent.mkdir(parents=True, exist_ok=True)
    hero_fig.savefig(hero_out_path, dpi=150, bbox_inches="tight")
    plt.close(hero_fig)
    print(f"Wrote {hero_out_path}")

    evidence_fig = build_evidence_chain_figure(
        panels, references, briefs, real_space_clip_band, real_space_dino_band
    )
    evidence_out_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_fig.savefig(evidence_out_path, dpi=150, bbox_inches="tight")
    plt.close(evidence_fig)
    print(f"Wrote {evidence_out_path}")

    return hero_out_path, evidence_out_path


if __name__ == "__main__":
    main()
