"""Build the C8 final deliverable figures from C6/C7's real, already-generated evidence.

Pure image composition -- no GPU, no generation, no scoring. Consumes 3 artifacts already on
disk: `reports/tables/design_briefs.json` (C5), `reports/tables/exemplar_images_final_three.csv`
(A8, real catalogue reference images), and `reports/tables/concept_qc_results.csv` (C7's FULL
retry history: every attempt, every style, real CLIP/DINOv2 margins and VLM attribute-fidelity
scores).

HONEST RESULT, NOT SUPPRESSED: C7 found 0 of the 3 winning styles pass the full QC gate within
the 2-retry cap (`concept_qc_results.csv`'s `style_final_pass` column is `False` for all 3). This
module does not fabricate a passing result -- both figures it produces show the real generated
images and the real scores, including the failures. A QC gate that correctly rejects real defects
(a degenerate fabric close-up, a human model where none was allowed, weak attribute fidelity) is a
working gate, not a broken deliverable; that is the story these figures are built to tell.

SELECTION RULE (documented, not left implicit -- see `select_best_attempt`): for each style, EVERY
attempt logged in `concept_qc_results.csv` (not just attempt 0) is scored as
`composite = float(clip_in_band) + float(dino_in_band) + mean_attribute_fidelity`, ties broken by
the LOWER `attempt_number`. This happens to select attempt 0 (C6's original candidate) for all 3
styles in this run -- not hardcoded, see that function's docstring for why the retry history itself
produces that outcome.

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

from nss.generate.concept_qc_pipeline import MAX_RETRIES
from nss.generate.final_concepts import load_design_briefs, load_final_three_references

QC_RESULTS_PATH = Path("reports/tables/concept_qc_results.csv")
HERO_OUT_PATH = Path("reports/figures/FINAL_concepts.png")
EVIDENCE_OUT_PATH = Path("reports/figures/evidence_chain.png")

# Display order for both figures, matching `final_concepts.csv`/`concept_qc_results.csv` row order.
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


def load_qc_results(path: Path = QC_RESULTS_PATH) -> pl.DataFrame:
    """Load `concept_qc_results.csv` (C7's full retry-history QC gate results).

    Args:
        path: Path to `concept_qc_results.csv`.

    Returns:
        The raw QC results frame, one row per `(style_id, attempt_number)`.
    """
    return pl.read_csv(path)


def display_name(style_id: str) -> str:
    """Human-readable panel title for one `style_id` (see `STYLE_DISPLAY_NAMES`).

    Args:
        style_id: A `design_briefs.json`/`concept_qc_results.csv` `style_id`.

    Returns:
        The registered display name.

    Raises:
        ValueError: if no display name is registered for `style_id`.
    """
    try:
        return STYLE_DISPLAY_NAMES[style_id]
    except KeyError as exc:
        raise ValueError(f"No display name registered for style_id={style_id!r}") from exc


def select_best_attempt(qc_df: pl.DataFrame, style_id: str) -> dict[str, Any]:
    """Pick the best-available attempt for one style across its full QC retry history.

    SELECTION RULE (documented, not left implicit): every logged attempt (attempt 0 = C6's
    original candidate, plus up to `MAX_RETRIES` retries) is scored as
    `composite = float(clip_in_band) + float(dino_in_band) + mean_attribute_fidelity` -- each
    margin-band pass is worth 1.0, the same 0-1 scale as `mean_attribute_fidelity`, so band
    proximity and attribute fidelity are weighted equally rather than one dominating the other.
    Ties are broken by the LOWER `attempt_number`: when two attempts score identically, prefer the
    earlier/least-perturbed one, since later retries push `ip_adapter_scale` further from C3's only
    evidence-based operating point (0.2) without a demonstrated benefit -- the more conservative
    pick, not an arbitrary one.

    NOTE (see the C8 task report for the worked numbers): for all 3 of this project's winning
    styles, this rule selects attempt 0. That is not hardcoded -- it falls out of the rule because
    every retry in `concept_qc_results.csv` either ties or WORSENS `mean_attribute_fidelity` and
    never improves the in-band count for any of the 3 styles actually tried.

    Args:
        qc_df: Output of `load_qc_results` (all attempts, all styles).
        style_id: The style to select the best-available attempt for.

    Returns:
        The selected attempt's row as a dict (every `concept_qc_results.csv` column, plus the
        computed `composite_score`).

    Raises:
        ValueError: if `style_id` has no rows in `qc_df`.
    """
    style_df = qc_df.filter(pl.col("style_id") == style_id)
    if style_df.is_empty():
        raise ValueError(f"No QC attempts found for style_id={style_id!r}")

    scored = style_df.with_columns(
        (
            pl.col("clip_in_band").cast(pl.Float64)
            + pl.col("dino_in_band").cast(pl.Float64)
            + pl.col("mean_attribute_fidelity")
        ).alias("composite_score")
    ).sort(["composite_score", "attempt_number"], descending=[True, False])
    return scored.row(0, named=True)


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


def build_rationale(brief: dict[str, Any]) -> str:
    """One-line 'why this reads as next season, not just a copy' rationale for one style.

    Drawn directly from the design brief's own `preserve`/`change` fields (never invented): the
    `preserve` side names the exact winning combination (garment category, construction family,
    anchor colour, surface treatment) that must NOT change, because it is what is driving demand;
    the `change` side names the specific axes (a graphic/trim accent, a small proportion tweak)
    that DO change -- what makes the concept a forward-looking variation rather than a literal
    reproduction of an existing style.

    Args:
        brief: One `design_briefs.json` entry (must have `preserve` with the 4 dimensions in
            `_PRESERVE_PREFIXES`).

    Returns:
        A single-sentence rationale string.
    """
    preserve = brief["preserve"]
    category = _preserve_value(preserve, _PRESERVE_PREFIXES["category"])
    construction = _preserve_value(preserve, _PRESERVE_PREFIXES["construction"])
    colour = _preserve_value(preserve, _PRESERVE_PREFIXES["colour"])
    surface = _preserve_value(preserve, _PRESERVE_PREFIXES["surface"])
    return (
        f"Preserves the winning {colour} {construction} {category} combination exactly "
        f"({surface} surface treatment included) -- demand tracks this exact combination, not a "
        f"seasonal cue; introduces a subtle new graphic/trim accent and a small proportion tweak "
        f"so the concept reads as next season, not a literal copy."
    )


@dataclass(frozen=True)
class PanelData:
    """Everything one hero/evidence-chain panel needs for one winning style."""

    style_id: str
    display_name: str
    rationale: str
    image_path: Path
    attempt: dict[str, Any]
    style_final_pass: bool


def build_panel_data(
    qc_df: pl.DataFrame, briefs: dict[str, dict[str, Any]], style_id: str
) -> PanelData:
    """Assemble one style's `PanelData`: best-available attempt + brief-derived rationale.

    Args:
        qc_df: Output of `load_qc_results`.
        briefs: Output of `nss.generate.final_concepts.load_design_briefs`.
        style_id: The style to build panel data for.

    Returns:
        The assembled `PanelData`.
    """
    attempt = select_best_attempt(qc_df, style_id)
    brief = briefs[style_id]
    return PanelData(
        style_id=style_id,
        display_name=display_name(style_id),
        rationale=build_rationale(brief),
        image_path=Path(attempt["image_path"]),
        attempt=attempt,
        style_final_pass=bool(attempt["style_final_pass"]),
    )


def build_hero_figure(panels: list[PanelData]) -> plt.Figure:
    """Build the `FINAL_concepts.png` hero figure: one clean panel per winning style.

    Each panel shows the best-available generated concept image (`select_best_attempt`), captioned
    with the style's human-readable name and a one-line design rationale. Per the honesty
    requirement (C7 found 0/3 concepts pass the full QC gate within the 2-retry cap), each caption
    also states the concept's QC status in one short line -- never hidden, but kept small/textual
    rather than a large visual badge, to keep this a clean headline image (the full per-metric
    evidence lives in `evidence_chain.png`, not repeated here).

    Args:
        panels: One `PanelData` per winning style, in display order.

    Returns:
        The constructed `Figure`, ready to save.

    Raises:
        ValueError: if `panels` is empty.
    """
    if not panels:
        raise ValueError("panels must be non-empty")

    fig, axes_raw = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 7.6))
    axes = [axes_raw] if len(panels) == 1 else list(axes_raw)

    for ax, panel in zip(axes, panels, strict=True):
        img = Image.open(panel.image_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(panel.display_name, fontsize=13, fontweight="bold", pad=10)

        caption = "\n".join(textwrap.wrap(panel.rationale, width=42))
        qc_note = (
            "QC: PASSED full gate"
            if panel.style_final_pass
            else "QC: did not pass full gate within the 2-retry cap -- see evidence_chain.png"
        )
        qc_color = "#1B7A3D" if panel.style_final_pass else "#B36B00"
        n_caption_lines = caption.count("\n") + 1

        ax.text(
            0.5,
            -0.05,
            caption,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
        )
        ax.text(
            0.5,
            -0.05 - 0.024 * n_caption_lines - 0.025,
            "\n".join(textwrap.wrap(qc_note, width=46)),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
            color=qc_color,
        )

    fig.suptitle(
        "next-season-styles -- 3 Winning Styles, Generated Concepts (task C8)",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
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
    """Wrap one `preserve`/`change` bullet to a fixed width, continuation lines indented.

    Some `design_briefs.json` `preserve`/`change` items run past 100 characters unwrapped (e.g.
    the "winning combination as a whole" item) -- rendered as a single unwrapped line inside a
    narrow evidence-chain column, a line that long forces `tight_layout()` to collapse ALL
    columns to near-zero width trying to accommodate it (see the C8 task report for the
    before/after). Every bullet is wrapped here, unconditionally, so no rendered line can ever be
    wide enough to cause that.

    Args:
        item: One bullet's raw text.
        width: Max characters per wrapped line.

    Returns:
        A `"  - "`-prefixed, possibly multi-line, indented bullet string.
    """
    return textwrap.fill(item, width=width, initial_indent="  - ", subsequent_indent="    ")


def build_brief_excerpt(brief: dict[str, Any]) -> str:
    """Readable multi-line excerpt of one design brief's silhouette/colour/preserve/change fields.

    Args:
        brief: One `design_briefs.json` entry.

    Returns:
        A newline-joined text block, ready for direct rendering via `ax.text`.
    """
    preserve_lines = "\n".join(_wrap_bullet(item) for item in brief["preserve"][:4])
    change_lines = "\n".join(_wrap_bullet(item) for item in brief["change"])
    return (
        f"Silhouette:\n  {textwrap.fill(brief['silhouette'], width=46)}\n\n"
        f"Colour direction:\n  {textwrap.fill(brief['colour_direction'], width=46)}\n\n"
        f"PRESERVE:\n{preserve_lines}\n\n"
        f"CHANGE:\n{change_lines}"
    )


def build_scores_text(attempt: dict[str, Any]) -> str:
    """Readable multi-line block of the selected attempt's real, printed QC scores.

    Args:
        attempt: Output of `select_best_attempt` (one `concept_qc_results.csv` row + the computed
            `composite_score`).

    Returns:
        A newline-joined text block: CLIP margin, DINOv2 margin, attribute fidelity, and the final
        QC pass/fail verdict, with the actual numbers -- never just a checkmark/x.
    """
    clip_status = "in-band" if attempt["clip_in_band"] else "OUT of band"
    dino_status = "in-band" if attempt["dino_in_band"] else "OUT of band"
    fidelity_status = "PASS" if attempt["fidelity_pass"] else "FAIL"
    overall_status = "PASS" if attempt["overall_pass"] else "FAIL"
    style_status = "PASSED" if attempt["style_final_pass"] else "DID NOT PASS"
    return (
        f"Attempt {attempt['attempt_number']} "
        f"(seed={attempt['seed']}, ip_adapter_scale={attempt['ip_adapter_scale']:.2f})\n\n"
        f"CLIP margin: {attempt['clip_margin']:.4f} ({clip_status})\n"
        f"DINOv2 margin: {attempt['dino_margin']:.4f} ({dino_status})\n"
        f"Mean attribute fidelity: {attempt['mean_attribute_fidelity']:.3f} "
        f"(n_judges={attempt['n_contributing_judges']}, {fidelity_status})\n\n"
        f"This attempt's overall_pass: {overall_status}\n"
        f"Style QC status ({attempt['n_attempts_for_style']} attempt(s), "
        f"{MAX_RETRIES}-retry cap): {style_status}\n\n"
        f"Selection composite score: {attempt['composite_score']:.3f}\n"
        f"(clip_in_band + dino_in_band + mean_attribute_fidelity)"
    )


def build_evidence_chain_figure(
    panels: list[PanelData],
    references: dict[str, list[Path]],
    briefs: dict[str, dict[str, Any]],
) -> plt.Figure:
    """Build `evidence_chain.png`: exemplar refs -> design brief -> generated concept -> scores.

    One row per winning style (3 rows), 4 columns: (1) a composite of real catalogue reference
    images for that style, (2) the design brief's silhouette/colour/preserve/change excerpt as
    readable text, (3) the SAME generated concept image selected for the hero figure
    (`select_best_attempt`), (4) the real, printed QC scores for that selected attempt (CLIP
    margin, DINOv2 margin, attribute fidelity, and the final pass/fail verdict -- numbers, not
    just a checkmark). Lets a reviewer trace, end to end, exactly how each winning style's real
    reference photos and design brief led to its generated concept, and see the real evidence
    (including failure) for that concept's quality.

    Args:
        panels: One `PanelData` per winning style, in display order.
        references: Output of `nss.generate.final_concepts.load_final_three_references`.
        briefs: Output of `nss.generate.final_concepts.load_design_briefs`.

    Returns:
        The constructed `Figure`, ready to save.

    Raises:
        ValueError: if `panels` is empty.
    """
    if not panels:
        raise ValueError("panels must be non-empty")

    col_titles = (
        "Real catalogue references",
        "Design brief excerpt",
        "Generated concept",
        "QC scores (real numbers)",
    )
    fig, axes = plt.subplots(len(panels), 4, figsize=(20.0, 5.4 * len(panels)))
    if len(panels) == 1:
        axes = axes.reshape(1, 4)

    for row_idx, panel in enumerate(panels):
        ax_ref, ax_brief, ax_img, ax_scores = axes[row_idx]

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
            fontsize=7.6,
            family="monospace",
        )

        img = Image.open(panel.image_path)
        ax_img.imshow(img)
        ax_img.axis("off")

        ax_scores.axis("off")
        ax_scores.text(
            0.02,
            0.98,
            build_scores_text(panel.attempt),
            transform=ax_scores.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            family="monospace",
        )

    for ax, title in zip(axes[0], col_titles, strict=True):
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    fig.suptitle(
        "next-season-styles -- Evidence Chain: References -> Brief -> Concept -> QC Scores "
        "(task C8)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return fig


def main(
    qc_results_path: Path = QC_RESULTS_PATH,
    hero_out_path: Path = HERO_OUT_PATH,
    evidence_out_path: Path = EVIDENCE_OUT_PATH,
) -> tuple[Path, Path]:
    """Run the full C8 pipeline: select best-available attempts, build + save both figures.

    Args:
        qc_results_path: Path to `concept_qc_results.csv`.
        hero_out_path: Destination for the hero figure; parent directories are created if missing.
        evidence_out_path: Destination for the evidence-chain figure; parent directories are
            created if missing.

    Returns:
        `(hero_out_path, evidence_out_path)`.
    """
    qc_df = load_qc_results(qc_results_path)
    briefs = load_design_briefs()
    references = load_final_three_references()

    panels = [build_panel_data(qc_df, briefs, style_id) for style_id in STYLE_ORDER]

    for panel in panels:
        status = "PASSED" if panel.style_final_pass else "DID NOT PASS"
        print(
            f"[select] {panel.display_name}: attempt={panel.attempt['attempt_number']} "
            f"seed={panel.attempt['seed']} scale={panel.attempt['ip_adapter_scale']:.2f} "
            f"composite_score={panel.attempt['composite_score']:.3f} "
            f"style_qc={status}"
        )

    hero_fig = build_hero_figure(panels)
    hero_out_path.parent.mkdir(parents=True, exist_ok=True)
    hero_fig.savefig(hero_out_path, dpi=150, bbox_inches="tight")
    plt.close(hero_fig)
    print(f"Wrote {hero_out_path}")

    evidence_fig = build_evidence_chain_figure(panels, references, briefs)
    evidence_out_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_fig.savefig(evidence_out_path, dpi=150, bbox_inches="tight")
    plt.close(evidence_fig)
    print(f"Wrote {evidence_out_path}")

    return hero_out_path, evidence_out_path


if __name__ == "__main__":
    main()
