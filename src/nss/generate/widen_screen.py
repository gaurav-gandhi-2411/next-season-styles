"""Screen the widened reference pool with the local judge and write the reference base (task N5).

Every fetched candidate (`reports/tables/reference_pool_widened.csv`) is checked, blind to nothing
but the question asked, by the calibrated local VLM for:

1. FRAMING: a full garment product photo (whole garment visible, flat-lay / ghost mannequin /
   hanger),
   not a texture crop or close-up, and not a garment worn by a person;
2. COLOUR conformity: the garment's main colour is the style's `perceived_colour_master_name`;
3. PATTERN conformity: the fabric pattern matches the style's graphical appearance where that label
   is a visual one (non-visual catch-alls are skipped, as in Gate 2).

Fail-closed: an image is kept only when every applicable check answers yes. Survivors are ordered by
units sold in the trailing 26 weeks (best-selling first, so `references[0]` remains the best
seller) and capped at `TARGET_REFERENCES`. A style whose catalogue cannot reach the target is
reported as `exhausted`, never padded. The result is written as the screened-reference manifest
that `screen_references.load_screened_references` reads, with the same columns it needs plus the
per-check answers, so the provenance of every keep/drop is auditable.

Usage:
    NSS_LOCAL_VLM=moondream2 uv run python -m nss.generate.widen_screen
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image

from nss.generate import local_vlm
from nss.generate.fidelity import NON_VISUAL_GRAPHICAL_VALUES

POOL_PATH = Path("reports/tables/reference_pool_widened.csv")
OUT_PATH = Path("reports/tables/reference_base_widened.csv")
TARGET_REFERENCES = 25
# A product photo has a plain studio border; a fabric close-up does not. Measured on the pool: real
# product shots sit at 2-10 (grey-level std of the outer 5% border), the one known texture crop
# (article 673677023) at 21.4. The small VLM's framing answer was "yes" for 100% of images, so it
# cannot be the only framing check; this image statistic backs it up.
MAX_BORDER_STD = 15.0


def border_std(path: Path, frac: float = 0.05) -> float:
    """Grey-level standard deviation of the image's outer border strip."""
    im = np.asarray(Image.open(path).convert("L"), dtype=float)
    h, w = im.shape
    bh, bw = max(2, int(h * frac)), max(2, int(w * frac))
    strips = np.concatenate(
        [im[:bh].ravel(), im[-bh:].ravel(), im[:, :bw].ravel(), im[:, -bw:].ravel()]
    )
    return float(strips.std())


FRAMING_QUESTION = (
    "Does this photo show ONE complete garment, fully visible, laid flat or on a mannequin or "
    "hanger, on a plain background (not a close-up of fabric, not several items, not a person "
    "wearing it)?"
)


def screen_one(path: Path, colour: str, pattern: str) -> dict[str, object]:
    """The three checks for one image (pattern check skipped for non-visual labels)."""
    framing, framing_raw = local_vlm.ask_yes_no(path, FRAMING_QUESTION)
    bstd = border_std(path)
    framing = framing and bstd <= MAX_BORDER_STD
    colour_ok, colour_raw = local_vlm.ask_yes_no(
        path, f"Is the main colour of the garment {colour.lower()}?"
    )
    if pattern.strip() in NON_VISUAL_GRAPHICAL_VALUES:
        pattern_ok, pattern_raw = True, "skipped (non-visual label)"
    elif pattern.strip() == "Solid":
        # Asked as a NEGATIVE ("is it patterned?"): the small judge answered "no" to the direct
        # "is the fabric solid?" for 31 of 40 plain red dresses (measured), which would have
        # discarded most of the base; a visible print is what conformity actually excludes.
        patterned, pattern_raw = local_vlm.ask_yes_no(
            path,
            "Does this garment have a printed pattern, stripes, checks, floral or lace design?",
        )
        pattern_ok = not patterned
    else:
        pattern_ok, pattern_raw = local_vlm.ask_yes_no(
            path, f"Is the fabric of the garment {pattern.lower()}?"
        )
    return {
        "framing_ok": framing,
        "framing_raw": framing_raw,
        "border_std": bstd,
        "colour_ok": colour_ok,
        "colour_raw": colour_raw,
        "pattern_ok": pattern_ok,
        "pattern_raw": pattern_raw,
        "is_full_garment": framing and colour_ok and pattern_ok,
    }


SUMMER_POOL = Path("reports/tables/reference_pool_widened_summer.csv")
SUMMER_OUT = Path("reports/tables/reference_base_widened_summer.csv")


def main(summer: bool = False) -> None:
    """Screen every fetched candidate, keep the best-selling survivors, write the manifest."""
    pool = pl.read_csv(SUMMER_POOL if summer else POOL_PATH).filter(pl.col("fetch_success"))
    rows = []
    for style_id in pool["style_id"].unique(maintain_order=True).to_list():
        _dept, _ptype, _group, colour, pattern = (p.strip() for p in style_id.split(" || "))
        cands = pool.filter(pl.col("style_id") == style_id).sort(
            "units_sold_last_26w", descending=True
        )
        kept = 0
        for rec in cands.iter_rows(named=True):
            verdict = screen_one(Path(rec["image_path"]), colour, pattern)
            keep = bool(verdict["is_full_garment"]) and kept < TARGET_REFERENCES
            kept += int(keep)
            rows.append(
                {
                    "style_id": style_id,
                    "article_id": rec["article_id"],
                    "image_path": rec["image_path"],
                    "units_sold_last_26w": rec["units_sold_last_26w"],
                    **verdict,
                    "is_full_garment": keep,
                    "screening_note": (
                        "kept"
                        if keep
                        else "dropped: failed a check"
                        if not verdict["is_full_garment"]
                        else f"dropped: beyond target {TARGET_REFERENCES}"
                    ),
                }
            )
        print(f"{style_id}: kept {kept} of {cands.height} screened (target {TARGET_REFERENCES})")
    pl.DataFrame(rows).write_csv(SUMMER_OUT if summer else OUT_PATH)


if __name__ == "__main__":
    import sys

    main(summer="summer" in sys.argv[1:])
