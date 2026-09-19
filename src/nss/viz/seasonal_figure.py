# ruff: noqa: E501  -- long f-string labels are figure text, not logic
"""`reports/figures/seasonal_comparison.png` (task K8c): autumn/winter 2020 vs summer, side by side.

Left: the autumn/winter rank-1 concept (the black jersey T-shirt from the 21 Sep 2020 forecast).
Right: the Summer rank-1 concept, from the SAME pipeline at the same settings, seeded by the
model's own summer-origin forecast (`nss.generate.seasonal_concept`). Each panel is labelled with
its style and the model's predicted intensity (units sold per product on sale per week). Labels
read from committed tables; the captions describe what is visible in each picture.

Usage:
    uv run python -m nss.viz.seasonal_figure
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import polars as pl
from PIL import Image

from nss.generate.h4_deliverables import OBSERVED_CAPTIONS, SELECTED, T_SHIRT, _image
from nss.generate.seasonal_concept import SELECTED_SEED, candidate_path

OUT_PATH = Path("reports/figures/seasonal_comparison.png")
T = Path("reports/tables")
SUMMER_CAPTION = (
    "Black high-waisted swimwear bottom in a ribbed, structured fabric, with a folded-over "
    "waistband and a small picot-edged trim along the leg openings. Solid colour."
)
INK, ACCENT, MUTED = "#15181e", "#2a3fd0", "#5f646d"


def main() -> Path:
    """Render and save the figure."""
    aw = pl.read_csv(T / "top_styles_final_three.csv").filter(pl.col("style_key") == T_SHIRT)
    aw_row = aw.to_dicts()[0]
    summer = pl.read_csv(T / "seasonal_summer_forecast.csv").to_dicts()[0]
    seed, _src = SELECTED[T_SHIRT]
    panels = [
        (
            "Autumn / winter 2020",
            "Black jersey T-shirt",
            _image(T_SHIRT, seed, "f5"),
            f"Model forecast from 21 Sep 2020: {aw_row['predicted_intensity']:.1f} units per product per week",
            OBSERVED_CAPTIONS[T_SHIRT],
        ),
        (
            "Summer 2020",
            "Black swimwear bottom",
            candidate_path(SELECTED_SEED),
            f"Model forecast from 1 Jun 2020: {summer['predicted_intensity']:.1f} units per product per week"
            f" (realised {summer['realised_intensity']:.1f})",
            SUMMER_CAPTION,
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 7.4), facecolor="white")
    for ax, (season, name, path, label, caption) in zip(axes, panels, strict=True):
        ax.imshow(Image.open(path))
        ax.axis("off")
        ax.set_title(season, fontsize=11, color=MUTED, loc="left", pad=10)
        ax.text(0, 1.045, "", transform=ax.transAxes)
        ax.text(
            0.0,
            -0.035,
            name,
            transform=ax.transAxes,
            fontsize=17,
            family="serif",
            color=INK,
            va="top",
        )
        ax.text(
            0.0,
            -0.105,
            "\n".join(textwrap.wrap(label, 58)),
            transform=ax.transAxes,
            fontsize=10.5,
            color=ACCENT,
            va="top",
            fontweight="bold",
        )
        ax.text(
            0.0,
            -0.2,
            "\n".join(textwrap.wrap(caption, 62)),
            transform=ax.transAxes,
            fontsize=9.5,
            color=MUTED,
            va="top",
        )
    fig.suptitle(
        "Same pipeline, two seasons: the forecast changes which garment is designed",
        fontsize=14,
        family="serif",
        color=INK,
        x=0.03,
        ha="left",
        y=0.97,
    )
    fig.subplots_adjust(top=0.9, bottom=0.2, left=0.03, right=0.97, wspace=0.08)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")
    return OUT_PATH


if __name__ == "__main__":
    main()
