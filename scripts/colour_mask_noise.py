"""M3: measure the noise of the garment mask and set the ESCALATE band from it.

Rule pre-registered in `reports/v3/PREREGISTRATION.md` (section M, commit 3b63aa9): for each real
reference article of the four final styles, 8 jittered crops (a window of 90% of the width and of
the height at a uniformly random offset, `random.Random(42 + image index)`) go through the full
pipeline (rembg mask, k-means); the deviation is the CIEDE2000 between the crop's dominant colour
and the full image's. The band half-width is the 95th percentile of the pooled deviations. Measured
on real references only, before any concept is scored.

    uv run --no-sync python scripts/colour_mask_noise.py
"""

from __future__ import annotations

import random

import numpy as np
import polars as pl
from PIL import Image

from nss.generate import colour_check as cc
from nss.generate import qc_gates
from nss.generate.final_selection_figures import ALL_SELECTED

N_CROPS = 8
KEEP = 0.9


def main() -> None:
    """Write the deviations and print the pooled and per-style p95."""
    rows = []
    idx = 0
    for style in ALL_SELECTED:
        paths = qc_gates.reference_paths_for_style(style)
        for path, (_name, full) in zip(paths, cc._reference_colours(style), strict=True):
            rng = random.Random(42 + idx)
            img = Image.open(path).convert("RGB")
            w, h = img.size
            cw, ch = round(KEEP * w), round(KEEP * h)
            for j in range(N_CROPS):
                x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
                dom = cc.dominant_from_image(img.crop((x, y, x + cw, y + ch)))
                rows.append(
                    {
                        "style": style.split(" || ")[1],
                        "image": path.name,
                        "crop": j,
                        "deviation": cc.delta_e(dom.lab, full.lab),
                        "fallback_crop": dom.used_fallback,
                        "fallback_full": full.used_fallback,
                    }
                )
            idx += 1
    df = pl.DataFrame(rows)
    df.write_csv("reports/tables/v3_colour_mask_noise.csv")
    dev = df["deviation"].to_numpy()
    print(
        "crops",
        len(dev),
        "pooled p50",
        round(float(np.percentile(dev, 50)), 3),
        "p90",
        round(float(np.percentile(dev, 90)), 3),
        "p95",
        round(float(np.percentile(dev, 95)), 3),
        "max",
        round(float(dev.max()), 3),
    )
    for style, g in df.group_by("style", maintain_order=True):
        d = g["deviation"].to_numpy()
        print(
            style[0],
            "n",
            len(d),
            "p95",
            round(float(np.percentile(d, 95)), 3),
            "median",
            round(float(np.median(d)), 3),
        )  # noqa: E501


if __name__ == "__main__":
    main()
