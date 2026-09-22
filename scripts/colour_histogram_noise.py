"""N4: measure the noise of the colour-histogram statistic and set the patterned-class ESCALATE
band from it (`W_BAND_PATTERNED`), the same procedure M3 used for the dominant-colour band.

Rule pre-registered in `reports/v3/PREREGISTRATION.md` (section N4, commit 635ac6d): for each real
reference article of the patterned-class calibrated styles (`colour_check.pattern_class`), 8
jittered crops (a window of 90% of the width and of the height at a uniformly random offset,
`random.Random(42 + image index)`) go through the full histogram pipeline (rembg mask, per-channel
Lab histogram); the deviation is `histogram_distance` between the crop's histogram and the full
image's. The band half-width is the 95th percentile of the pooled deviations. Measured on real
references only, before any concept is scored with the histogram statistic.

    uv run --no-sync python scripts/colour_histogram_noise.py
"""

from __future__ import annotations

import random

import numpy as np
import polars as pl
from PIL import Image

from nss.generate import colour_check as cc
from nss.generate import qc_gates

N_CROPS = 8
KEEP = 0.9


def main() -> None:
    """Write the deviations and print the pooled and per-style p95."""
    patterned_styles = [s for s in cc.calibrated_styles() if cc.pattern_class(s) == "patterned"]
    print("patterned calibrated styles:", [s.split(" || ")[1] for s in patterned_styles])
    rows = []
    idx = 0
    for style in patterned_styles:
        paths = qc_gates.reference_paths_for_style(style)
        for path, (_name, full) in zip(paths, cc._reference_histograms(style), strict=True):
            rng = random.Random(42 + idx)
            img = Image.open(path).convert("RGB")
            w, h = img.size
            cw, ch = round(KEEP * w), round(KEEP * h)
            for j in range(N_CROPS):
                x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
                hist = cc.colour_histogram_from_image(img.crop((x, y, x + cw, y + ch)))
                rows.append(
                    {
                        "style": style.split(" || ")[1],
                        "image": path.name,
                        "crop": j,
                        "deviation": cc.histogram_distance(hist, full),
                        "fallback_crop": hist.used_fallback,
                        "fallback_full": full.used_fallback,
                    }
                )
            idx += 1
    df = pl.DataFrame(rows)
    df.write_csv("reports/tables/v3_colour_histogram_noise.csv")
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
        )


if __name__ == "__main__":
    main()
