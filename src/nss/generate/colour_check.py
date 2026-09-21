"""L1: a MEASURED garment-colour check for Gate 2 (replaces reading colour from a VLM).

A hard constraint amplifies whatever error its input carries, and the small VLM misread red
dresses as orange (K4). So the colour is measured from pixels. Method, pre-registered in
`reports/v3/PREREGISTRATION.md` (section L1) and fixed before any concept was scored:

1. resize to 256 px, convert to CIELAB;
2. background colour = median Lab of the outer 4% ring (border sampling; `rembg` is not
   installed and would need a model download);
3. mask = pixels more than 12 (CIE76) from the background, opened, largest component, holes filled,
   eroded by 3 px; under 3% of the image (a white garment on a near-white background) falls back to
   the central 40% box, and the fallback is reported;
4. dominant colour = centre of the largest of 3 k-means clusters (seed 42) over the mask's Lab
   pixels.

A concept's statistic is the minimum CIEDE2000 from its dominant colour to those of the style's real
reference articles. The threshold is calibrated exactly like Gate 1b: the p90 of the real
nearest-sibling distances of that style. `loo_pass_rate` validates it by nested leave-one-out on the
real articles.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.color import deltaE_ciede2000, rgb2lab
from sklearn.cluster import KMeans

SIZE = 256
RING = 0.04  # fraction of width/height sampled as background
BG_DELTA = 12.0  # CIE76 distance from the background that counts as garment
MIN_MASK = 0.03  # below this fraction of the image the mask is not trusted
ERODE_PX = 3
K = 3
SEED = 42
MAX_PIXELS = 20_000  # deterministic stride subsample for k-means
PERCENTILE = 90


@dataclass(frozen=True)
class Dominant:
    """A measured garment colour and how it was obtained."""

    lab: tuple[float, float, float]
    used_fallback: bool
    mask_fraction: float


def _load_lab(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    scale = SIZE / max(img.size)
    img = img.resize(
        (max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS
    )
    return rgb2lab(np.asarray(img, dtype=np.float64) / 255.0)


def garment_mask(lab: np.ndarray) -> np.ndarray:
    """Boolean mask of the garment from border-sampled background subtraction (steps 2-3)."""
    h, w, _ = lab.shape
    r = max(1, round(RING * min(h, w)))
    ring = np.concatenate(
        [
            lab[:r].reshape(-1, 3),
            lab[-r:].reshape(-1, 3),
            lab[:, :r].reshape(-1, 3),
            lab[:, -r:].reshape(-1, 3),
        ]
    )
    bg = np.median(ring, axis=0)
    mask = np.linalg.norm(lab - bg, axis=2) > BG_DELTA
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    labels, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask)
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    mask = labels == (int(np.argmax(sizes)) + 1)
    mask = ndimage.binary_fill_holes(mask)
    return ndimage.binary_erosion(mask, iterations=ERODE_PX)


def dominant_colour(path: Path | str) -> Dominant:
    """The dominant garment colour of one image (steps 1-4)."""
    lab = _load_lab(Path(path))
    mask = garment_mask(lab)
    fraction = float(mask.mean())
    fallback = fraction < MIN_MASK
    if fallback:
        h, w, _ = lab.shape
        mask = np.zeros(mask.shape, dtype=bool)
        mask[int(0.3 * h) : int(0.7 * h), int(0.3 * w) : int(0.7 * w)] = True
    pixels = lab[mask]
    if len(pixels) > MAX_PIXELS:
        pixels = pixels[:: len(pixels) // MAX_PIXELS]
    km = KMeans(n_clusters=min(K, len(pixels)), n_init=10, random_state=SEED).fit(pixels)
    biggest = int(np.argmax(np.bincount(km.labels_)))
    centre = km.cluster_centers_[biggest]
    return Dominant((float(centre[0]), float(centre[1]), float(centre[2])), fallback, fraction)


def delta_e(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """CIEDE2000 between two Lab colours."""
    return float(deltaE_ciede2000(np.array(a), np.array(b)))


def nearest_sibling_distances(labs: list[tuple[float, float, float]]) -> list[float]:
    """For each colour, the minimum CIEDE2000 to the others."""
    return [min(delta_e(a, b) for j, b in enumerate(labs) if j != i) for i, a in enumerate(labs)]


def threshold_from(labs: list[tuple[float, float, float]]) -> float:
    """The p90 of the real nearest-sibling distances (calibrated like Gate 1b)."""
    return float(np.percentile(nearest_sibling_distances(labs), PERCENTILE))


@functools.cache
def _reference_colours(style_key: str) -> tuple[tuple[str, Dominant], ...]:
    from nss.generate import qc_gates

    return tuple(
        (p.name, dominant_colour(p)) for p in qc_gates.reference_paths_for_style(style_key)
    )


def profile(style_key: str) -> dict[str, Any]:
    """The style's reference colours and its calibrated threshold."""
    refs = _reference_colours(style_key)
    labs = [d.lab for _n, d in refs]
    return {"n_references": len(refs), "threshold": threshold_from(labs), "labs": labs}


def check(concept_path: Path | str, style_key: str) -> dict[str, Any]:
    """Measured colour check of one concept against its style: pass iff within the p90."""
    prof = profile(style_key)
    dom = dominant_colour(concept_path)
    nearest = min(delta_e(dom.lab, ref) for ref in prof["labs"])
    return {
        "pass": bool(nearest <= prof["threshold"]),
        "nearest_delta_e": nearest,
        "threshold": prof["threshold"],
        "dominant_lab": dom.lab,
        "used_fallback": dom.used_fallback,
        "mask_fraction": dom.mask_fraction,
        "n_references": prof["n_references"],
    }


def loo_pass_rate(style_key: str) -> dict[str, Any]:
    """Nested leave-one-out on the real articles (threshold never sees the held-out one)."""
    refs = _reference_colours(style_key)
    labs = [d.lab for _n, d in refs]
    passed = 0
    for i, held in enumerate(labs):
        rest = labs[:i] + labs[i + 1 :]
        thr = threshold_from(rest)
        passed += min(delta_e(held, r) for r in rest) <= thr
    return {
        "style": style_key,
        "n": len(labs),
        "passed": passed,
        "pass_rate": passed / len(labs),
        "fallbacks": sum(d.used_fallback for _n, d in refs),
    }
