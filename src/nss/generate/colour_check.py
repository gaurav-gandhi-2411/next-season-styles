"""A MEASURED garment-colour check for Gate 2 (L1): rembg masks (M2), shrunk thresholds (M1) and
a borderline ESCALATE band (M3).

A hard constraint amplifies whatever error its input carries, and the small VLM misread red dresses
as orange (K4). So the colour is measured from pixels. Rules pre-registered in
`reports/v3/PREREGISTRATION.md` (sections L1 and M), fixed before any concept was scored:

1. resize to 256 px, convert to CIELAB;
2. garment mask: the rembg `u2net` alpha mask (M2; binarised at 128), then the largest component,
   holes filled, eroded by 3 px; under 3% of the image falls back to the central 40% box and the
   fallback is reported. (`garment_mask_border` is the L1 border mask, kept for the record.)
3. dominant colour = centre of the largest of 3 k-means clusters (seed 42) over the mask pixels.

A concept's statistic is the minimum CIEDE2000 from its dominant colour to those of the style's real
reference articles. The per-style threshold is the p90 of the real nearest-sibling distances
(calibrated like Gate 1b), shrunk toward the global median with weight `n / (n + K)`, `K` = the
median reference count over the calibrated styles (M1). Verdict: pass at or below `T - W_BAND`, fail
at or above `T + W_BAND`, otherwise ESCALATE (M3; `W_BAND` is measured mask noise, see
`scripts/colour_mask_noise.py`).
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.color import deltaE_ciede2000, rgb2lab
from sklearn.cluster import KMeans

SIZE = 256
RING = 0.04  # fraction of width/height sampled as background (L1 border mask only)
BG_DELTA = 12.0  # CIE76 distance from the background that counts as garment (L1 border mask only)
MIN_MASK = 0.03  # below this fraction of the image the mask is not trusted
ERODE_PX = 3
K = 3
SEED = 42
MAX_PIXELS = 20_000  # deterministic stride subsample for k-means
PERCENTILE = 90
K_SHRINKAGE = 17.0  # M1: median reference count over the five calibrated styles (17, 25, 19, 8, 4)
W_BAND: float | None = (
    0.403  # M3: p95 of the pooled mask-noise deviation (scripts/colour_mask_noise.py)
)
UNDERWEAR = "Ladieswear || Underwear bottom || Under-, Nightwear || Red || Solid"

PASS, FAIL, ESCALATE = "pass", "fail", "escalate"


@dataclass(frozen=True)
class Dominant:
    """A measured garment colour and how it was obtained."""

    lab: tuple[float, float, float]
    used_fallback: bool
    mask_fraction: float


@functools.lru_cache(maxsize=1)
def _rembg_session() -> Any:
    os.environ.setdefault("U2NET_HOME", str(Path("data/rembg_models").resolve()))
    from rembg import new_session

    return new_session("u2net")


def _resize(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    scale = SIZE / max(img.size)
    return img.resize(
        (max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS
    )


def garment_mask_border(lab: np.ndarray) -> np.ndarray:
    """L1: border-sampled background subtraction (kept for the record; not used by the gate)."""
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
    return ndimage.binary_opening(mask, structure=np.ones((3, 3)))


def garment_mask_rembg(img: Image.Image) -> np.ndarray:
    """M2: the rembg `u2net` alpha mask of the resized image, binarised at 128."""
    from rembg import remove

    alpha = remove(img, session=_rembg_session(), only_mask=True)
    return np.asarray(alpha.convert("L")) >= 128


def _clean(mask: np.ndarray) -> np.ndarray:
    labels, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask)
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    mask = labels == (int(np.argmax(sizes)) + 1)
    mask = ndimage.binary_fill_holes(mask)
    return ndimage.binary_erosion(mask, iterations=ERODE_PX)


def dominant_from_image(img: Image.Image, masker: str = "rembg") -> Dominant:
    """The dominant garment colour of a PIL image."""
    img = _resize(img)
    lab = rgb2lab(np.asarray(img, dtype=np.float64) / 255.0)
    raw = garment_mask_rembg(img) if masker == "rembg" else garment_mask_border(lab)
    mask = _clean(raw)
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
    centre = km.cluster_centers_[int(np.argmax(np.bincount(km.labels_)))]
    return Dominant((float(centre[0]), float(centre[1]), float(centre[2])), fallback, fraction)


def dominant_colour(path: Path | str, masker: str = "rembg") -> Dominant:
    """The dominant garment colour of one image file."""
    return dominant_from_image(Image.open(path), masker)


def delta_e(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """CIEDE2000 between two Lab colours."""
    return float(deltaE_ciede2000(np.array(a), np.array(b)))


def nearest_sibling_distances(labs: list[tuple[float, float, float]]) -> list[float]:
    """For each colour, the minimum CIEDE2000 to the others."""
    return [min(delta_e(a, b) for j, b in enumerate(labs) if j != i) for i, a in enumerate(labs)]


def threshold_from(labs: list[tuple[float, float, float]]) -> float:
    """The raw p90 of the real nearest-sibling distances (calibrated like Gate 1b)."""
    return float(np.percentile(nearest_sibling_distances(labs), PERCENTILE))


def shrink(raw: float, n: int, t_global: float, k: float = K_SHRINKAGE) -> float:
    """M1: empirical-Bayes shrinkage toward the global threshold, weight n / (n + K)."""
    w = n / (n + k)
    return w * raw + (1 - w) * t_global


def calibrated_styles() -> list[str]:
    """The five styles with a reference base: the four final styles and underwear."""
    from nss.generate.final_selection_figures import ALL_SELECTED

    return [*ALL_SELECTED, UNDERWEAR]


@functools.cache
def _reference_colours(style_key: str) -> tuple[tuple[str, Dominant], ...]:
    from nss.generate import qc_gates

    return tuple(
        (p.name, dominant_colour(p)) for p in qc_gates.reference_paths_for_style(style_key)
    )


def raw_threshold(style_key: str) -> float:
    """The style's unshrunk p90 threshold."""
    return threshold_from([d.lab for _n, d in _reference_colours(style_key)])


@functools.lru_cache(maxsize=1)
def global_threshold() -> float:
    """The median of the raw per-style thresholds over the calibrated styles."""
    return float(np.median([raw_threshold(s) for s in calibrated_styles()]))


def profile(style_key: str) -> dict[str, Any]:
    """The style's reference colours and its raw and shrunk thresholds."""
    refs = _reference_colours(style_key)
    labs = [d.lab for _n, d in refs]
    raw = threshold_from(labs)
    return {
        "n_references": len(refs),
        "raw_threshold": raw,
        "threshold": shrink(raw, len(refs), global_threshold()),
        "labs": labs,
    }


def verdict_for(distance: float, threshold: float, band: float | None = None) -> str:
    """M3: pass / fail / escalate for a distance against a threshold and a band half-width."""
    band = W_BAND if band is None else band
    if band is None:
        return PASS if distance <= threshold else FAIL
    if distance <= threshold - band:
        return PASS
    if distance >= threshold + band:
        return FAIL
    return ESCALATE


def check(concept_path: Path | str, style_key: str) -> dict[str, Any]:
    """Measured colour check of one concept against its style (shrunk threshold, ESCALATE band)."""
    prof = profile(style_key)
    dom = dominant_colour(concept_path)
    nearest = min(delta_e(dom.lab, ref) for ref in prof["labs"])
    return {
        "verdict": verdict_for(nearest, prof["threshold"]),
        "pass": bool(nearest <= prof["threshold"]),  # the binary reading, without the band
        "nearest_delta_e": nearest,
        "threshold": prof["threshold"],
        "raw_threshold": prof["raw_threshold"],
        "band": W_BAND,
        "dominant_lab": dom.lab,
        "used_fallback": dom.used_fallback,
        "mask_fraction": dom.mask_fraction,
        "n_references": prof["n_references"],
    }


def loo_pass_rate(style_key: str, shrunk: bool = True) -> dict[str, Any]:
    """Nested leave-one-out on the real articles (threshold never sees the held-out one).

    With `shrunk`, the held-out article is removed (n - 1), the style's p90 is recomputed without it
    and the global median is recomputed with that leave-one-out p90 in place of the style's raw one.
    """
    refs = _reference_colours(style_key)
    labs = [d.lab for _n, d in refs]
    others = [raw_threshold(s) for s in calibrated_styles() if s != style_key]
    passed = 0
    for i, held in enumerate(labs):
        rest = labs[:i] + labs[i + 1 :]
        raw = threshold_from(rest)
        thr = shrink(raw, len(rest), float(np.median([*others, raw]))) if shrunk else raw
        passed += min(delta_e(held, r) for r in rest) <= thr
    return {
        "style": style_key,
        "n": len(labs),
        "passed": passed,
        "pass_rate": passed / len(labs),
        "fallbacks": sum(d.used_fallback for _n, d in refs),
        "shrunk": shrunk,
    }
