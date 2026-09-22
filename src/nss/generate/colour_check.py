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

**N4**: dominant colour is one number; a pattern is a distribution. For styles in the
**patterned** class (`pattern_class`; solid = `{Solid, Melange}` in `graphical_appearance_name`,
everything else patterned -- `reports/v3/PREREGISTRATION.md` section N), the check instead uses
the sum of three per-channel 1-D Wasserstein distances between the garment's Lab colour histogram
and the style's reference histograms (`histogram_distance`), thresholded and shrunk the same way
(p90 of real nearest-sibling distances, shrunk toward `N3`'s catalogue-wide patterned-class prior,
`scripts/colour_catalogue_priors.py`), with its own re-derived ESCALATE band (`W_BAND_PATTERNED`,
`scripts/colour_histogram_noise.py`). Solid-class styles are untouched by N4.
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
from scipy.stats import wasserstein_distance
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

# N4: patterned-class colour histogram (PREREGISTRATION.md section N)
SOLID_APPEARANCES = frozenset({"Solid", "Melange"})  # colour-stable per the M3 mask-noise evidence
HIST_BINS = 32
L_RANGE = (0.0, 100.0)
AB_RANGE = (-100.0, 100.0)
CATALOGUE_PRIORS_PATH = Path("reports/tables/v3_catalogue_colour_priors_by_class.csv")
W_BAND_PATTERNED: float | None = (
    36.962  # N4: p95 of the pooled histogram-distance jitter deviation (colour_histogram_noise.py)
)


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


def _masked_lab_pixels(img: Image.Image, masker: str = "rembg") -> tuple[np.ndarray, bool, float]:
    """The garment's Lab pixels under the M2 mask (steps shared by dominant colour and histogram
    extraction): resize, mask, largest component, the 3% central-box fallback (counted)."""
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
    return lab[mask], fallback, fraction


def dominant_from_image(img: Image.Image, masker: str = "rembg") -> Dominant:
    """The dominant garment colour of a PIL image."""
    pixels, fallback, fraction = _masked_lab_pixels(img, masker)
    if len(pixels) > MAX_PIXELS:
        pixels = pixels[:: len(pixels) // MAX_PIXELS]
    km = KMeans(n_clusters=min(K, len(pixels)), n_init=10, random_state=SEED).fit(pixels)
    centre = km.cluster_centers_[int(np.argmax(np.bincount(km.labels_)))]
    return Dominant((float(centre[0]), float(centre[1]), float(centre[2])), fallback, fraction)


def dominant_colour(path: Path | str, masker: str = "rembg") -> Dominant:
    """The dominant garment colour of one image file."""
    return dominant_from_image(Image.open(path), masker)


def pattern_class(style_key: str) -> str:
    """N3/N4 class: `graphical_appearance_name` (the style key's last ' || '-joined segment) in
    `SOLID_APPEARANCES` is 'solid'; every other value is 'patterned'. See PREREGISTRATION.md
    section N for why Melange is grouped with Solid (the M3 mask-noise evidence)."""
    appearance = style_key.rsplit(" || ", 1)[-1]
    return "solid" if appearance in SOLID_APPEARANCES else "patterned"


@dataclass(frozen=True)
class ColourHistogram:
    """N4: a garment's per-channel Lab colour histogram and how it was obtained."""

    l_hist: np.ndarray
    a_hist: np.ndarray
    b_hist: np.ndarray
    used_fallback: bool
    mask_fraction: float


def _normalised_hist(values: np.ndarray, lo: float, hi: float, bins: int) -> np.ndarray:
    counts, _ = np.histogram(values, bins=bins, range=(lo, hi))
    total = counts.sum()
    return counts / total if total else counts.astype(np.float64)


def _bin_centres(lo: float, hi: float, bins: int) -> np.ndarray:
    edges = np.linspace(lo, hi, bins + 1)
    return (edges[:-1] + edges[1:]) / 2


def colour_histogram_from_image(
    img: Image.Image, masker: str = "rembg", bins: int = HIST_BINS
) -> ColourHistogram:
    """N4: the garment's per-channel (L, a, b) normalised colour histogram of a PIL image."""
    pixels, fallback, fraction = _masked_lab_pixels(img, masker)
    return ColourHistogram(
        l_hist=_normalised_hist(pixels[:, 0], *L_RANGE, bins),
        a_hist=_normalised_hist(pixels[:, 1], *AB_RANGE, bins),
        b_hist=_normalised_hist(pixels[:, 2], *AB_RANGE, bins),
        used_fallback=fallback,
        mask_fraction=fraction,
    )


def colour_histogram(
    path: Path | str, masker: str = "rembg", bins: int = HIST_BINS
) -> ColourHistogram:
    """N4: the garment's colour histogram of one image file."""
    return colour_histogram_from_image(Image.open(path), masker, bins)


def histogram_distance(h1: ColourHistogram, h2: ColourHistogram, bins: int = HIST_BINS) -> float:
    """N4: sum of the three per-channel 1-D Wasserstein (earth-mover's) distances -- a tractable
    simplification of the joint 3-D EMD between two Lab colour histograms."""
    l_c, ab_c = _bin_centres(*L_RANGE, bins), _bin_centres(*AB_RANGE, bins)
    return float(
        wasserstein_distance(l_c, l_c, h1.l_hist, h2.l_hist)
        + wasserstein_distance(ab_c, ab_c, h1.a_hist, h2.a_hist)
        + wasserstein_distance(ab_c, ab_c, h1.b_hist, h2.b_hist)
    )


def nearest_sibling_distances_hist(hists: list[ColourHistogram]) -> list[float]:
    """N4: for each histogram, the minimum `histogram_distance` to the others."""
    return [
        min(histogram_distance(a, b) for j, b in enumerate(hists) if j != i)
        for i, a in enumerate(hists)
    ]


def threshold_from_hist(hists: list[ColourHistogram]) -> float:
    """N4: the raw p90 of the real nearest-sibling histogram distances."""
    return float(np.percentile(nearest_sibling_distances_hist(hists), PERCENTILE))


@functools.lru_cache(maxsize=1)
def catalogue_class_prior(cls: str) -> float:
    """N3: the catalogue-wide median raw dominant-colour threshold for pattern class `cls`
    ('solid' or 'patterned'), over the 1,980 autumn forecast-eligible styles."""
    import polars as pl

    if not CATALOGUE_PRIORS_PATH.exists():
        raise ValueError(
            f"{CATALOGUE_PRIORS_PATH} does not exist; run scripts/colour_catalogue_priors.py first"
        )
    df = pl.read_csv(CATALOGUE_PRIORS_PATH).filter(pl.col("class") == cls)
    if df.height == 0:
        raise ValueError(f"no catalogue prior computed for class {cls!r}")
    return float(df["median_threshold"][0])


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


@functools.cache
def _reference_histograms(style_key: str) -> tuple[tuple[str, ColourHistogram], ...]:
    from nss.generate import qc_gates

    return tuple(
        (p.name, colour_histogram(p)) for p in qc_gates.reference_paths_for_style(style_key)
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


def profile_hist(style_key: str) -> dict[str, Any]:
    """N4: the patterned style's reference histograms and its raw and shrunk (toward the N3
    catalogue-wide patterned-class prior) histogram-distance thresholds."""
    refs = _reference_histograms(style_key)
    hists = [h for _n, h in refs]
    raw = threshold_from_hist(hists)
    return {
        "n_references": len(refs),
        "raw_threshold": raw,
        "threshold": shrink(raw, len(refs), catalogue_class_prior("patterned")),
        "hists": hists,
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
    """Measured colour check of one concept against its style. Dominant-colour distance and
    shrinkage (M1-M3) for solid-class styles; N4's colour-histogram distance for patterned-class
    styles (`pattern_class`). Solid-class behaviour is byte-for-byte unchanged by N4."""
    if pattern_class(style_key) == "patterned":
        return _check_histogram(concept_path, style_key)
    return _check_dominant(concept_path, style_key)


def _check_dominant(concept_path: Path | str, style_key: str) -> dict[str, Any]:
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
        "method": "dominant",
    }


def _check_histogram(concept_path: Path | str, style_key: str) -> dict[str, Any]:
    if W_BAND_PATTERNED is None:
        raise ValueError(
            "W_BAND_PATTERNED is not set; run scripts/colour_histogram_noise.py and hardcode its "
            "p95 (the M3/W_BAND convention) before scoring a patterned-class style"
        )
    prof = profile_hist(style_key)
    ch = colour_histogram(concept_path)
    nearest = min(histogram_distance(ch, ref) for ref in prof["hists"])
    return {
        "verdict": verdict_for(nearest, prof["threshold"], band=W_BAND_PATTERNED),
        "pass": bool(nearest <= prof["threshold"]),
        "nearest_delta_e": nearest,  # same key name as _check_dominant: it is the statistic's value
        "threshold": prof["threshold"],
        "raw_threshold": prof["raw_threshold"],
        "band": W_BAND_PATTERNED,
        "dominant_lab": None,
        "used_fallback": ch.used_fallback,
        "mask_fraction": ch.mask_fraction,
        "n_references": prof["n_references"],
        "method": "histogram",
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


def loo_pass_rate_hist(style_key: str) -> dict[str, Any]:
    """N4: nested leave-one-out for the histogram statistic. Unlike `loo_pass_rate`, the shrinkage
    target (N3's catalogue-wide patterned-class prior) does not depend on the calibrated styles, so
    it needs no recomputation per fold -- it is held fixed, as it was computed independently."""
    refs = _reference_histograms(style_key)
    hists = [h for _n, h in refs]
    prior = catalogue_class_prior("patterned")
    passed = 0
    for i, held in enumerate(hists):
        rest = hists[:i] + hists[i + 1 :]
        raw = threshold_from_hist(rest)
        thr = shrink(raw, len(rest), prior)
        passed += min(histogram_distance(held, r) for r in rest) <= thr
    return {
        "style": style_key,
        "n": len(hists),
        "passed": passed,
        "pass_rate": passed / len(hists),
        "fallbacks": sum(h.used_fallback for _n, h in refs),
    }
