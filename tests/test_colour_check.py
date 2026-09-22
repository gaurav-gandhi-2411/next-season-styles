from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from nss.generate import colour_check as cc


def _flat_lay(path: Path, garment: tuple[int, int, int], bg: tuple[int, int, int]) -> Path:
    img = Image.new("RGB", (300, 400), bg)
    img.paste(Image.new("RGB", (160, 220), garment), (70, 90))
    img.save(path)
    return path


def test_dominant_colour_is_the_garment_not_the_background(tmp_path: Path) -> None:
    dom = cc.dominant_colour(
        _flat_lay(tmp_path / "red.png", (200, 30, 30), (200, 200, 200)), masker="border"
    )
    assert not dom.used_fallback and dom.mask_fraction > 0.1
    other = cc.dominant_colour(
        _flat_lay(tmp_path / "r2.png", (190, 35, 35), (230, 230, 230)), masker="border"
    )
    assert cc.delta_e(dom.lab, other.lab) < 5  # the same red on another background


def test_red_is_far_from_green_and_orange_is_closer_than_green(tmp_path: Path) -> None:
    def dom(name: str, rgb: tuple[int, int, int]) -> cc.Dominant:
        return cc.dominant_colour(_flat_lay(tmp_path / name, rgb, (210, 210, 210)), masker="border")

    red, orange, green = (
        dom("r.png", (200, 30, 30)),
        dom("o.png", (230, 110, 30)),
        dom("g.png", (20, 150, 70)),
    )
    assert cc.delta_e(red.lab, orange.lab) < cc.delta_e(red.lab, green.lab)
    assert cc.delta_e(red.lab, green.lab) > 30


def test_an_empty_mask_falls_back_to_the_centre_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cc, "garment_mask_rembg", lambda img: np.zeros((img.height, img.width), bool)
    )
    dom = cc.dominant_colour(_flat_lay(tmp_path / "w.png", (248, 248, 248), (250, 250, 250)))
    assert dom.used_fallback and dom.lab[0] > 90


def test_rembg_mask_is_cleaned_to_the_largest_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake(img: Image.Image) -> np.ndarray:
        m = np.zeros((img.height, img.width), bool)
        m[40:200, 40:200] = True  # the garment
        m[5:12, 5:12] = True  # a speck that must be dropped
        return m

    monkeypatch.setattr(cc, "garment_mask_rembg", fake)
    dom = cc.dominant_colour(_flat_lay(tmp_path / "x.png", (200, 30, 30), (200, 200, 200)))
    assert not dom.used_fallback and 0.1 < dom.mask_fraction < 0.5


def test_threshold_is_the_p90_of_real_nearest_sibling_distances() -> None:
    labs = [(50.0, 40.0, 30.0 + i * 0.5) for i in range(10)] + [(50.0, 40.0, 60.0)]
    dists = cc.nearest_sibling_distances(labs)
    assert cc.threshold_from(labs) == pytest.approx(float(np.percentile(dists, 90)))


def test_shrinkage_weight_is_n_over_n_plus_k() -> None:
    assert cc.K_SHRINKAGE == 17.0  # the median reference count of 17, 25, 19, 8, 4
    assert cc.shrink(10.0, 17, 4.0) == pytest.approx(7.0)  # w = 0.5
    assert cc.shrink(10.0, 8, 4.0) == pytest.approx(4.0 + 6.0 * 8 / 25)  # sparse: pulled to 4
    assert cc.shrink(10.0, 10_000, 4.0) == pytest.approx(10.0, abs=0.02)


def test_band_gives_pass_fail_or_escalate() -> None:
    assert cc.verdict_for(2.0, 4.5, band=1.0) == cc.PASS
    assert cc.verdict_for(3.5, 4.5, band=1.0) == cc.PASS  # exactly on the edge passes
    assert cc.verdict_for(4.5, 4.5, band=1.0) == cc.ESCALATE
    assert cc.verdict_for(5.4, 4.5, band=1.0) == cc.ESCALATE
    assert cc.verdict_for(5.5, 4.5, band=1.0) == cc.FAIL
    assert cc.verdict_for(4.6, 4.5, band=0.0) == cc.FAIL  # zero band: the binary rule
    assert cc.verdict_for(4.5, 4.5, band=0.0) == cc.PASS
    assert cc.verdict_for(4.6, 4.5) == cc.ESCALATE  # default band is the measured W_BAND


# N4: colour-histogram distance for the patterned class


def test_pattern_class_from_graphical_appearance() -> None:
    assert cc.pattern_class("Ladieswear || Dress || Dresses Ladies || Red || Solid") == "solid"
    assert cc.pattern_class("Ladieswear || Sweater || Knitwear || Beige || Melange") == "solid"
    assert (
        cc.pattern_class("Ladieswear || Bikini top || Swimwear || Orange || All over pattern")
        == "patterned"
    )
    assert cc.pattern_class("Sport || Leggings/Tights || Jersey Fancy || Black || Stripe") == "patterned"


def test_histogram_distance_is_zero_for_identical_and_positive_for_different(tmp_path: Path) -> None:
    def hist(name: str, rgb: tuple[int, int, int]) -> cc.ColourHistogram:
        return cc.colour_histogram(_flat_lay(tmp_path / name, rgb, (210, 210, 210)), masker="border")

    red_a, red_b, green = hist("a.png", (200, 30, 30)), hist("b.png", (198, 32, 28)), hist(
        "c.png", (20, 150, 70)
    )
    assert cc.histogram_distance(red_a, red_a) == pytest.approx(0.0, abs=1e-9)
    assert cc.histogram_distance(red_a, green) > cc.histogram_distance(red_a, red_b)


def test_threshold_from_hist_is_the_p90_of_nearest_sibling_histogram_distances() -> None:
    def onehot(bin_l: int, bin_a: int, bin_b: int) -> cc.ColourHistogram:
        l_hist, a_hist, b_hist = (np.zeros(cc.HIST_BINS) for _ in range(3))
        l_hist[bin_l], a_hist[bin_a], b_hist[bin_b] = 1.0, 1.0, 1.0
        return cc.ColourHistogram(l_hist, a_hist, b_hist, False, 1.0)

    hists = [onehot(16, 16, 16 + i) for i in range(6)]
    dists = cc.nearest_sibling_distances_hist(hists)
    assert cc.threshold_from_hist(hists) == pytest.approx(float(np.percentile(dists, 90)))


def test_check_dispatches_by_pattern_class(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(
        cc, "_check_dominant", lambda p, s: calls.append(("dominant", s)) or {"method": "dominant"}
    )
    monkeypatch.setattr(
        cc, "_check_histogram", lambda p, s: calls.append(("histogram", s)) or {"method": "histogram"}
    )
    solid, patterned = (
        "Ladieswear || Dress || Dresses Ladies || Red || Solid",
        "Ladieswear || Bikini top || Swimwear || Orange || All over pattern",
    )
    assert cc.check("x.png", solid)["method"] == "dominant"
    assert cc.check("x.png", patterned)["method"] == "histogram"
    assert calls == [("dominant", solid), ("histogram", patterned)]


def test_check_histogram_fails_loudly_if_the_band_is_not_yet_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cc, "W_BAND_PATTERNED", None)
    with pytest.raises(ValueError, match="W_BAND_PATTERNED"):
        cc._check_histogram("irrelevant.png", "any || style || key || here || All over pattern")


def test_catalogue_class_prior_fails_loudly_when_not_yet_computed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cc.catalogue_class_prior.cache_clear()
    monkeypatch.setattr(cc, "CATALOGUE_PRIORS_PATH", tmp_path / "missing.csv")
    with pytest.raises(ValueError, match="colour_catalogue_priors"):
        cc.catalogue_class_prior("patterned")
    cc.catalogue_class_prior.cache_clear()
