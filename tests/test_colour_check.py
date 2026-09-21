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
    assert cc.verdict_for(4.6, 4.5, band=None) == cc.FAIL  # no band: the binary rule
