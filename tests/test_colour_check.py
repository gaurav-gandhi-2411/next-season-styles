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
    dom = cc.dominant_colour(_flat_lay(tmp_path / "red.png", (200, 30, 30), (200, 200, 200)))
    assert not dom.used_fallback and dom.mask_fraction > 0.1
    a_red = cc.dominant_colour(_flat_lay(tmp_path / "r2.png", (190, 35, 35), (230, 230, 230)))
    assert cc.delta_e(dom.lab, a_red.lab) < 5  # the same red on another background


def test_red_is_far_from_green_and_orange_is_closer_than_green(tmp_path: Path) -> None:
    red = cc.dominant_colour(_flat_lay(tmp_path / "r.png", (200, 30, 30), (210, 210, 210)))
    orange = cc.dominant_colour(_flat_lay(tmp_path / "o.png", (230, 110, 30), (210, 210, 210)))
    green = cc.dominant_colour(_flat_lay(tmp_path / "g.png", (20, 150, 70), (210, 210, 210)))
    assert cc.delta_e(red.lab, orange.lab) < cc.delta_e(red.lab, green.lab)
    assert cc.delta_e(red.lab, green.lab) > 30


def test_white_on_white_falls_back_to_the_centre_and_says_so(tmp_path: Path) -> None:
    dom = cc.dominant_colour(_flat_lay(tmp_path / "w.png", (248, 248, 248), (250, 250, 250)))
    assert dom.used_fallback and dom.lab[0] > 90


def test_threshold_is_the_p90_of_real_nearest_sibling_distances() -> None:
    labs = [(50.0, 40.0, 30.0 + i * 0.5) for i in range(10)] + [(50.0, 40.0, 60.0)]
    dists = cc.nearest_sibling_distances(labs)
    assert cc.threshold_from(labs) == pytest.approx(float(np.percentile(dists, 90)))
    assert max(dists) == dists[-1]  # the outlier is its own farthest nearest-sibling


def test_check_passes_a_colour_inside_the_real_spread_and_fails_a_far_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refs = [
        _flat_lay(tmp_path / f"ref{i}.png", (200 - i, 30 + i, 30), (215, 215, 215))
        for i in range(8)
    ]
    monkeypatch.setattr(
        "nss.generate.qc_gates.reference_paths_for_style", lambda _s: refs, raising=True
    )
    cc._reference_colours.cache_clear()
    ok = cc.check(_flat_lay(tmp_path / "ok.png", (198, 32, 30), (200, 200, 200)), "style-x")
    far = cc.check(_flat_lay(tmp_path / "far.png", (20, 150, 70), (200, 200, 200)), "style-x")
    cc._reference_colours.cache_clear()
    assert ok["pass"] is True and far["pass"] is False
    assert far["nearest_delta_e"] > ok["nearest_delta_e"]
