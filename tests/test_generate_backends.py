from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from nss.generate import backends
from nss.generate.backends import generate_concept


def _fake_images(n: int) -> list[Image.Image]:
    """Small in-memory PIL images standing in for real SDXL/Gemini output."""
    return [Image.new("RGB", (4, 4), color=(i, i, i)) for i in range(n)]


def test_unknown_backend_raises() -> None:
    """An unsupported backend name is rejected before any generation work happens."""
    with pytest.raises(ValueError, match="Unknown backend"):
        generate_concept(
            prompt="p",
            reference_images=[Path("ref.jpg")],
            backend="not_a_real_backend",
            ip_adapter_scale=None,
            seed=42,
            n=1,
        )


def test_empty_reference_images_raises() -> None:
    """Both backends require at least one reference image."""
    with pytest.raises(ValueError, match="reference_images must be non-empty"):
        generate_concept(
            prompt="p",
            reference_images=[],
            backend=backends.LOCAL_SDXL,
            ip_adapter_scale=None,
            seed=42,
            n=1,
        )


def test_n_less_than_one_raises() -> None:
    """n must be a positive image count."""
    with pytest.raises(ValueError, match="n must be >= 1"):
        generate_concept(
            prompt="p",
            reference_images=[Path("ref.jpg")],
            backend=backends.LOCAL_SDXL,
            ip_adapter_scale=None,
            seed=42,
            n=0,
        )


def test_gemini_rejects_non_none_ip_adapter_scale() -> None:
    """ip_adapter_scale has no Gemini equivalent -- a non-None value must fail loudly, not be
    silently dropped while looking honoured."""
    with pytest.raises(ValueError, match="not applicable to the gemini backend"):
        generate_concept(
            prompt="p",
            reference_images=[Path("ref.jpg")],
            backend=backends.GEMINI,
            ip_adapter_scale=0.8,
            seed=42,
            n=1,
        )


def test_local_sdxl_dispatch_writes_images_and_metadata(tmp_path: Path) -> None:
    """generate_concept() routes to the local_sdxl backend, saves n images, and records the
    ip_adapter_scale that was actually passed through (not silently defaulted)."""
    with (
        patch.object(backends, "OUTPUT_ROOT", tmp_path),
        patch.object(
            backends, "_generate_local_sdxl", return_value=(_fake_images(2), 12.3)
        ) as mock_gen,
    ):
        paths = generate_concept(
            prompt="a black cotton t-shirt, product photography",
            reference_images=[Path("data/images/0800691008.jpg")],
            backend=backends.LOCAL_SDXL,
            ip_adapter_scale=0.6,
            seed=42,
            n=2,
        )

    mock_gen.assert_called_once_with(
        "a black cotton t-shirt, product photography",
        [Path("data/images/0800691008.jpg")],
        0.6,
        42,
        2,
        None,
        None,
        None,
    )
    assert len(paths) == 2
    for path in paths:
        assert path.exists()
        assert path.parent == tmp_path / "local_sdxl"

    metadata = json.loads(paths[0].with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["backend"] == "local_sdxl"
    assert metadata["ip_adapter_scale"] == 0.6
    assert metadata["seed"] == 42
    assert metadata["index"] == 0
    assert metadata["negative_prompt"] is None


def test_local_sdxl_passes_negative_prompt_through(tmp_path: Path) -> None:
    """A caller-supplied negative_prompt reaches `_generate_local_sdxl` and the metadata sidecar,
    not just the (already-tested) default-None case above."""
    with (
        patch.object(backends, "OUTPUT_ROOT", tmp_path),
        patch.object(
            backends, "_generate_local_sdxl", return_value=(_fake_images(1), 5.0)
        ) as mock_gen,
    ):
        paths = generate_concept(
            prompt="a red solid underwear bottom, product photography",
            reference_images=[Path("data/images/0803986005.jpg")],
            backend=backends.LOCAL_SDXL,
            ip_adapter_scale=0.2,
            seed=42,
            n=1,
            negative_prompt="person, human, model, face, skin, body, worn",
        )

    mock_gen.assert_called_once_with(
        "a red solid underwear bottom, product photography",
        [Path("data/images/0803986005.jpg")],
        0.2,
        42,
        1,
        "person, human, model, face, skin, body, worn",
        None,
        None,
    )
    metadata = json.loads(paths[0].with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["negative_prompt"] == "person, human, model, face, skin, body, worn"


def test_local_sdxl_passes_prompt_2_and_negative_prompt_2_through(tmp_path: Path) -> None:
    """A caller-supplied prompt_2/negative_prompt_2 (SDXL's second text encoder) reaches
    `_generate_local_sdxl` and the metadata sidecar."""
    with (
        patch.object(backends, "OUTPUT_ROOT", tmp_path),
        patch.object(
            backends, "_generate_local_sdxl", return_value=(_fake_images(1), 5.0)
        ) as mock_gen,
    ):
        paths = generate_concept(
            prompt="a red solid underwear bottom, product photography",
            reference_images=[Path("data/images/0803986005.jpg")],
            backend=backends.LOCAL_SDXL,
            ip_adapter_scale=0.2,
            seed=42,
            n=1,
            negative_prompt="person, human",
            prompt_2="Underwear bottom, red solid.",
            negative_prompt_2="person, human",
        )

    mock_gen.assert_called_once_with(
        "a red solid underwear bottom, product photography",
        [Path("data/images/0803986005.jpg")],
        0.2,
        42,
        1,
        "person, human",
        "Underwear bottom, red solid.",
        "person, human",
    )
    metadata = json.loads(paths[0].with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["prompt_2"] == "Underwear bottom, red solid."
    assert metadata["negative_prompt_2"] == "person, human"


def test_gemini_rejects_non_none_prompt_2() -> None:
    """prompt_2 has no Gemini equivalent -- a non-None value must fail loudly, not be silently
    dropped while looking honoured (same convention as ip_adapter_scale/negative_prompt)."""
    with pytest.raises(ValueError, match="not applicable to the gemini backend"):
        generate_concept(
            prompt="p",
            reference_images=[Path("ref.jpg")],
            backend=backends.GEMINI,
            ip_adapter_scale=None,
            seed=42,
            n=1,
            prompt_2="attributes only",
        )


def test_gemini_rejects_non_none_negative_prompt_2() -> None:
    """negative_prompt_2 has no Gemini equivalent -- same fail-loud convention."""
    with pytest.raises(ValueError, match="not applicable to the gemini backend"):
        generate_concept(
            prompt="p",
            reference_images=[Path("ref.jpg")],
            backend=backends.GEMINI,
            ip_adapter_scale=None,
            seed=42,
            n=1,
            negative_prompt_2="no humans",
        )


def test_gemini_rejects_non_none_negative_prompt() -> None:
    """Gemini's image API has no negative-prompt equivalent -- a non-None value must fail loudly,
    not be silently dropped while looking honoured (same convention as ip_adapter_scale)."""
    with pytest.raises(ValueError, match="not applicable to the gemini backend"):
        generate_concept(
            prompt="p",
            reference_images=[Path("ref.jpg")],
            backend=backends.GEMINI,
            ip_adapter_scale=None,
            seed=42,
            n=1,
            negative_prompt="no humans",
        )


def test_gemini_dispatch_records_ip_adapter_scale_as_null(tmp_path: Path) -> None:
    """The gemini backend has no ip_adapter_scale knob -- metadata must show it explicitly as
    None/null, never silently omitted or faked as a real value."""
    with (
        patch.object(backends, "OUTPUT_ROOT", tmp_path),
        patch.object(backends, "_generate_gemini", return_value=(_fake_images(1), 3.4)),
    ):
        paths = generate_concept(
            prompt="a beige melange knit sweater concept",
            reference_images=[Path("data/images/0863646003.jpg")],
            backend=backends.GEMINI,
            ip_adapter_scale=None,
            seed=7,
            n=1,
        )

    metadata = json.loads(paths[0].with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["backend"] == "gemini"
    assert metadata["ip_adapter_scale"] is None


def test_require_gemini_api_key_raises_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing GEMINI_API_KEY produces an actionable RuntimeError, not a fabricated workaround."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # Avoid picking up a real .env from the actual repo root or CWD during this test.
    with (
        patch("dotenv.load_dotenv", return_value=False),
        pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"),
    ):
        backends._require_gemini_api_key()


def test_require_gemini_api_key_returns_key_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GEMINI_API_KEY present in the environment is returned as-is."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    with patch("dotenv.load_dotenv", return_value=True):
        assert backends._require_gemini_api_key() == "fake-test-key"
