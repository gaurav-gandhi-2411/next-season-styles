from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from nss.data.fetch_images import fetch_images, fetch_one, local_path, remote_path


def test_remote_path_zero_pads_and_prefixes() -> None:
    """9-digit article_id zero-pads to 10 digits; remote prefix is the first 3 of those."""
    assert remote_path(108775015) == "images/010/0108775015.jpg"


def test_local_path_is_flat() -> None:
    """Local cache mirrors the CLI's flat single-file download layout (no subfolders)."""
    out_dir = Path("data/images")
    assert local_path(108775015, out_dir) == out_dir / "0108775015.jpg"


def test_fetch_one_skips_download_when_cached(tmp_path: Path) -> None:
    """An existing non-empty cached file is reused without calling the Kaggle API."""
    out_dir = tmp_path
    cached = local_path(108775015, out_dir)
    cached.write_bytes(b"fake-jpeg-bytes")

    with patch("nss.data.fetch_images.kaggle.api.competition_download_file") as mock_dl:
        ok = fetch_one(108775015, out_dir)

    assert ok is True
    mock_dl.assert_not_called()


def test_fetch_one_retries_then_succeeds(tmp_path: Path) -> None:
    """A failure followed by a success (which writes the file) is retried, not given up on."""
    out_dir = tmp_path
    dest = local_path(108775015, out_dir)
    calls = {"n": 0}

    def _side_effect(*args: object, **kwargs: object) -> None:
        calls["n"] += 1
        if calls["n"] < 2:
            raise ConnectionError("simulated network error")
        dest.write_bytes(b"fake-jpeg-bytes")

    target = "nss.data.fetch_images.kaggle.api.competition_download_file"
    with patch(target, side_effect=_side_effect), patch("nss.data.fetch_images.time.sleep"):
        ok = fetch_one(108775015, out_dir)

    assert ok is True
    assert calls["n"] == 2
    assert dest.read_bytes() == b"fake-jpeg-bytes"


def test_fetch_one_gives_up_after_max_retries(tmp_path: Path) -> None:
    """Persistent failure (e.g. a 404) is retried exactly `max_retries` times, then reported."""
    out_dir = tmp_path
    calls = {"n": 0}

    def _always_fail(*args: object, **kwargs: object) -> None:
        calls["n"] += 1
        raise RuntimeError("simulated 404")

    target = "nss.data.fetch_images.kaggle.api.competition_download_file"
    with patch(target, side_effect=_always_fail), patch("nss.data.fetch_images.time.sleep"):
        ok = fetch_one(9999999999, out_dir, max_retries=3)

    assert ok is False
    assert calls["n"] == 3
    assert not local_path(9999999999, out_dir).exists()


def test_fetch_images_reports_per_id_success_and_failure(tmp_path: Path) -> None:
    """Batch fetch returns a per-article_id success map; one bad id doesn't kill the batch."""
    out_dir = tmp_path
    good_id, bad_id = 108775015, 9999999999

    def _side_effect(competition: str, file_name: str, **kwargs: object) -> None:
        if file_name == remote_path(good_id):
            local_path(good_id, out_dir).write_bytes(b"fake-jpeg-bytes")
        else:
            raise RuntimeError("simulated 404")

    target = "nss.data.fetch_images.kaggle.api.competition_download_file"
    with patch(target, side_effect=_side_effect), patch("nss.data.fetch_images.time.sleep"):
        results = fetch_images([good_id, bad_id], out_dir)

    assert results == {str(good_id): True, str(bad_id): False}
