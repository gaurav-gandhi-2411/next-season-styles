from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from nss.generate import v3_judge_reread as mod


def test_every_live_reading_is_cached_and_reused(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mod, "CACHE", tmp_path / "cache.jsonl")
    monkeypatch.setattr(mod, "GAP_SECONDS", 0.0)
    calls: list[str] = []

    def fake_read(image: Path, dims: tuple[str, ...]) -> dict[str, str]:
        calls.append(str(image))
        return {d: "x" for d in dims}

    monkeypatch.setattr(mod.gemini_panel, "read", fake_read)
    first = mod._read_or_stop("a.png", 0, ("product_type",))
    again = mod._read_or_stop("a.png", 0, ("product_type",))
    assert first == again == {"product_type": "x"}
    assert len(calls) == 1  # the second call came from the cache, no request spent
    assert len((tmp_path / "cache.jsonl").read_text().splitlines()) == 1


def test_a_429_stops_cleanly_and_keeps_what_was_cached(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mod, "CACHE", tmp_path / "cache.jsonl")
    monkeypatch.setattr(mod, "GAP_SECONDS", 0.0)
    monkeypatch.setattr(mod.gemini_panel, "read", lambda image, dims: {d: "x" for d in dims})
    mod._read_or_stop("a.png", 0, ("product_type",))

    def quota(image: Path, dims: tuple[str, ...]) -> dict[str, str]:
        raise RuntimeError("429 RESOURCE_EXHAUSTED: 20 requests per day")

    monkeypatch.setattr(mod.gemini_panel, "read", quota)
    with pytest.raises(mod.QuotaStop):
        mod._read_or_stop("a.png", 1, ("product_type",))
    kept = [json.loads(line) for line in (tmp_path / "cache.jsonl").read_text().splitlines()]
    assert [(r["image"], r["reading"]) for r in kept] == [("a.png", 0)]


def test_a_non_quota_error_is_not_swallowed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mod, "CACHE", tmp_path / "cache.jsonl")

    def boom(image: Path, dims: tuple[str, ...]) -> dict[str, str]:
        raise RuntimeError("network unreachable")

    monkeypatch.setattr(mod.gemini_panel, "read", boom)
    with pytest.raises(RuntimeError, match="network"):
        mod._read_or_stop("a.png", 0, ("product_type",))


def test_kappa_table_reports_n_and_agreement() -> None:
    rows = []
    for i in range(6):
        for judge, score in (("a", 1.0 if i < 3 else 0.0), ("b", 1.0 if i < 3 else 0.0)):
            rows.append({"image": f"i{i}", "dim": "d", "judge": judge, "score": score})
    out = mod.kappa_table(pl.DataFrame(rows))
    r = out.row(0, named=True)
    assert r["n_items"] == 6 and r["raw_agreement"] == 1.0 and r["kappa"] == pytest.approx(1.0)
