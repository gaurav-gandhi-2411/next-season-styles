from __future__ import annotations

import json
from pathlib import Path

import pytest

from nss.viz import demo_explorer

DATA = Path("reports/tables/explorer_styles.json")


@pytest.mark.skipif(not DATA.exists(), reason="requires the committed explorer data")
def test_explorer_rows_have_everything_the_page_reads() -> None:
    rows = json.loads(DATA.read_text(encoding="utf-8"))
    assert len(rows) >= 200
    need = {"key", "rank", "pred", "trail", "growth", "guard", "season", "shap", "traj", "concept"}
    for r in rows:
        assert need <= set(r)
        assert len(r["shap"]) == 5 and all(len(x) == 2 for x in r["shap"])
        assert r["season"] in {"spring", "summer", "autumn", "winter"}
        assert set(r["guard"]) == {"n_active", "price_index", "weeks_active"}
    assert sum(r["concept"] for r in rows) == 3  # the three autumn/winter concept styles
    assert [r["rank"] for r in rows] == sorted(r["rank"] for r in rows)


@pytest.mark.skipif(not DATA.exists(), reason="requires the committed explorer data")
def test_payload_cannot_break_out_of_its_script_tag() -> None:
    blob = demo_explorer.payload({"a": "</script><b>x"})
    assert "</" not in blob  # escaped so embedded data can never close the script element
    parsed = json.loads(blob.replace("<\\/", "</"))
    assert parsed["concepts"]["a"] == "</script><b>x"


def test_page_script_touches_data_only_through_text_nodes() -> None:
    js = demo_explorer.JS
    assert "innerHTML" not in js and "outerHTML" not in js and "document.write" not in js
    assert "fetch(" not in js and "XMLHttpRequest" not in js
