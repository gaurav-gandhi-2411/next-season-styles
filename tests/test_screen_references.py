"""Tests for `nss.generate.screen_references`.

No real GPU/API calls anywhere in this module -- `google.genai.Client`/`groq.Groq` are patched at
their SDK-module source (mirrors `tests/test_vlm_judges.py`'s convention), and the fetch-more /
screening-batch logic is tested with injected fakes (`classify_fn`/`fetch_fn`), never real network
calls.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from nss.generate import screen_references as sr

_STYLE = "Ladieswear || Sweater || Knitwear || Beige || Melange"


def _classification(
    label: str = sr.FRAMING_FULL_GARMENT, *, available: bool = True
) -> dict[str, Any]:
    """Build a fake `classify_reference_image`-shaped result for a given image path/label."""

    def _fake(image_path: Path, groq_available: bool, groq_detail: str = "") -> dict[str, Any]:
        note = f"fake -- label={label} available={available}"
        return {
            "image_path": str(image_path),
            "gemini_available": available,
            "gemini_label": label if available else None,
            "gemini_excluded_reason": None if available else "fake unavailable",
            "groq_available": available,
            "groq_label": label if available else None,
            "groq_excluded_reason": None if available else "fake unavailable",
            "is_full_garment": available and label == sr.FRAMING_FULL_GARMENT,
            "screening_note": note,
        }

    return _fake  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# build_framing_classification_prompt / _parse_framing_label
# ---------------------------------------------------------------------------


def test_build_framing_classification_prompt_mentions_both_labels() -> None:
    prompt = sr.build_framing_classification_prompt()
    assert "full_garment" in prompt
    assert "texture_crop" in prompt


def test_parse_framing_label_valid() -> None:
    assert sr._parse_framing_label('{"framing": "full_garment"}') == "full_garment"
    assert sr._parse_framing_label('{"framing": "texture_crop"}') == "texture_crop"


def test_parse_framing_label_empty_raises() -> None:
    with pytest.raises(RuntimeError, match="empty"):
        sr._parse_framing_label(None)


def test_parse_framing_label_malformed_json_raises() -> None:
    with pytest.raises(RuntimeError, match="not valid JSON"):
        sr._parse_framing_label("not json")


def test_parse_framing_label_missing_key_raises() -> None:
    with pytest.raises(RuntimeError, match="missing 'framing' key"):
        sr._parse_framing_label('{"other": "x"}')


def test_parse_framing_label_unrecognized_value_raises() -> None:
    with pytest.raises(RuntimeError, match="unrecognized framing label"):
        sr._parse_framing_label('{"framing": "close-up-ish"}')


# ---------------------------------------------------------------------------
# classify_framing_gemini / classify_framing_groq -- mocked SDK clients
# ---------------------------------------------------------------------------


def test_classify_framing_gemini_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with patch("dotenv.load_dotenv", return_value=False):
        result = sr.classify_framing_gemini(Path("irrelevant.jpg"), api_key=None)
    assert result["available"] is False
    assert "GEMINI_API_KEY is not set" in result["excluded_reason"]


def test_classify_framing_gemini_success(tmp_path: Path) -> None:
    from PIL import Image

    image_path = tmp_path / "ref.jpg"
    Image.new("RGB", (8, 8)).save(image_path)

    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = MagicMock(
            text='{"framing": "texture_crop"}'
        )
        result = sr.classify_framing_gemini(image_path, api_key="fake-key")

    assert result["available"] is True
    assert result["label"] == "texture_crop"
    assert result["excluded_reason"] is None


def test_classify_framing_gemini_exception_is_soft_failure(tmp_path: Path) -> None:
    """Any Gemini failure (network, malformed response) becomes `available=False`, never raised --
    this is a screening filter, not the fail-loud generation path."""
    from PIL import Image

    image_path = tmp_path / "ref.jpg"
    Image.new("RGB", (8, 8)).save(image_path)

    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.side_effect = RuntimeError("network exploded")
        result = sr.classify_framing_gemini(image_path, api_key="fake-key")

    assert result["available"] is False
    assert "network exploded" in result["excluded_reason"]


def test_classify_framing_groq_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with patch("dotenv.load_dotenv", return_value=False):
        result = sr.classify_framing_groq(Path("irrelevant.jpg"), api_key=None)
    assert result["available"] is False
    assert "GROQ_API_KEY is not set" in result["excluded_reason"]


def test_classify_framing_groq_success(tmp_path: Path) -> None:
    image_path = tmp_path / "ref.jpg"
    image_path.write_bytes(b"\xff\xd8\xff" + b"0" * 32)

    with patch("groq.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_completion = MagicMock()
        mock_completion.choices = [
            MagicMock(message=MagicMock(content='{"framing": "full_garment"}'))
        ]
        mock_client.chat.completions.create.return_value = mock_completion
        result = sr.classify_framing_groq(image_path, api_key="fake-key")

    assert result["available"] is True
    assert result["label"] == "full_garment"


def test_classify_framing_groq_exception_is_soft_failure(tmp_path: Path) -> None:
    image_path = tmp_path / "ref.jpg"
    image_path.write_bytes(b"\xff\xd8\xff" + b"0" * 32)

    with patch("groq.Groq") as mock_groq_cls:
        mock_groq_cls.side_effect = RuntimeError("boom")
        result = sr.classify_framing_groq(image_path, api_key="fake-key")

    assert result["available"] is False
    assert "boom" in result["excluded_reason"]


# ---------------------------------------------------------------------------
# _consensus_is_full_garment / classify_reference_image
# ---------------------------------------------------------------------------


def test_consensus_both_available_agree_full_garment() -> None:
    gemini = {"available": True, "label": sr.FRAMING_FULL_GARMENT}
    groq = {"available": True, "label": sr.FRAMING_FULL_GARMENT}
    is_full, note = sr._consensus_is_full_garment(gemini, groq)
    assert is_full is True
    assert "full_garment" in note


def test_consensus_disagreement_fails_closed_to_texture_crop() -> None:
    gemini = {"available": True, "label": sr.FRAMING_FULL_GARMENT}
    groq = {"available": True, "label": sr.FRAMING_TEXTURE_CROP}
    is_full, note = sr._consensus_is_full_garment(gemini, groq)
    assert is_full is False
    assert "texture_crop" in note


def test_consensus_no_judge_available_fails_closed() -> None:
    gemini = {"available": False, "label": None}
    groq = {"available": False, "label": None}
    is_full, note = sr._consensus_is_full_garment(gemini, groq)
    assert is_full is False
    assert "no judge available" in note


def test_consensus_only_one_judge_available_uses_it() -> None:
    gemini = {"available": True, "label": sr.FRAMING_FULL_GARMENT}
    groq = {"available": False, "label": None}
    is_full, _ = sr._consensus_is_full_garment(gemini, groq)
    assert is_full is True


def test_classify_reference_image_skips_groq_when_unavailable(tmp_path: Path) -> None:
    image_path = tmp_path / "ref.jpg"
    image_path.write_bytes(b"x")

    with patch.object(
        sr,
        "classify_framing_gemini",
        return_value={
            "judge_name": "gemini",
            "available": True,
            "label": sr.FRAMING_FULL_GARMENT,
            "raw_response": "{}",
            "excluded_reason": None,
        },
    ):
        result = sr.classify_reference_image(image_path, groq_available=False, groq_detail="down")

    assert result["groq_available"] is False
    assert "down" in result["groq_excluded_reason"]
    assert result["is_full_garment"] is True


# ---------------------------------------------------------------------------
# screen_candidates
# ---------------------------------------------------------------------------


def test_screen_candidates_carries_metadata_through() -> None:
    candidates = [
        {"article_id": 1, "local_image_path": Path("a.jpg"), "units_sold_last_26w": 100},
        {"article_id": 2, "local_image_path": Path("b.jpg"), "units_sold_last_26w": 50},
    ]
    rows = sr.screen_candidates(
        _STYLE,
        candidates,
        groq_available=True,
        classify_fn=_classification(sr.FRAMING_FULL_GARMENT),
    )
    assert len(rows) == 2
    assert rows[0]["style_id"] == _STYLE
    assert rows[0]["article_id"] == 1
    assert rows[0]["units_sold_last_26w"] == 100
    assert rows[0]["newly_fetched"] is False
    assert rows[0]["is_full_garment"] is True


def test_screen_candidates_tags_newly_fetched() -> None:
    candidates = [{"article_id": 9, "local_image_path": Path("c.jpg"), "units_sold_last_26w": 5}]
    rows = sr.screen_candidates(
        _STYLE, candidates, groq_available=True, classify_fn=_classification(), newly_fetched=True
    )
    assert rows[0]["newly_fetched"] is True


# ---------------------------------------------------------------------------
# ensure_min_full_garment_references -- the "fetch more if <3 survive" logic
# ---------------------------------------------------------------------------


def _fake_transactions_and_articles() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build a tiny transactions/articles pair with 10 distinct constituent articles for one
    style, each sold a distinct number of times (so ranking by units sold is deterministic)."""
    article_ids = list(range(101, 111))
    articles = pl.DataFrame(
        {
            "article_id": article_ids,
            "index_group_name": ["Ladieswear"] * 10,
            "product_type_name": ["Sweater"] * 10,
            "garment_group_name": ["Knitwear"] * 10,
            "perceived_colour_master_name": ["Beige"] * 10,
            "graphical_appearance_name": ["Melange"] * 10,
        }
    )
    # Article 101 sells 10 units, 102 sells 9, ..., 110 sells 1.
    rows = []
    for rank, article_id in enumerate(article_ids):
        n_sales = 10 - rank
        rows.extend([{"article_id": article_id, "t_dat": date(2020, 9, 1)}] * n_sales)
    transactions = pl.DataFrame(rows)
    return transactions, articles


_STYLE_VALUES = {
    "index_group_name": "Ladieswear",
    "product_type_name": "Sweater",
    "garment_group_name": "Knitwear",
    "perceived_colour_master_name": "Beige",
    "graphical_appearance_name": "Melange",
}


def test_ensure_min_full_garment_references_noop_when_already_enough() -> None:
    screened = [
        {"is_full_garment": True, "image_path": f"{i}.jpg"}
        for i in range(sr.MIN_SURVIVING_BEFORE_FETCH)
    ]
    transactions, articles = _fake_transactions_and_articles()
    result = sr.ensure_min_full_garment_references(
        _STYLE,
        _STYLE_VALUES,
        screened,
        already_tried_article_ids=set(range(101, 109)),
        groq_available=True,
        groq_detail="",
        transactions=transactions,
        articles=articles,
        window_start=date(2020, 1, 1),
        window_end=date(2020, 9, 21),
    )
    assert result["rounds_used"] == 0
    assert result["exhausted"] is False
    assert result["n_survivors"] == sr.MIN_SURVIVING_BEFORE_FETCH


def test_ensure_min_full_garment_references_fetches_until_target_reached() -> None:
    """Only 1 full-garment survivor initially (below MIN_SURVIVING_BEFORE_FETCH=3); the 2
    already-tried article_ids (101, 102) are excluded from re-fetch; every newly fetched candidate
    classifies as full_garment, so fetching stops once TARGET_FULL_GARMENT_REFERENCES=4 total
    survivors are reached."""
    screened = [{"is_full_garment": True, "image_path": "101.jpg"}]
    already_tried = {101, 102}
    transactions, articles = _fake_transactions_and_articles()

    fetch_calls: list[list[int]] = []

    def fake_fetch(article_ids: list[int], out_dir: Path) -> dict[str, bool]:
        fetch_calls.append(list(article_ids))
        return dict.fromkeys((str(a) for a in article_ids), True)

    result = sr.ensure_min_full_garment_references(
        _STYLE,
        _STYLE_VALUES,
        screened,
        already_tried_article_ids=already_tried,
        groq_available=True,
        groq_detail="",
        transactions=transactions,
        articles=articles,
        window_start=date(2020, 1, 1),
        window_end=date(2020, 9, 21),
        images_dir=Path("unused"),
        classify_fn=_classification(sr.FRAMING_FULL_GARMENT),
        fetch_fn=fake_fetch,
    )

    assert result["n_survivors"] >= sr.TARGET_FULL_GARMENT_REFERENCES
    assert result["exhausted"] is False
    assert result["rounds_used"] >= 1
    # article_ids 101/102 (already tried) must never be re-fetched.
    for call in fetch_calls:
        assert 101 not in call
        assert 102 not in call
    assert already_tried >= {101, 102}  # mutated in place, original ids retained


def test_ensure_min_full_garment_references_reports_exhaustion() -> None:
    """If every newly-ranked candidate classifies as texture_crop, fetching keeps trying (up to
    MAX_FETCH_ROUNDS) but never reaches the target; once every distinct constituent article has
    already been tried, `exhausted=True` is reported -- a genuine finding, not padded."""
    screened = [{"is_full_garment": False, "image_path": "101.jpg"}]
    # Pre-mark every distinct article in the fake catalogue as already tried, so the very first
    # fetch-more round finds nothing new and reports exhaustion immediately.
    already_tried = set(range(101, 111))
    transactions, articles = _fake_transactions_and_articles()

    result = sr.ensure_min_full_garment_references(
        _STYLE,
        _STYLE_VALUES,
        screened,
        already_tried_article_ids=already_tried,
        groq_available=True,
        groq_detail="",
        transactions=transactions,
        articles=articles,
        window_start=date(2020, 1, 1),
        window_end=date(2020, 9, 21),
        classify_fn=_classification(sr.FRAMING_TEXTURE_CROP),
        fetch_fn=lambda ids, out_dir: dict.fromkeys((str(a) for a in ids), True),
    )

    assert result["exhausted"] is True
    assert result["n_survivors"] == 0
    assert result["rounds_used"] == 1


# ---------------------------------------------------------------------------
# load_screened_references
# ---------------------------------------------------------------------------


def test_load_screened_references_keeps_only_full_garment_ordered_by_units_sold(
    tmp_path: Path,
) -> None:
    path = tmp_path / "exemplar_images_screened.csv"
    path.write_text(
        "style_id,article_id,image_path,units_sold_last_26w,is_full_garment\n"
        "Style A,1,a1.jpg,50,true\n"
        "Style A,2,a2.jpg,200,true\n"
        "Style A,3,a3.jpg,999,false\n"
        "Style B,4,b1.jpg,10,true\n",
        encoding="utf-8",
    )
    grouped = sr.load_screened_references(path)
    assert set(grouped) == {"Style A", "Style B"}
    # Style A: full-garment survivors only, best-selling (higher units_sold) first.
    assert grouped["Style A"] == [Path("a2.jpg"), Path("a1.jpg")]
    assert grouped["Style B"] == [Path("b1.jpg")]


def test_load_screened_references_raises_when_style_has_zero_survivors(tmp_path: Path) -> None:
    path = tmp_path / "exemplar_images_screened.csv"
    path.write_text(
        "style_id,article_id,image_path,units_sold_last_26w,is_full_garment\n"
        "Style A,1,a1.jpg,50,false\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no full-garment survivors"):
        sr.load_screened_references(path)


# ---------------------------------------------------------------------------
# retry_inconclusive_candidates -- the targeted, quota-cheap re-screening pass
# ---------------------------------------------------------------------------


def _write_screened_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pl.DataFrame(rows).write_csv(path)


def test_retry_inconclusive_candidates_reclassifies_only_inconclusive_rows(tmp_path: Path) -> None:
    path = tmp_path / "exemplar_images_screened.csv"
    _write_screened_csv(
        path,
        [
            {
                "style_id": "Style A",
                "article_id": 1,
                "image_path": "a1.jpg",
                "units_sold_last_26w": 50,
                "newly_fetched": False,
                "gemini_available": False,
                "gemini_label": None,
                "gemini_excluded_reason": "quota",
                "groq_available": False,
                "groq_label": None,
                "groq_excluded_reason": "quota",
                "is_full_garment": False,
                "screening_note": "no judge available -- screening inconclusive, excluded",
            },
            {
                "style_id": "Style A",
                "article_id": 2,
                "image_path": "a2.jpg",
                "units_sold_last_26w": 40,
                "newly_fetched": False,
                "gemini_available": False,
                "gemini_label": None,
                "gemini_excluded_reason": None,
                "groq_available": True,
                "groq_label": "full_garment",
                "groq_excluded_reason": None,
                "is_full_garment": True,
                "screening_note": "already resolved",
            },
        ],
    )
    calls: list[str] = []

    def fake_classify(image_path: Path, groq_available: bool, groq_detail: str) -> dict[str, Any]:
        calls.append(str(image_path))
        return {
            "image_path": str(image_path),
            "gemini_available": False,
            "gemini_label": None,
            "gemini_excluded_reason": "still quota",
            "groq_available": True,
            "groq_label": "full_garment",
            "groq_excluded_reason": None,
            "is_full_garment": True,
            "screening_note": "retried successfully",
        }

    out_df = sr.retry_inconclusive_candidates(path, classify_fn=fake_classify)

    # Only the inconclusive row (article_id=1) was re-classified -- the already-resolved row
    # (article_id=2) must never be re-queried (that would waste scarce quota for no reason).
    assert calls == ["a1.jpg"]
    row1 = out_df.filter(pl.col("article_id") == 1).to_dicts()[0]
    assert row1["is_full_garment"] is True
    assert row1["screening_note"] == "retried successfully"
    row2 = out_df.filter(pl.col("article_id") == 2).to_dicts()[0]
    assert row2["screening_note"] == "already resolved"  # untouched


def test_retry_inconclusive_candidates_skips_newly_fetched_rows_by_default(tmp_path: Path) -> None:
    path = tmp_path / "exemplar_images_screened.csv"
    _write_screened_csv(
        path,
        [
            {
                "style_id": "Style A",
                "article_id": 1,
                "image_path": "a1.jpg",
                "units_sold_last_26w": 50,
                "newly_fetched": True,  # from an over-eager fetch-more round
                "gemini_available": False,
                "gemini_label": None,
                "gemini_excluded_reason": "quota",
                "groq_available": False,
                "groq_label": None,
                "groq_excluded_reason": "quota",
                "is_full_garment": False,
                "screening_note": "inconclusive",
            },
        ],
    )
    calls: list[str] = []

    def fake_classify(image_path: Path, groq_available: bool, groq_detail: str) -> dict[str, Any]:
        calls.append(str(image_path))
        raise AssertionError("must not be called for a newly_fetched row by default")

    sr.retry_inconclusive_candidates(path, classify_fn=fake_classify, only_original_fetch=True)
    assert calls == []


def test_retry_inconclusive_candidates_is_idempotent_when_still_quota_blocked(
    tmp_path: Path,
) -> None:
    path = tmp_path / "exemplar_images_screened.csv"
    _write_screened_csv(
        path,
        [
            {
                "style_id": "Style A",
                "article_id": 1,
                "image_path": "a1.jpg",
                "units_sold_last_26w": 50,
                "newly_fetched": False,
                "gemini_available": False,
                "gemini_label": None,
                "gemini_excluded_reason": "quota",
                "groq_available": False,
                "groq_label": None,
                "groq_excluded_reason": "quota",
                "is_full_garment": False,
                "screening_note": "no judge available",
            },
        ],
    )

    def still_unavailable(
        image_path: Path, groq_available: bool, groq_detail: str
    ) -> dict[str, Any]:
        return {
            "image_path": str(image_path),
            "gemini_available": False,
            "gemini_label": None,
            "gemini_excluded_reason": "still quota",
            "groq_available": False,
            "groq_label": None,
            "groq_excluded_reason": "still quota",
            "is_full_garment": False,
            "screening_note": "no judge available -- inconclusive, excluded (fail-closed)",
        }

    first = sr.retry_inconclusive_candidates(path, classify_fn=still_unavailable)
    second = sr.retry_inconclusive_candidates(path, classify_fn=still_unavailable)
    assert first.to_dicts() == second.to_dicts()
    assert first.filter(pl.col("article_id") == 1)["is_full_garment"].to_list() == [False]
