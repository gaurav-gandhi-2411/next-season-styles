"""Tests for `nss.generate.vlm_judges`: the Groq judge adapter, mocked end-to-end.

No real API calls anywhere in this module -- `groq.Groq` (imported locally inside
`check_groq_availability`/`extract_attributes_groq`) is patched at its `groq.Groq` source so the
local `from groq import Groq` binds to the mock regardless of import timing. The live-discovery
work that picked `GROQ_JUDGE_MODEL_ID` (querying `GET /openai/v1/models` and a real,
one-off end-to-end vision call) is documented in `vlm_judges.py`'s module docstring --
deliberately NOT repeated here, since this suite must never hit a real API.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from groq import NotFoundError

from nss.generate import vlm_judges

_DIMENSIONS = vlm_judges.ATTRIBUTE_DIMENSIONS
_FAKE_IMAGE_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 32  # not a real PNG -- never decoded here


def _not_found_error(message: str = "model not found") -> NotFoundError:
    """Build a real `groq.NotFoundError` (needs a real `httpx.Response`, not a bare Mock)."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(404, request=request)
    return NotFoundError(message, response=response, body=None)


def _mock_chat_completion(content: str) -> MagicMock:
    """Build a mock `ChatCompletion`-shaped object with `.choices[0].message.content`."""
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content=content))]
    return completion


# ---------------------------------------------------------------------------
# check_groq_availability
# ---------------------------------------------------------------------------


def test_check_groq_availability_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    # Avoid picking up a real .env from the actual repo root or CWD during this test.
    with patch("dotenv.load_dotenv", return_value=False):
        available, detail = vlm_judges.check_groq_availability(api_key=None)
    assert available is False
    assert "GROQ_API_KEY is not set" in detail


def test_check_groq_availability_success() -> None:
    with patch("groq.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_chat_completion("pong")

        available, detail = vlm_judges.check_groq_availability(api_key="fake-key")

    assert available is True
    assert detail == "OK"
    mock_groq_cls.assert_called_once_with(api_key="fake-key")
    _, call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs["model"] == vlm_judges.GROQ_JUDGE_MODEL_ID


def test_check_groq_availability_model_not_found() -> None:
    with patch("groq.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = _not_found_error()

        available, detail = vlm_judges.check_groq_availability(api_key="fake-key")

    assert available is False
    assert vlm_judges.GROQ_JUDGE_MODEL_ID in detail
    assert "unreachable" in detail


# ---------------------------------------------------------------------------
# extract_attributes_groq
# ---------------------------------------------------------------------------


def test_extract_attributes_groq_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with (
        patch("dotenv.load_dotenv", return_value=False),
        pytest.raises(vlm_judges.JudgeUnavailableError, match="GROQ_API_KEY is not set"),
    ):
        vlm_judges.extract_attributes_groq(Path("irrelevant.png"), _DIMENSIONS, api_key=None)


def test_extract_attributes_groq_model_not_found_raises_judge_unavailable(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "concept.png"
    image_path.write_bytes(_FAKE_IMAGE_BYTES)

    with patch("groq.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = _not_found_error()

        with pytest.raises(vlm_judges.JudgeUnavailableError, match="unreachable"):
            vlm_judges.extract_attributes_groq(image_path, _DIMENSIONS, api_key="fake-key")


def test_extract_attributes_groq_success_parses_json_response(tmp_path: Path) -> None:
    image_path = tmp_path / "concept.png"
    image_path.write_bytes(_FAKE_IMAGE_BYTES)
    fake_response = (
        '{"product_type": "t-shirt", "colour_family": "black", '
        '"graphical_treatment": "solid", "garment_group": "jersey basic"}'
    )

    with patch("groq.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_chat_completion(fake_response)

        result = vlm_judges.extract_attributes_groq(image_path, _DIMENSIONS, api_key="fake-key")

    # The checklist no longer asks for `garment_group`, so an extra key a judge volunteers
    # is dropped rather than returned.
    assert result == {
        "product_type": "t-shirt",
        "colour_family": "black",
        "graphical_treatment": "solid",
    }


def test_extract_attributes_groq_call_is_blind_and_uses_the_configured_model(
    tmp_path: Path,
) -> None:
    """The judge is called with the image + a generic prompt, never the ground-truth attribute
    values (the skill's blindness contract -- see `run_qc.py`'s `JudgeCaller` docstring), and
    always targets `GROQ_JUDGE_MODEL_ID`."""
    image_path = tmp_path / "concept.jpg"
    image_path.write_bytes(_FAKE_IMAGE_BYTES)
    ground_truth_value_that_must_never_appear = "Jersey Basic Black Solid"

    with patch("groq.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_chat_completion(
            '{"product_type": "x", "colour_family": "x", '
            '"graphical_treatment": "x", "garment_group": "x"}'
        )

        vlm_judges.extract_attributes_groq(image_path, _DIMENSIONS, api_key="fake-key")

    _, call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs["model"] == vlm_judges.GROQ_JUDGE_MODEL_ID
    assert call_kwargs["response_format"] == {"type": "json_object"}
    sent_content = call_kwargs["messages"][0]["content"]
    text_part = next(part["text"] for part in sent_content if part["type"] == "text")
    image_part = next(part for part in sent_content if part["type"] == "image_url")
    assert ground_truth_value_that_must_never_appear not in text_part
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")


# ---------------------------------------------------------------------------
# build_judge_prompt / _parse_judge_json -- shared helpers exercised via the Groq adapter above,
# plus direct edge-case coverage here (empty/malformed responses must not silently mis-score).
# ---------------------------------------------------------------------------


def test_parse_judge_json_missing_keys_default_to_empty_string() -> None:
    parsed = vlm_judges._parse_judge_json('{"product_type": "t-shirt"}', _DIMENSIONS)
    assert parsed["product_type"] == "t-shirt"
    assert parsed["colour_family"] == ""


def test_parse_judge_json_empty_response_raises() -> None:
    with pytest.raises(RuntimeError, match="empty"):
        vlm_judges._parse_judge_json(None, _DIMENSIONS)


def test_parse_judge_json_malformed_json_raises() -> None:
    with pytest.raises(RuntimeError, match="not valid JSON"):
        vlm_judges._parse_judge_json("not json at all", _DIMENSIONS)
