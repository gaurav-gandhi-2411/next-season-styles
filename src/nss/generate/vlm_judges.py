"""Concrete Gemini + Groq blind-VLM judge adapters for the `concept-qc` skill (task C7).

DATASET/PROVIDER-SPECIFIC ADAPTER CODE LIVES HERE, NOT IN THE SKILL: `skills/concept-qc/run_qc.py`
deliberately knows nothing about Gemini, Groq, `google-genai`, or the `groq` SDK -- it only defines
the generic `JudgeCaller` protocol (an image + attribute-dimension names in, a raw per-dimension
extraction dict out) and the verdict/scoring math that consumes already-scored judges. This module
is exactly the "calling code" that skill's docstring describes -- it implements two concrete
`JudgeCaller`s conforming to that protocol.

MODEL-ID DEVIATION FROM THE TASK BRIEF (documented, not silently substituted): the task brief named
`gemini-2.5-flash` as Judge A. Tested directly against the live API on 2026-09-19 (this project's
`GEMINI_API_KEY`) and it 404s: `'This model models/gemini-2.5-flash is no longer available to new
users. Please update your code to use models/gemini-3.6-flash...'` -- even though it still appears
in `client.models.list()`. `GEMINI_JUDGE_MODEL_ID` below uses the model the API itself names as the
replacement, confirmed working end-to-end (text + vision, JSON response mode) against this
project's key before being wired in here.

GROQ VISION JUDGE MODEL-ID CORRECTION (task E6, documented, not silently substituted): the task
brief originally named Groq's "Llama 4 Scout vision" as Judge B
(`meta-llama/llama-4-scout-17b-16e-instruct`), which 404s for this account, as do three other
guessed vision model names -- see task C7's report for that history. Rather than guessing a 5th
name, task E6 queried the LIVE `GET https://api.groq.com/openai/v1/models` endpoint directly
(2026-09-19, this project's `GROQ_API_KEY`) and inspected every returned model's
`input_modalities`. Exactly one model in the account's live list declares `"image"` as an input
modality: `qwen/qwen3.8-27b` (Alibaba Cloud, `input_modalities: ["text", "image"]`,
`output_modalities: ["text"]`, supports `json_mode`). Every other returned model is text-only
(`openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `openai/gpt-oss-safeguard-20b`, `groq/compound`,
`groq/compound-mini`, `allam-2-7b`, two `meta-llama/llama-prompt-guard-2-*` text classifiers) or
audio-only (`whisper-large-v3`, `whisper-large-v3-turbo`; two `canopylabs/orpheus-*` TTS models).
`GROQ_JUDGE_MODEL_ID` below is `qwen/qwen3.8-27b`, confirmed working end-to-end (vision input, JSON
response mode, a real H&M catalogue image) against this project's key before being wired in here.
`check_groq_availability` still performs a live reachability check at runtime (never assumed from
this docstring) and `extract_attributes_groq` still raises `JudgeUnavailableError` on any future
404/decommission, so the QC pipeline continues to treat a Groq outage as a documented, reported
"blocked" condition -- never silently skipped, never a crash.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SKILL_MODULE_PATH = _REPO_ROOT / "skills" / "concept-qc" / "run_qc.py"


def _load_skill_module(module_path: Path = _SKILL_MODULE_PATH) -> ModuleType:
    """Load `skills/concept-qc/run_qc.py` via an explicit file-path import.

    Mirrors `nss.generate.build_design_briefs._load_skill_module`'s identical pattern for the
    `style-brief` skill -- the skill module lives outside the installed `nss` package on purpose
    (portability), so this project's adapter code loads it explicitly rather than via a normal
    package import.

    Args:
        module_path: Path to `run_qc.py`.

    Returns:
        The loaded module, exposing `JudgeUnavailableError`, `ATTRIBUTE_DIMENSIONS`-shaped
        dimension handling, etc.

    Raises:
        FileNotFoundError: if `module_path` does not exist.
        ImportError: if the module spec/loader cannot be constructed.
    """
    if not module_path.exists():
        raise FileNotFoundError(f"concept-qc skill module not found at {module_path}")
    spec = importlib.util.spec_from_file_location("concept_qc_run_qc", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build an import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    # Registered in sys.modules before exec so the module's own internal references (and any code
    # that later does `import concept_qc_run_qc`) resolve consistently -- same defensive pattern
    # `importlib`'s own docs recommend for exec_module-based loading.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SKILL = _load_skill_module()
"""The loaded `skills/concept-qc/run_qc.py` module -- the single shared instance. Other adapter
code (`nss.generate.concept_qc_pipeline`) imports THIS reference rather than loading its own
separate copy, so e.g. `isinstance(exc, SKILL.JudgeUnavailableError)` checks stay valid across
modules (two independent `exec_module` loads of the same file would otherwise produce two
distinct, `isinstance`-incompatible class objects for what looks like "the same" exception type)."""

JudgeUnavailableError = SKILL.JudgeUnavailableError

# Task H2: only VISUALLY OBSERVABLE attributes are scored. `garment_group` (e.g. "Jersey Basic",
# "Under-, Nightwear") is an internal H&M merchandising taxonomy term with no visual referent -- a
# judge shown a plain black T-shirt correctly answers "top" and scores 0.0 against "Jersey Basic"
# (measured, `reports/tables/h2_judge_rescore.csv`), so no image can ever score on it and it
# capped every candidate's fidelity by up to 25%. Dropped rather than mapped to a visual
# descriptor: a mapping needs one curated descriptor per taxonomy value (unbounded across
# catalogues) and would itself be an unvalidated construct; see skills/concept-qc/SKILL.md.
ATTRIBUTE_DIMENSIONS: tuple[str, ...] = (
    "product_type",
    "colour_family",
    "graphical_treatment",
)
# The pre-H2 checklist, kept ONLY so H2's paired before/after comparison can be recomputed from one
# set of stored per-attribute scores. Never used for gating.
LEGACY_ATTRIBUTE_DIMENSIONS: tuple[str, ...] = (*ATTRIBUTE_DIMENSIONS, "garment_group")

GEMINI_JUDGE_MODEL_ID = "models/gemini-3.6-flash"  # see module docstring MODEL-ID DEVIATION note
GROQ_JUDGE_MODEL_ID = "qwen/qwen3.8-27b"  # see module docstring GROQ VISION JUDGE MODEL-ID note


def build_judge_prompt(dimensions: Sequence[str] = ATTRIBUTE_DIMENSIONS) -> str:
    """Build the blind extraction prompt sent to every judge (see run_qc.py's blindness contract).

    Args:
        dimensions: Which attribute dimension NAMES to ask about (never the target values).

    Returns:
        A prompt instructing the judge to report its own perceived value per dimension as strict
        JSON, with no mention of any target style, product, or design brief.
    """
    dims_text = "\n".join(f"- {d.replace('_', ' ')}" for d in dimensions)
    schema_hint = json.dumps({d: "..." for d in dimensions})
    return (
        "You are a blind fashion-QC image analyst. You are shown ONE product photograph. You are "
        "NOT told what product, brand, colour, or design brief was used to create this image -- "
        "judge only what you directly observe in the image itself. For each of the following "
        "attribute dimensions, report your own single best free-text description of what the "
        f"image shows:\n{dims_text}\n\n"
        "Return STRICT JSON with exactly this shape (one short free-text value per key, no other "
        f"keys, no commentary, no markdown code fencing): {schema_hint}"
    )


def _parse_judge_json(text: str | None, dimensions: Sequence[str]) -> dict[str, str]:
    """Parse a judge's raw JSON response text into a `{dimension: value}` dict.

    Args:
        text: The judge's raw response text (expected to be a JSON object per `build_judge_prompt`).
        dimensions: The requested dimensions -- any missing from the parsed JSON are filled with
            `""` (an empty extraction is a real, low-scoring answer, not a crash).

    Returns:
        `{dimension: str}` for every `dimensions` entry.

    Raises:
        RuntimeError: if `text` is empty/`None`, or does not parse as a JSON object.
    """
    if not text:
        raise RuntimeError("judge response was empty -- cannot parse an attribute extraction")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"judge response was not valid JSON: {text!r}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"judge response JSON was not an object: {text!r}")
    return {dim: str(parsed.get(dim, "")) for dim in dimensions}


def extract_attributes_gemini(
    image_path: Path,
    attribute_dimensions: Sequence[str] = ATTRIBUTE_DIMENSIONS,
    api_key: str | None = None,
) -> dict[str, str]:
    """Judge A: blind attribute extraction via Gemini (`GEMINI_JUDGE_MODEL_ID`), a `JudgeCaller`.

    Args:
        image_path: Path to the concept (or, for calibration, real catalogue) image.
        attribute_dimensions: Which dimensions to ask about.
        api_key: Overrides `GEMINI_API_KEY` from the environment/`.env` (mainly for tests).

    Returns:
        `{dimension: str}` per `attribute_dimensions`.

    Raises:
        JudgeUnavailableError: `GEMINI_API_KEY` is not set.
        RuntimeError: any other Gemini failure (network, malformed response) -- NOT swallowed,
            same "only the documented recoverable case is caught" convention as
            `nss.generate.backends._require_gemini_api_key` / `generate_concept`.
    """
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types as genai_types
    from PIL import Image

    load_dotenv()
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise JudgeUnavailableError(
            "GEMINI_API_KEY is not set -- create a .env file at the repo root containing "
            "GEMINI_API_KEY=<your key>."
        )
    client = genai.Client(api_key=key)
    image = Image.open(image_path).convert("RGB")
    prompt = build_judge_prompt(attribute_dimensions)
    response = client.models.generate_content(
        model=GEMINI_JUDGE_MODEL_ID,
        contents=[prompt, image],
        config=genai_types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return _parse_judge_json(response.text, attribute_dimensions)


def check_groq_availability(api_key: str | None = None) -> tuple[bool, str]:
    """One-shot live check of whether `GROQ_JUDGE_MODEL_ID` is reachable for this `GROQ_API_KEY`.

    Performs a minimal, cheap (`max_tokens=5`, no image) completion call -- run ONCE per QC
    pipeline invocation (never per-image/per-attempt) to avoid hammering a confirmed-unreachable
    model with repeated identical failures.

    Args:
        api_key: Overrides `GROQ_API_KEY` from the environment/`.env` (mainly for tests).

    Returns:
        `(available, detail)` -- `detail` explains why, either way (never a bare `True`/`False`
        with no evidence trail).
    """
    from dotenv import load_dotenv

    load_dotenv()
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        return False, "GROQ_API_KEY is not set in .env."

    from groq import BadRequestError, Groq, NotFoundError

    client = Groq(api_key=key)
    try:
        client.chat.completions.create(
            model=GROQ_JUDGE_MODEL_ID,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
    except (NotFoundError, BadRequestError) as exc:
        return False, (
            f"Groq vision judge model {GROQ_JUDGE_MODEL_ID!r} is unreachable for this "
            f"GROQ_API_KEY (model not found / decommissioned): {exc}"
        )
    return True, "OK"


def extract_attributes_groq(
    image_path: Path,
    attribute_dimensions: Sequence[str] = ATTRIBUTE_DIMENSIONS,
    api_key: str | None = None,
) -> dict[str, str]:
    """Judge B: blind attribute extraction via Groq (`GROQ_JUDGE_MODEL_ID`), a `JudgeCaller`.

    Args:
        image_path: Path to the concept (or, for calibration, real catalogue) image.
        attribute_dimensions: Which dimensions to ask about.
        api_key: Overrides `GROQ_API_KEY` from the environment/`.env` (mainly for tests).

    Returns:
        `{dimension: str}` per `attribute_dimensions`.

    Raises:
        JudgeUnavailableError: `GROQ_API_KEY` is not set, or `GROQ_JUDGE_MODEL_ID` is unreachable
            for this account (model not found / decommissioned) -- see module docstring GROQ
            VISION JUDGE AVAILABILITY note.
        RuntimeError: any other Groq failure -- NOT swallowed.
    """
    from dotenv import load_dotenv
    from groq import BadRequestError, Groq, NotFoundError

    load_dotenv()
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise JudgeUnavailableError(
            "GROQ_API_KEY is not set -- create a .env file at the repo root containing "
            "GROQ_API_KEY=<your key>."
        )
    client = Groq(api_key=key)
    image_bytes = image_path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    prompt = build_judge_prompt(attribute_dimensions)
    try:
        response = client.chat.completions.create(
            model=GROQ_JUDGE_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{encoded}"},
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
        )
    except (NotFoundError, BadRequestError) as exc:
        raise JudgeUnavailableError(
            f"Groq vision judge model {GROQ_JUDGE_MODEL_ID!r} is unreachable for this "
            f"GROQ_API_KEY (model not found / decommissioned): {exc}"
        ) from exc
    return _parse_judge_json(response.choices[0].message.content, attribute_dimensions)
