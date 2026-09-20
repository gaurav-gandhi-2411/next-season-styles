"""Gate 3 (design-change verification) and the garment-integrity check.

WHY: Gates 1/1b test "not a copy" and Gate 2 tests "still the style"; nothing tested "implements
the brief". The mandatory human check was doing that by hand with no automated counterpart, and
Gate 2's attribute-only questions let visibly malformed garments (a cut-out defect, sheer mesh, an
unrecognisable folded object) pass every gate.

GATE 3: for each briefed change the judge is asked one binary question ("Does this garment have a
high funnel neck collar?"). Pass = a strict majority of the briefed changes are present. A judge
that cannot answer (quota, parse failure) yields `None` for that judge -- never a pass.

INTEGRITY: one binary question per image ("Is this ONE coherent, correctly formed garment ...?").
It is validated on known-bad AND known-good images before it is trusted
(`nss.generate.integrity_validation`); if it does not fail the known-malformed images the check does
not work and is reported as such.

All judges see the SAME questions. API judges (Groq, Gemini) get every question for an image in ONE
call (one JSON object of yes/no answers) to spend the free-tier quota once per image; the local
judge is asked one question at a time.
"""

from __future__ import annotations

import base64
import io
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path

from PIL import Image

from nss.generate import local_vlm, vlm_judges

AskFn = Callable[[Path, str], tuple[bool, str]]

CHANGE_QUESTION = "Does this garment have {change}?"
INTEGRITY_QUESTION = (
    "Is this ONE coherent, correctly formed garment, with no impossible geometry, no floating or "
    "detached parts, no missing sections, and clearly recognisable as a single wearable item?"
)


def change_questions(changes: Sequence[str]) -> dict[str, str]:
    """`{c0: question, c1: ...}` for the briefed changes, in order."""
    return {f"c{i}": CHANGE_QUESTION.format(change=c) for i, c in enumerate(changes)}


def majority_present(answers: Sequence[bool]) -> bool:
    """Strict majority of briefed changes present (2 of 2, 2 of 3, ...); empty -> False."""
    return len(answers) > 0 and sum(answers) * 2 > len(answers)


def gate3_local(image: Path, changes: Sequence[str]) -> dict[str, object]:
    """Gate 3 with the local judge: per-change yes/no, raw answers, and the majority verdict."""
    answers, raws = [], []
    for question in change_questions(changes).values():
        yes, raw = local_vlm.ask_yes_no(image, question)
        answers.append(yes)
        raws.append(raw)
    return {
        "changes": list(changes),
        "answers": answers,
        "raw": raws,
        "n_present": sum(answers),
        "pass": majority_present(answers),
    }


def integrity_local(image: Path) -> tuple[bool, str]:
    """Garment-integrity check with the local judge -> `(coherent, raw_text)`."""
    return local_vlm.ask_yes_no(image, INTEGRITY_QUESTION)


INTEGRITY_FLOOR_PERCENTILE = 10


def integrity_floor(
    concept_dino: object, reference_dino: Sequence[object], exclude_index: int | None = None
) -> dict[str, float | bool]:
    """Reference-based plausibility floor: the mirror image of Gate 1b's ceiling.

    A generic yes/no "is this a coherent garment?" question to a small VLM answers yes to
    everything (validated: it caught 0 of the 3 known-malformed candidates), so the automatic
    integrity check adds an embedding test that does work: the concept's closest real reference
    (DINOv2, the structural space) must be at least as close as the 10th percentile of the REAL
    articles' own nearest-sibling similarity. A garment that resembles no real garment of its
    style (cut-out defect, folded object, fabric swatch) falls below it. `exclude_index` drops a
    real image from its own reference set. Being in-distribution is necessary, not sufficient:
    the human check remains.
    """
    import numpy as np

    from nss.generate.clip_scoring import _cosine

    refs = [r for i, r in enumerate(reference_dino) if i != exclude_index]
    nearest_sibling = [
        max(_cosine(a, b) for j, b in enumerate(refs) if j != i) for i, a in enumerate(refs)
    ]
    floor = float(np.percentile(nearest_sibling, INTEGRITY_FLOOR_PERCENTILE))
    best = max(_cosine(concept_dino, r) for r in refs)
    return {"max_sim": float(best), "floor": floor, "pass": bool(best >= floor)}


API_IMAGE_MAX_SIDE = 768


def _downscaled(image: Path) -> Image.Image:
    """The image shrunk to `API_IMAGE_MAX_SIDE` px: a 1024 px PNG as base64 blew Groq's
    per-request token limit ("Request too large"); design-change questions do not need 1024 px."""
    im = Image.open(image).convert("RGB")
    im.thumbnail((API_IMAGE_MAX_SIDE, API_IMAGE_MAX_SIDE))
    return im


def _batched_prompt(questions: dict[str, str]) -> str:
    listing = "\n".join(f'- "{k}": {q}' for k, q in questions.items())
    schema = json.dumps({k: "yes|no" for k in questions})
    return (
        "You are a strict fashion-QC image analyst looking at ONE product photograph. Answer each "
        "question about what is directly visible, with exactly 'yes' or 'no'. Do not assume; if a "
        f"feature is not clearly visible answer 'no'.\n{listing}\n\n"
        f"Return STRICT JSON of this shape, no commentary: {schema}"
    )


def _parse_yes_no(text: str | None, keys: Sequence[str]) -> dict[str, bool]:
    if not text:
        raise RuntimeError("judge response was empty")
    parsed = json.loads(text)
    return {k: str(parsed.get(k, "no")).strip().lower().startswith("yes") for k in keys}


def ask_batch_groq(image: Path, questions: dict[str, str]) -> dict[str, bool]:
    """All `questions` about `image` in one Groq call -> `{key: yes?}` (raises on failure)."""
    from dotenv import load_dotenv
    from groq import Groq

    load_dotenv()
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise vlm_judges.JudgeUnavailableError("GROQ_API_KEY is not set")
    buf = io.BytesIO()
    _downscaled(image).save(buf, format="JPEG", quality=88)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg"
    response = Groq(api_key=key).chat.completions.create(
        model=vlm_judges.GROQ_JUDGE_MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _batched_prompt(questions)},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                ],
            }
        ],
        response_format={"type": "json_object"},
    )
    return _parse_yes_no(response.choices[0].message.content, list(questions))


def ask_batch_gemini(image: Path, questions: dict[str, str]) -> dict[str, bool]:
    """All `questions` about `image` in one Gemini call -> `{key: yes?}` (raises on failure)."""
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types as genai_types

    load_dotenv()
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise vlm_judges.JudgeUnavailableError("GEMINI_API_KEY is not set")
    client = genai.Client(api_key=key)  # keep a reference: a temporary is closed mid-request
    response = client.models.generate_content(
        model=vlm_judges.GEMINI_JUDGE_MODEL_ID,
        contents=[_batched_prompt(questions), _downscaled(image)],
        config=genai_types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return _parse_yes_no(response.text, list(questions))


API_BATCH_CALLERS: dict[str, Callable[[Path, dict[str, str]], dict[str, bool]]] = {
    "groq": ask_batch_groq,
    "gemini": ask_batch_gemini,
}


def gate3_api(judge: str, image: Path, changes: Sequence[str]) -> dict[str, object]:
    """Gate 3 plus integrity for one API judge in a single call (`answers`, `pass`, `coherent`)."""
    questions = {**change_questions(changes), "integrity": INTEGRITY_QUESTION}
    got = API_BATCH_CALLERS[judge](image, questions)
    answers = [got[k] for k in change_questions(changes)]
    return {
        "changes": list(changes),
        "answers": answers,
        "n_present": sum(answers),
        "pass": majority_present(answers),
        "coherent": got["integrity"],
    }
