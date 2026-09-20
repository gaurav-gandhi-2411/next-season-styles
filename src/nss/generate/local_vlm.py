"""Local vision-language judge: no API quota, runs on the 8 GB GPU / CPU (task N4).

WHY: the Groq and Gemini judges are free-tier and quota-limited (Gemini 20 requests/day; Groq daily
token budget), which made every earlier Gate-2 run partial, delayed or single-judge. A local model
has no quota, so every concept gets a full reading every time, and framing/pattern screening of a
25-image reference base no longer burns API calls.

Two model backends are supported behind one `ask(image, question) -> str` interface, chosen by
calibration (`nss.generate.local_judge_calibration`), not by preference:

- ``moondream2`` (`vikhyatk/moondream2`, pinned revision, `trust_remote_code`): ~1.9B parameters,
  purpose-built for short visual question answering.
- ``smolvlm`` (`HuggingFaceTB/SmolVLM-500M-Instruct`): native `transformers`, no remote code, ~0.5B.

`extract_attributes_local` is a `JudgeCaller` (same signature and blind contract as the Groq and
Gemini extractors: only the dimension NAMES are asked about, never the target values), so it plugs
into `judge_repeat.CALLERS` and the shared scoring code unchanged. `ask_yes_no` backs Gate 3
(design-change verification) and the garment-integrity check.

Never used uncalibrated: a local judge that fails calibration is reported and dropped, not shipped.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image

MOONDREAM_REPO = "vikhyatk/moondream2"
# Pinned: the model repo ships executable modelling code (`trust_remote_code`), so track an exact
# revision rather than whatever HEAD is today.
MOONDREAM_REVISION = "2025-06-21"
SMOLVLM_REPO = "HuggingFaceTB/SmolVLM-500M-Instruct"
FLORENCE_REPO = "florence-community/Florence-2-base"
BACKEND = os.environ.get("NSS_LOCAL_VLM", "moondream2")

_STATE: dict[str, Any] = {}

# One short, free-text question per attribute dimension (blind: no target value appears).
DIMENSION_QUESTIONS: dict[str, str] = {
    "product_type": "What type of garment is shown? Answer with one or two words.",
    "colour_family": "What is the main colour of the garment? Answer with one word.",
    "graphical_treatment": (
        "What is the fabric pattern of the garment (for example solid, striped, checked, floral, "
        "melange, lace)? Answer with one word."
    ),
}


def _load_moondream() -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, PreTrainedModel

    # Moondream's remote code predates transformers 5, which expects this attribute at load time.
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}
    model = AutoModelForCausalLM.from_pretrained(
        MOONDREAM_REPO,
        revision=MOONDREAM_REVISION,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map={"": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    return model, None


def _load_smolvlm() -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(SMOLVLM_REPO)
    model = AutoModelForImageTextToText.from_pretrained(
        SMOLVLM_REPO,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map={"": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    return model, processor


def _load_florence() -> tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor, Florence2ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(FLORENCE_REPO)
    model = Florence2ForConditionalGeneration.from_pretrained(
        FLORENCE_REPO,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map={"": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    return model, processor


def load(backend: str | None = None) -> None:
    """Load (once) the chosen backend's model; `unload` frees the GPU for image generation."""
    backend = backend or _STATE.get("backend") or BACKEND
    if _STATE.get("backend") == backend:
        return
    unload()
    if backend == "moondream2":
        model, processor = _load_moondream()
    elif backend == "smolvlm":
        model, processor = _load_smolvlm()
    elif backend == "florence2":
        model, processor = _load_florence()
    else:
        raise ValueError(f"unknown local VLM backend {backend!r}")
    _STATE.update({"backend": backend, "model": model, "processor": processor})


def unload() -> None:
    """Drop the loaded model and free GPU memory (SDXL generation needs the whole card)."""
    import gc

    _STATE.clear()
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover - torch is a hard dependency of this package
        pass


def ask(image_path: Path | str, question: str, max_new_tokens: int = 24) -> str:
    """Answer one question about one image; deterministic (greedy) decoding."""
    load()
    image = Image.open(image_path).convert("RGB")
    backend = _STATE["backend"]
    model, processor = _STATE["model"], _STATE["processor"]
    if backend == "moondream2":
        return str(model.query(image, question)["answer"]).strip()
    import torch

    if backend == "florence2":
        # Florence-2 is a captioner, not a question-answerer: the question is ignored and its
        # detailed caption is returned; attribute scoring then works by containment.
        task = "<MORE_DETAILED_CAPTION>"
        inputs = processor(text=task, images=image, return_tensors="pt").to(
            model.device, model.dtype
        )
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=96, do_sample=False, num_beams=3)
        return str(processor.batch_decode(out, skip_special_tokens=True)[0]).strip()
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = processor.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return str(text[0]).strip()


def ask_yes_no(image_path: Path | str, question: str) -> tuple[bool, str]:
    """Binary question -> `(answer_is_yes, raw_text)`. Anything not starting with 'yes' is 'no'.

    Fail-closed: an empty or unparseable answer counts as 'no', never as a pass.
    """
    if (_STATE.get("backend") or BACKEND) == "florence2":
        raise NotImplementedError("Florence-2 cannot answer yes/no questions (captioner only)")
    raw = ask(image_path, question + " Answer yes or no.", max_new_tokens=6)
    return raw.strip().lower().startswith("yes"), raw


def extract_attributes_local(
    image_path: Path,
    attribute_dimensions: Sequence[str] = tuple(DIMENSION_QUESTIONS),
    api_key: str | None = None,
) -> dict[str, str]:
    """`JudgeCaller`: blind attribute extraction with the local model (`api_key` is unused)."""
    del api_key
    return {d: ask(image_path, DIMENSION_QUESTIONS[d]) for d in attribute_dimensions}
