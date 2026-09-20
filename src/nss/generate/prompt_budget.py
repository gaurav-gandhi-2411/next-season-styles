"""SDXL/CLIP text-encoder 77-token prompt budget enforcement.

WHY THIS MODULE EXISTS: both of SDXL's text encoders (`prompt` via CLIP ViT-L/14, `prompt_2` via
OpenCLIP ViT-bigG) truncate at 77 tokens -- confirmed directly against
`CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")`: the first drafted
`design_briefs.json` prompts measured 175 tokens, and the pipeline's own truncation warning named
the ENTIRE "Novel accents to introduce" clause as the silently-dropped remainder -- i.e. the part
of the prompt that had been added specifically never actually reached the model. That defect was
fixed BY HAND for the 3 specific prompts written then (see `reports/tables/design_briefs.json`'s
`rendered_prompt` fields, hand-shortened and hand-reordered). This module makes the fix SYSTEMATIC
so it survives any future prompt this project builds (e.g. new styles a retraining run
selects) without needing another by-hand pass.

Two guarantees this module provides that did not exist anywhere in the codebase before this module
(grep confirms `CLIPTokenizer` was previously referenced only in a module DOCSTRING, in
`nss.generate.final_concepts_v2`, never actually imported or called -- the earlier "fix" was a
one-time manual check, not code):

1. `count_clip_tokens` -- the REAL token count (never an estimate like `len(text.split())`), via
   the exact tokenizer that governs truncation at generation time.
2. `fit_prompt_to_token_budget` -- if the assembled prompt would exceed budget, SHORTEN it
   programmatically by dropping lower-priority clauses from the end, one at a time, until it fits
   -- never a silent, unpredictable truncation that could drop content mid-sentence or mid-clause.
   The mandatory clause (a style's defining attributes, plus -- for styles requiring it -- a hard
   framing requirement) is NEVER dropped; see that function's docstring for the full priority
   order.

`skills/style-brief/generate_brief.py` (the generic, dataset-agnostic skill) stays generation-
backend-agnostic and does no token counting itself -- see that module's docstring. This project-
specific module is exactly the "calling code" that pairs the skill's attribute-first prompt
structure with SDXL's concrete 77-token limit. See
`nss.generate.final_concepts.build_generation_spec` for the actual call site.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

CLIP_TOKENIZER_ID = "openai/clip-vit-large-patch14"
SDXL_TOKEN_BUDGET = 77


@lru_cache(maxsize=1)
def _tokenizer() -> Any:
    """Load (once, cached) the real SDXL/CLIP tokenizer -- the SAME tokenizer family that governs
    truncation at generation time (`nss.generate.backends._generate_local_sdxl`'s pipeline call).

    Returns:
        A `transformers.CLIPTokenizer`.
    """
    from transformers import CLIPTokenizer

    return CLIPTokenizer.from_pretrained(CLIP_TOKENIZER_ID)


def count_clip_tokens(text: str) -> int:
    """The real SDXL/CLIP token count for `text`, BOS/EOS included.

    This is the same count SDXL's 77-token position-embedding limit is checked against -- never an
    estimate (e.g. whitespace-splitting or `len(text) // 4`), since the whole point of this module
    is to replace exactly that kind of guess with a measured number (the original defect was
    found by measuring, not estimating -- see module docstring).

    Args:
        text: The prompt (or a single clause) to count.

    Returns:
        Token count including the tokenizer's `<|startoftext|>`/`<|endoftext|>` special tokens
        (added by default, matching what the SDXL pipeline actually sends to the text encoder).
    """
    return len(_tokenizer()(text)["input_ids"])


def fit_prompt_to_token_budget(
    mandatory_clause: str,
    novelty_clauses: Sequence[str] = (),
    descriptive_clauses: Sequence[str] = (),
    budget: int = SDXL_TOKEN_BUDGET,
    joiner: str = " ",
) -> tuple[str, int, list[str]]:
    """Assemble `mandatory_clause` + `novelty_clauses` + `descriptive_clauses` into one prompt,
    dropping lower-priority clauses from the END until the REAL CLIP token count fits `budget`.

    Priority order, highest to lowest (lowest dropped first):
    1. `mandatory_clause` -- NEVER dropped. This is the style's defining-attribute clause (see
       `skills/style-brief/generate_brief.py`'s `attribute_clause`), plus, for styles that require
       it, a hard framing requirement (e.g. `nss.generate.final_concepts.UNDERWEAR_PROMPT_SUFFIX`)
       folded directly in by the caller BEFORE calling this function -- both must survive budget
       fitting unconditionally, so both belong in this one never-dropped clause.
    2. `descriptive_clauses` -- silhouette/fabric/colour-accent/surface-treatment detail plus the
       closing product-photography boilerplate. Dropped only after every `novelty_clauses` entry
       is already gone and the prompt is STILL over budget.
    3. `novelty_clauses` -- LOWEST priority, dropped FIRST. This is deliberate: the attribute-
       fidelity gate (`skills/concept-qc/run_qc.py`) scores the 4 defining attributes, not the
       novelty axes, so losing a novelty clause to a budget trim is a much smaller correctness risk
       than losing an attribute or the framing requirement -- exactly the opposite of what the
       original defect did (it silently dropped the ENTIRE novelty clause while leaving less
       important content in place, purely because novelty happened to sit at the string's tail).

    Args:
        mandatory_clause: The never-dropped clause. Must fit `budget` on its own (see Raises).
        novelty_clauses: Droppable, dropped from the end first (index -1, -2, ...).
        descriptive_clauses: Droppable, dropped only after every `novelty_clauses` entry is gone,
            also from the end first.
        budget: Maximum real CLIP token count for the assembled prompt (default
            `SDXL_TOKEN_BUDGET`, 77).
        joiner: String used to join surviving clauses.

    Returns:
        `(prompt, token_count, dropped_clauses)` -- `token_count` is the REAL measured count of the
        returned `prompt` (guaranteed `<= budget`); `dropped_clauses` lists every clause text that
        was removed to fit budget, in the order removed (empty if nothing needed dropping).

    Raises:
        ValueError: if `mandatory_clause` ALONE already exceeds `budget` -- there is nothing left
            this function can drop; the caller must shorten `mandatory_clause` itself. Never
            silently truncated here (that is exactly the defect this module exists to prevent).
    """
    mandatory_tokens = count_clip_tokens(mandatory_clause)
    if mandatory_tokens > budget:
        raise ValueError(
            f"mandatory_clause alone is {mandatory_tokens} tokens, over the {budget}-token "
            f"budget -- cannot fit by dropping novelty/descriptive clauses: {mandatory_clause!r}"
        )

    novelty = list(novelty_clauses)
    descriptive = list(descriptive_clauses)
    dropped: list[str] = []

    def _assemble() -> str:
        return joiner.join([mandatory_clause, *novelty, *descriptive])

    prompt = _assemble()
    token_count = count_clip_tokens(prompt)

    # Lowest priority first: novelty.
    while token_count > budget and novelty:
        dropped.append(novelty.pop())
        prompt = _assemble()
        token_count = count_clip_tokens(prompt)

    # Still over budget with no novelty left: descriptive detail next, same end-first rule.
    while token_count > budget and descriptive:
        dropped.append(descriptive.pop())
        prompt = _assemble()
        token_count = count_clip_tokens(prompt)

    return prompt, token_count, dropped
