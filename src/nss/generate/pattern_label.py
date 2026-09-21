"""The "All over pattern" label is satisfied by ANY named visible pattern (pre-registered rule).

H&M's `graphical_appearance_name` bucket "All over pattern" is a merchandising bucket, not a
describable pattern: a judge looking at an orange geometric print says "Polka dot", "Floral" or
"Checkered" and never the bucket name, so the literal string match scores 0 for every seed, a
zero-variance gate. This is the same class as "Jersey Basic" and "Other structure" (see
`fidelity.py`), handled the same way: on principle, from an EXPLICIT word list, never per image.

RULE (`reports/v3/PREREGISTRATION.md`, 1e, committed before any control was run):

- Active only when the ground-truth label is exactly "All over pattern" and the judge's answer is
  short (at most `MAX_ANSWER_WORDS` words). A long free-text caption (Florence-2) is left to the
  legacy scorer, because captions mention "plain background" and would fail spuriously.
- The label is SATISFIED (score 1.0) iff the answer contains a word from `PATTERN_WORDS` and no
  word from `PLAIN_WORDS`. Otherwise it FAILS (0.0): a plain/solid answer, a nonsense answer and an
  empty answer all fail. An answer naming both a pattern and "solid" is ambiguous and fails.
- Every other label is untouched (`mapped_graphical_score` returns None; the caller keeps the
  legacy score), so no other style's verdict can change.

`melange` is in the word list because the rule is "any named pattern"; it is also a label SmolVLM
gives to plain heathered fabric, which is exactly what the mandatory negative control (real solid
bikini tops must FAIL) is there to catch. If a solid bikini passes, the mapping is reverted.
"""

from __future__ import annotations

import re

LABEL = "All over pattern"
MAX_ANSWER_WORDS = 6

PATTERN_WORDS: frozenset[str] = frozenset(
    {
        "pattern",
        "patterned",
        "floral",
        "flower",
        "flowers",
        "stripe",
        "stripes",
        "striped",
        "check",
        "checks",
        "checked",
        "checkered",
        "chequered",
        "plaid",
        "gingham",
        "tartan",
        "dot",
        "dots",
        "dotted",
        "polka",
        "spot",
        "spots",
        "spotted",
        "geometric",
        "animal",
        "leopard",
        "zebra",
        "tiger",
        "snake",
        "paisley",
        "camo",
        "camouflage",
        "tie-dye",
        "abstract",
        "graphic",
        "print",
        "printed",
        "chevron",
        "zigzag",
        "houndstooth",
        "melange",
        "marl",
        "marled",
        "mottled",
        "speckled",
        "tropical",
        "leaf",
        "leaves",
        "botanical",
        "hearts",
        "stars",
    }
)
PLAIN_WORDS: frozenset[str] = frozenset(
    {"solid", "plain", "none", "unpatterned", "monochrome", "uniform", "no"}
)
_WORD = re.compile(r"[a-z][a-z-]*")


def words(text: str) -> list[str]:
    """Lower-cased alphabetic words of an answer (punctuation dropped, hyphens kept)."""
    return _WORD.findall(text.lower())


def names_visible_pattern(answer: str) -> bool:
    """True iff the answer names a visible pattern and does not also call the garment plain."""
    found = set(words(answer))
    return bool(found & PATTERN_WORDS) and not (found & PLAIN_WORDS)


def mapped_graphical_score(answer: str, truth_label: str) -> float | None:
    """1.0 / 0.0 under the rule, or None when it does not apply (caller keeps the legacy score)."""
    if truth_label.strip() != LABEL:
        return None
    if len(words(answer)) > MAX_ANSWER_WORDS:
        return None
    return 1.0 if names_visible_pattern(answer) else 0.0


def apply_mapping(
    scores: dict[str, float], extraction: dict[str, str], truth: dict[str, str]
) -> dict[str, float]:
    """A copy of `scores` with the graphical dimension replaced where the rule applies."""
    out = dict(scores)
    graphical = "graphical_treatment"
    if graphical in out:
        mapped = mapped_graphical_score(extraction.get(graphical, ""), truth.get(graphical, ""))
        if mapped is not None:
            out[graphical] = mapped
    return out
