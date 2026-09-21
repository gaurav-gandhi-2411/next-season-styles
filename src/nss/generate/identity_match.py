"""K4: product type and colour are HARD identity constraints for Gate 2.

For a style keyed on colour and product type, a wrong colour or a different garment is a different
product, not a one-third penalty in an average. Gate 2 keeps its averaged fidelity and per-judge
threshold EXACTLY as calibrated (scorer and threshold untouched) and adds an AND: the gating judge's
product-type reading and colour reading must each match the style's value, allowing the synonyms
below. Because it is an AND on top of the old rule, this can only turn a pass into a fail, never a
fail into a pass; the synonym lists can therefore only PREVENT new failures, never loosen Gate 2.
Pattern is not constrained (judges misread melange and "All over pattern"; it stays in the average).

Pre-registered in `reports/v3/PREREGISTRATION.md` (section K4) and committed before any scoring;
each entry is justified there. An unknown style value falls back to whole-word token containment in
either direction. A reading that names a word from a DIFFERENT family ("sweater dress",
"red and white") is a conflict and fails: the constraint is deliberately strict.
"""

from __future__ import annotations

import re

# H&M `product_type_name` -> words a judge may use for that same garment (lower-case, whole words).
PRODUCT_SYNONYMS: dict[str, frozenset[str]] = {
    "sweater": frozenset(
        {
            "sweater",
            "jumper",
            "pullover",
            "turtleneck",
            "polo neck",
            "roll neck",
            "knit",
            "knitwear",
        }
    ),
    "dress": frozenset({"dress", "sundress", "frock"}),
    "top": frozenset({"top", "blouse", "shirt", "t-shirt", "tee", "tank top"}),
    "bikini top": frozenset({"bikini top", "bikini", "swim top", "swimsuit top", "bikini bra"}),
    "t-shirt": frozenset({"t-shirt", "tee", "tshirt", "t shirt"}),
    "underwear bottom": frozenset(
        {"underwear bottom", "underwear", "briefs", "panties", "knickers", "underpants"}
    ),
    "trousers": frozenset({"trousers", "pants", "slacks", "chinos"}),
    "blazer": frozenset({"blazer", "suit jacket"}),
    "cardigan": frozenset({"cardigan", "cardi"}),
}

# H&M `perceived_colour_master_name` -> colour words that belong to that master colour.
COLOUR_SYNONYMS: dict[str, frozenset[str]] = {
    "beige": frozenset({"beige", "tan", "sand", "oatmeal"}),
    "white": frozenset({"white", "off-white", "ivory", "cream"}),
    "black": frozenset({"black"}),
    "red": frozenset({"red", "crimson", "scarlet", "burgundy", "maroon", "wine"}),
    "orange": frozenset({"orange", "tangerine", "amber"}),
    "blue": frozenset({"blue", "navy", "cobalt"}),
    "green": frozenset({"green", "emerald", "olive", "mint"}),
    "grey": frozenset({"grey", "gray", "silver", "charcoal"}),
    "pink": frozenset({"pink", "rose", "blush", "fuchsia"}),
    "brown": frozenset({"brown", "chocolate", "mocha", "chestnut"}),
    "yellow": frozenset({"yellow", "mustard"}),
    "purple": frozenset({"purple", "lavender", "violet"}),
    "turquoise": frozenset({"turquoise", "teal"}),
}


def normalise(text: str) -> str:
    """Lower-case, punctuation to spaces (hyphens kept inside words), single-spaced."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\- ]+", " ", text.lower())).strip()


def _words_in(text: str, vocabulary: dict[str, frozenset[str]]) -> set[str]:
    """Every vocabulary word found as a whole word in `text`, longest overlapping phrase wins."""
    found = {
        word
        for words in vocabulary.values()
        for word in words
        if re.search(rf"(?<![a-z0-9\-]){re.escape(word)}(?![a-z0-9\-])", text)
    }
    return {w for w in found if not any(w != w2 and w in w2 for w2 in found)}


def _match(reading: str, truth: str, vocabulary: dict[str, frozenset[str]]) -> bool:
    text, want = normalise(reading), normalise(truth)
    if not text or not want:
        return False
    if want in vocabulary:
        # every known word in the reading must belong to the style's own family (a word shared
        # with another family, such as "tee" for Top and T-shirt, is fine), and there must be one
        found = _words_in(text, vocabulary)
        return bool(found) and found <= vocabulary[want]
    a, b = set(text.split()), set(want.split())
    return bool(a) and bool(b) and (a <= b or b <= a)


def product_type_ok(reading: str, truth: str) -> bool:
    """Does the judge's product-type reading name the style's garment (synonyms allowed)?"""
    return _match(reading, truth, PRODUCT_SYNONYMS)


def colour_ok(reading: str, truth: str) -> bool:
    """Does the judge's colour reading name the style's master colour (and no other colour)?"""
    return _match(reading, truth, COLOUR_SYNONYMS)
