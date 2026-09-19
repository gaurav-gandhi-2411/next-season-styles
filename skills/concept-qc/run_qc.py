"""Concept-QC verdict engine -- the "concept-qc" skill (task C7).

Generic, dataset-agnostic QC gate: given (a) an already-computed margin-band result (CLIP +
DINOv2, both scored elsewhere -- see `nss.generate.clip_scoring` / `dino_scoring` /
`margin_scoring` for HOW those margins are computed, out of scope for this skill) and (b) a set of
blind multi-judge VLM attribute-fidelity extractions (each judge implemented elsewhere as a
`JudgeCaller` -- see `nss.generate.vlm_judges` for this project's concrete Gemini/Groq adapters),
combine both signals into one `QCVerdict`. This module never calls an LLM/VLM API itself and never
knows a specific dataset's style-taxonomy column names -- both are the calling code's job (mirrors
`skills/style-brief/generate_brief.py`'s split from `nss.generate.build_design_briefs`).

See `SKILL.md` for the full schema, the blindness contract, the self-scoring-contamination rule,
the retry-value-selection strategy, and two fully worked examples (real C7 run output, not
fabricated). This module's own docstrings cover implementation detail; `SKILL.md` is the skill's
actual specification.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, TypedDict

DEFAULT_ATTRIBUTE_FIDELITY_THRESHOLD = 0.75
DEFAULT_BINARIZE_THRESHOLD = 0.5
LLM_CONSENSUS_LABEL = "LLM-consensus (NOT human ground truth)"


class JudgeCaller(Protocol):
    """A blind attribute-extraction callable (see `SKILL.md`'s "Blindness contract" section).

    Implementations MUST raise `JudgeUnavailableError` (never a bare pass-through of a raw SDK
    exception) for any RECOVERABLE "this judge cannot score right now" condition (missing
    credentials, a decommissioned/unreachable model, an exhausted quota) -- see
    `nss.generate.vlm_judges` for this project's concrete Gemini/Groq implementations of that
    contract. Any other exception is treated by `run_judge` as a genuine, unexpected failure and
    is still captured (never silently discarded -- see `run_judge`'s docstring) but is NOT the
    intended way to signal "this judge is currently blocked."
    """

    def __call__(self, image_path: Path, attribute_dimensions: Sequence[str]) -> dict[str, str]:
        """Return the judge's own perceived value per dimension, blind to ground truth."""
        ...


class JudgeUnavailableError(RuntimeError):
    """A `JudgeCaller` raises this to signal a recoverable "cannot score right now" condition.

    Distinguishing this from a bare exception lets `run_judge` treat "the Gemini/Groq API key is
    missing" or "the requested model is decommissioned for this account" as an expected, reportable
    (never silently swallowed) non-fatal condition -- the same convention
    `nss.generate.final_concepts.generate_gemini_candidate` already uses for C6's Gemini appendix,
    extended here to judges.
    """


class MarginBandResult(TypedDict):
    """One image's already-computed CLIP + DINOv2 margin-band scoring result.

    Computed entirely OUTSIDE this module (`nss.generate.clip_scoring` / `dino_scoring` /
    `margin_scoring`) -- this skill only reads the already-derived booleans/floats, never
    recomputes an embedding or a margin itself.
    """

    clip_margin: float
    clip_in_band: bool
    dino_margin: float
    dino_in_band: bool


class JudgeResult(TypedDict):
    """One judge's result for one image: either a scored extraction, or a documented exclusion."""

    judge_name: str
    available: bool
    raw_extraction: dict[str, str] | None
    scores: dict[str, float] | None
    mean_score: float | None
    excluded_reason: str | None


class QCVerdict(TypedDict):
    """The combined margin-band + blind-VLM-panel QC verdict for one generated concept image."""

    style_id: str
    margin: MarginBandResult
    margin_band_pass: bool
    judges: dict[str, JudgeResult]
    consensus_mean_attribute_fidelity: float
    n_contributing_judges: int
    fidelity_pass: bool
    overall_pass: bool
    label: str


def _normalize(text: str) -> str:
    """Lowercase + strip a free-text attribute value for comparison."""
    return text.strip().lower()


def _tokens(text: str) -> set[str]:
    """Word-set tokenization: lowercased, hyphens/commas treated as word boundaries."""
    return set(text.replace("-", " ").replace(",", " ").split())


def score_attribute_match(extracted: str, ground_truth: str) -> float:
    """Fuzzy `[0.0, 1.0]` match score between a judge's free-text extraction and ground truth.

    Pure, deterministic, no I/O -- this is the entire "how do we grade a blind free-text VLM
    extraction against a known-correct value" logic for this skill, factored out so it is unit
    testable on hand-picked strings without ever calling a real judge.

    Rules, checked in order:
    1. Either string empty (after normalization) -> `0.0` (nothing to compare).
    2. Exact match (case/whitespace-insensitive) -> `1.0`.
    3. One string fully contains the other (e.g. extracted `"a red solid t-shirt"` contains
       ground truth `"t-shirt"`) -> `0.85` (strong but not exact -- the judge said MORE or LESS
       than the ground-truth phrase, not something incompatible with it).
    4. Otherwise, word-level Jaccard overlap (`|intersection| / |union|`) -- partial credit for
       a judge extraction that shares some but not all of the ground truth's descriptive words
       (e.g. extracted `"Basic Tops"` vs. ground truth `"Jersey Basic"` -> `1/3`).

    Args:
        extracted: The judge's own free-text value for one attribute dimension.
        ground_truth: The known-correct value for that dimension (never shown to the judge).

    Returns:
        A score in `[0.0, 1.0]`.
    """
    a, b = _normalize(extracted), _normalize(ground_truth)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.85
    tokens_a, tokens_b = _tokens(a), _tokens(b)
    if not tokens_a or not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


def score_attributes(extracted: dict[str, str], ground_truth: dict[str, str]) -> dict[str, float]:
    """Apply `score_attribute_match` across every ground-truth attribute dimension.

    Args:
        extracted: The judge's raw per-dimension extraction (missing keys score `0.0`, not KeyError
            -- a judge omitting a requested field is a real (low) score, not a crash).
        ground_truth: The known-correct value per dimension; this dict's keys define which
            dimensions are scored (so a judge returning EXTRA keys beyond what was asked is simply
            ignored, not penalized or rewarded).

    Returns:
        One score per `ground_truth` key.
    """
    return {
        dim: score_attribute_match(extracted.get(dim, ""), value)
        for dim, value in ground_truth.items()
    }


def mean_score(scores: dict[str, float]) -> float:
    """Mean of a per-attribute score dict; `0.0` for an empty dict (never a `ZeroDivisionError`)."""
    return sum(scores.values()) / len(scores) if scores else 0.0


def binarize_scores(
    scores: dict[str, float], threshold: float = DEFAULT_BINARIZE_THRESHOLD
) -> dict[str, int]:
    """Threshold a per-attribute score dict into binary present/absent calls (>= `threshold`)."""
    return {dim: int(value >= threshold) for dim, value in scores.items()}


def cohens_kappa(judge_a: Sequence[int], judge_b: Sequence[int]) -> float:
    """Cohen's kappa (chance-corrected agreement) between two judges' paired binary calls.

    Standard formula: `kappa = (p_o - p_e) / (1 - p_e)`, where `p_o` is observed agreement and
    `p_e` is the agreement expected by chance given each judge's own marginal positive rate.

    Args:
        judge_a: Judge A's binary (0/1) calls, e.g. one entry per attribute x concept pair
            (`binarize_scores` output, flattened and paired in the same order as `judge_b`).
        judge_b: Judge B's binary calls, same length and pairing order as `judge_a`.

    Returns:
        Kappa in roughly `[-1.0, 1.0]` (1.0 = perfect agreement, 0.0 = chance-level agreement,
        negative = worse than chance). The degenerate case `p_e == 1.0` (both judges' marginals
        are simultaneously fully one-sided) implies `p_o == 1.0` too (see docstring proof in the
        module's test suite) and is defined as `kappa = 1.0` (trivial perfect agreement) rather
        than raising a `ZeroDivisionError`.

    Raises:
        ValueError: if the two sequences differ in length, or are empty.
    """
    if len(judge_a) != len(judge_b):
        raise ValueError(
            f"judge_a and judge_b must be the same length, got {len(judge_a)} vs {len(judge_b)}"
        )
    n = len(judge_a)
    if n == 0:
        raise ValueError("judge_a/judge_b must be non-empty")
    p_observed = sum(1 for a, b in zip(judge_a, judge_b, strict=True) if a == b) / n
    a_positive_rate = sum(judge_a) / n
    b_positive_rate = sum(judge_b) / n
    p_expected = a_positive_rate * b_positive_rate + (1 - a_positive_rate) * (1 - b_positive_rate)
    if p_expected == 1.0:
        return 1.0
    return (p_observed - p_expected) / (1 - p_expected)


def is_self_scoring_contamination(judge_name: str, generation_backend: str) -> bool:
    """Whether `judge_name`'s model family generated the image via `generation_backend`.

    A judge must never score an image its own provider family generated -- family match is the
    rule, not a hardcoded pairing: `judge_name == generation_backend` (both are provider-family
    identifiers, e.g. `"gemini"`; this project's judges are `"gemini"` and `"groq"`, and its
    generation backends are `nss.generate.backends.LOCAL_SDXL` ("local_sdxl") and
    `nss.generate.backends.GEMINI` ("gemini")). Only the "gemini" judge vs. "gemini" backend
    pairing can trigger in this project today (no Groq-backend generator exists) -- enforced
    unconditionally anyway, since the rule must hold regardless of which pairings happen to be
    reachable right now.

    Args:
        judge_name: The judge's provider-family identifier.
        generation_backend: The `generation_backend` identifier of the image being scored.

    Returns:
        `True` iff the judge must be excluded from scoring this image.
    """
    return judge_name == generation_backend


def run_judge(
    judge_name: str,
    caller: JudgeCaller | None,
    image_path: Path,
    attribute_dimensions: Sequence[str],
    ground_truth: dict[str, str],
    generation_backend: str,
    unavailable_reason: str | None = None,
) -> JudgeResult:
    """Run one judge against one image: contamination check, call, score -- or a documented excl.

    Three ways a `JudgeResult` ends up `available=False`, each distinctly reported in
    `excluded_reason` (never silently indistinguishable from a genuine score of `0.0`):
    1. `caller is None` (the judge was never configured/reachable for this run at all --
       `unavailable_reason` should explain why, e.g. a missing API key).
    2. `is_self_scoring_contamination` (see that function).
    3. `caller(...)` raises -- `JudgeUnavailableError` for the documented "recoverable, expected"
       case, or ANY other exception, which is still captured here (message preserved in
       `excluded_reason`, printed by callers, never discarded) rather than crashing the whole QC
       gate over one judge's failure -- the same "one sub-task's failure must not block the rest"
       convention `nss.generate.final_concepts.generate_gemini_candidate` already uses for C6.

    Args:
        judge_name: This judge's provider-family identifier (see `is_self_scoring_contamination`).
        caller: The judge's `JudgeCaller`, or `None` if never configured/reachable for this run.
        image_path: The concept image to score.
        attribute_dimensions: Which attribute dimensions to ask the judge about (blind -- see
            `SKILL.md`'s "Blindness contract" section).
        ground_truth: The known-correct value per dimension, used ONLY for scoring after the
            (blind) call returns -- never passed into `caller`.
        generation_backend: The identifier of whatever generated `image_path` (see
            `is_self_scoring_contamination`).
        unavailable_reason: Explanation to record when `caller is None`.

    Returns:
        A `JudgeResult`.
    """
    if caller is None:
        return JudgeResult(
            judge_name=judge_name,
            available=False,
            raw_extraction=None,
            scores=None,
            mean_score=None,
            excluded_reason=unavailable_reason or f"{judge_name} judge not configured for this run",
        )
    if is_self_scoring_contamination(judge_name, generation_backend):
        return JudgeResult(
            judge_name=judge_name,
            available=False,
            raw_extraction=None,
            scores=None,
            mean_score=None,
            excluded_reason=(
                f"self-scoring contamination: {judge_name} judge excluded from scoring a "
                f"{generation_backend}-generated concept"
            ),
        )
    try:
        raw = caller(image_path, attribute_dimensions)
    except Exception as exc:
        return JudgeResult(
            judge_name=judge_name,
            available=False,
            raw_extraction=None,
            scores=None,
            mean_score=None,
            excluded_reason=f"{judge_name} judge call failed: {exc}",
        )
    scores = score_attributes(raw, ground_truth)
    return JudgeResult(
        judge_name=judge_name,
        available=True,
        raw_extraction=raw,
        scores=scores,
        mean_score=mean_score(scores),
        excluded_reason=None,
    )


def combine_judges(judge_results: dict[str, JudgeResult]) -> tuple[float, int]:
    """Consensus mean attribute fidelity across every AVAILABLE judge in `judge_results`.

    Args:
        judge_results: Output of `run_judge`, keyed by judge name.

    Returns:
        `(consensus_mean_attribute_fidelity, n_contributing_judges)` -- `0.0`/`0` if no judge was
        available (never a `ZeroDivisionError`; a caller should treat `n_contributing_judges == 0`
        as "no fidelity signal at all," not "fidelity confirmed at 0.0").
    """
    contributing = [
        r["mean_score"]
        for r in judge_results.values()
        if r["available"] and r["mean_score"] is not None
    ]
    if not contributing:
        return 0.0, 0
    return sum(contributing) / len(contributing), len(contributing)


def qc_verdict(
    style_id: str,
    margin: MarginBandResult,
    judge_results: dict[str, JudgeResult],
    attribute_fidelity_threshold: float = DEFAULT_ATTRIBUTE_FIDELITY_THRESHOLD,
) -> QCVerdict:
    """Combine an already-computed margin-band result + judge panel into the final `QCVerdict`.

    `margin_band_pass` requires BOTH `clip_in_band` AND `dino_in_band` -- see `SKILL.md`'s "Design
    notes" and worked examples for why this is deliberately stricter than C6's own CLIP-primary
    selection rule (a re-check gate, not a ranking rule, is held to the stricter joint bar).

    Args:
        style_id: Opaque identifier, passed through unchanged.
        margin: Output of the (external) margin-band scoring step.
        judge_results: Output of `run_judge`, one entry per judge.
        attribute_fidelity_threshold: Minimum `consensus_mean_attribute_fidelity` to pass.

    Returns:
        A `QCVerdict`.
    """
    consensus_fidelity, n_contributing = combine_judges(judge_results)
    margin_band_pass = margin["clip_in_band"] and margin["dino_in_band"]
    fidelity_pass = consensus_fidelity >= attribute_fidelity_threshold
    return QCVerdict(
        style_id=style_id,
        margin=margin,
        margin_band_pass=margin_band_pass,
        judges=judge_results,
        consensus_mean_attribute_fidelity=consensus_fidelity,
        n_contributing_judges=n_contributing,
        fidelity_pass=fidelity_pass,
        overall_pass=margin_band_pass and fidelity_pass,
        label=LLM_CONSENSUS_LABEL,
    )


def choose_next_retry_value(
    last_value: float,
    tried_values: frozenset[float],
    all_values: Sequence[float],
    primary_metric_value: float,
    primary_lower: float,
    primary_upper: float,
    secondary_metric_value: float | None = None,
    secondary_lower: float | None = None,
    secondary_upper: float | None = None,
    secondary_favors_higher: bool = True,
) -> float:
    """Pick the next untried tunable value for a retry, given a monotonic-with-value primary metric.

    Generic retry-value strategy for any QC gate whose generation pipeline exposes ONE scalar
    tunable (this project's `ip_adapter_scale`) with a fixed, already-swept `all_values` range, and
    a PRIMARY band-check metric assumed (from prior sweep evidence -- an assumption the CALLER
    supplies, never computed here) to move monotonically with that tunable.

    Rule, in priority order:
    1. `primary_metric_value` is BELOW `primary_lower` (undershoot): assume the primary metric
       rises with the tunable -- move to the smallest untried value greater than `last_value`
       (the smallest available step in the helpful direction), or the smallest untried value at
       all if none is greater (wraps rather than getting stuck).
    2. `primary_metric_value` is ABOVE `primary_upper` (overshoot): move to the largest untried
       value LESS than `last_value` (smallest step in the helpful direction), if one exists.
       If `last_value` is already the floor of `all_values` (no smaller untried value can exist),
       there is NO tested value expected to help -- take the smallest untried value ABOVE the
       floor anyway (gather direct per-style evidence rather than concluding "impossible" without
       trying -- the monotonic assumption may not hold for every style), with the caller expected
       to document that this step is likely, not certain, to worsen the overshoot.
    3. `primary_metric_value` is already in-band: the failure is the secondary metric and/or
       downstream fidelity scoring. If secondary bounds are supplied, pick the untried value that
       is expected (per `secondary_favors_higher`, an assumption the CALLER supplies from its own
       prior evidence) to help the secondary metric most -- `max(untried)` if
       `secondary_favors_higher`, else `min(untried)`. If no secondary bounds are supplied, default
       to the same "smallest step upward" rule as case 1.

    Args:
        last_value: The most recently tried tunable value.
        tried_values: Every tunable value already tried for this concept (including `last_value`).
        all_values: The full tested range this project's prior sweep evidence covers.
        primary_metric_value: The primary metric's most recent reading.
        primary_lower: Primary metric's band lower bound.
        primary_upper: Primary metric's band upper bound.
        secondary_metric_value: The secondary metric's most recent reading, or `None` if not
            relevant to this decision (only consulted when the primary metric is already in-band).
        secondary_lower: Secondary metric's band lower bound.
        secondary_upper: Secondary metric's band upper bound.
        secondary_favors_higher: Whether prior evidence shows the secondary metric improves at
            higher tunable values (`True`) or lower ones (`False`).

    Returns:
        The next tunable value to retry generation at.

    Raises:
        ValueError: if every value in `all_values` has already been tried.
    """
    untried = sorted(v for v in all_values if v not in tried_values)
    if not untried:
        raise ValueError(
            "every value in all_values has already been tried -- no retry value remains"
        )

    if primary_metric_value < primary_lower:
        higher = [v for v in untried if v > last_value]
        return min(higher) if higher else min(untried)

    if primary_metric_value > primary_upper:
        lower = [v for v in untried if v < last_value]
        if lower:
            return max(lower)
        return min(untried)

    secondary_known = (
        secondary_metric_value is not None
        and secondary_lower is not None
        and secondary_upper is not None
    )
    if secondary_known:
        return max(untried) if secondary_favors_higher else min(untried)

    higher = [v for v in untried if v > last_value]
    return min(higher) if higher else min(untried)
