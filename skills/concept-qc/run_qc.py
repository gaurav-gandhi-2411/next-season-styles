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
DEFAULT_COPY_ANCHOR_DISCOUNT = 0.10
LLM_CONSENSUS_LABEL = "LLM-consensus (NOT human ground truth)"

# Gate 1's default metric set -- both CLIP and DINOv2 required (the original, strict joint-AND
# convention). Calling code with evidence that one metric is non-discriminative even after
# correcting its anchor construction (see `copy_check_pass`'s docstring) can narrow this via its
# own `active_metrics` argument -- never done unconditionally by this generic skill module, since
# "is metric X discriminative" is a per-project, per-measurement finding, not a skill-level default.
GATE1_METRICS: frozenset[str] = frozenset({"clip", "dinov2"})

# Gate 2's default threshold FRACTION (task F2): a judge's pass threshold is `fraction * that
# judge's own positive-control calibration mean`, not a flat absolute score -- see
# `judge_fidelity_threshold`.
DEFAULT_FIDELITY_THRESHOLD_FRACTION = 0.75


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
    """One image's already-computed CLIP + DINOv2 two-sided margin-band scoring result.

    DIAGNOSTIC ONLY as of task E2 -- this two-sided band (estimated from real-catalogue reference
    images, cross-applied to a DIFFERENT distribution of generated images) no longer gates
    `overall_pass`. It is still computed and reported alongside every `QCVerdict` for comparison,
    but the actual pass/fail decision is `copy_check_pass` (Gate 1, see `CopyCheckResult`) AND
    `fidelity_pass` (Gate 2). See `SKILL.md`'s "Gate 1: the sign-safe discount formula" section for
    the full rationale.

    Computed entirely OUTSIDE this module (`nss.generate.clip_scoring` / `dino_scoring` /
    `margin_scoring`) -- this skill only reads the already-derived booleans/floats, never
    recomputes an embedding or a margin itself.
    """

    clip_margin: float
    clip_in_band: bool
    dino_margin: float
    dino_in_band: bool


class CopyCheckResult(TypedDict):
    """One image's Gate-1 ("not a copy") result: BOTH CLIP and DINOv2 margins checked against a
    per-style, per-embedding-space threshold derived from that style's OWN `copy_anchor_gen` (task
    E1's generated-space anchors -- see `SKILL.md`'s "Gate 1" section). Replaces the DIAGNOSTIC-ONLY
    `MarginBandResult.clip_in_band`/`dino_in_band` as the actual gating signal (task E2).
    """

    clip_margin: float
    clip_copy_anchor_gen: float
    clip_copy_anchor_threshold: float
    clip_below_copy_anchor: bool
    dino_margin: float
    dino_copy_anchor_gen: float
    dino_copy_anchor_threshold: float
    dino_below_copy_anchor: bool


def copy_anchor_threshold(
    copy_anchor_gen: float, discount: float = DEFAULT_COPY_ANCHOR_DISCOUNT
) -> float:
    """A `discount`-fraction-stricter Gate-1 threshold from a per-style `copy_anchor_gen` mean.

    SIGN-SAFE BY CONSTRUCTION -- this is the reason the formula is `copy_anchor_gen - discount *
    abs(copy_anchor_gen)` and NOT the naive `copy_anchor_gen * (1 - discount)`. Both formulas agree
    when `copy_anchor_gen` is positive (`0.90 * x == x - 0.10 * abs(x)` for `x > 0`), but they
    DISAGREE, in exactly the direction that matters, when `copy_anchor_gen` is negative (this
    project's Sweater-style CLIP anchor, `-0.0472`): multiplying a negative number by `0.90` moves
    it TOWARD zero (`-0.0472 * 0.90 = -0.0425`, a HIGHER/less-negative value than the anchor
    itself), which would make Gate 1 *easier* to pass than the un-discounted anchor -- the opposite
    of what a "10% stricter" threshold is supposed to do. Subtracting `discount * abs(x)` instead
    always moves the threshold AWAY from zero in the same direction the anchor already points
    (`-0.0472 - 0.10 * 0.0472 = -0.0519`, MORE negative, i.e. strictly stricter), so the invariant
    `copy_anchor_threshold(x, discount) < x` holds for any nonzero `x` and any `discount > 0`,
    regardless of `x`'s sign -- see this module's test suite for a parametrized check of exactly
    that invariant over both a positive and a negative anchor.

    Args:
        copy_anchor_gen: This style's mean `copy_anchor_gen` margin (task E1's
            `reports/tables/margin_anchors_generated_space.csv`), in ONE embedding space (CLIP or
            DINOv2 -- call this once per space).
        discount: Fraction of the anchor's own magnitude to subtract, moving the threshold strictly
            away from zero in the anchor's own direction (default `0.10`, i.e. 10% stricter).

    Returns:
        The Gate-1 threshold: a generated image's margin in this embedding space must be STRICTLY
        BELOW this value to pass (see `copy_check`).
    """
    return copy_anchor_gen - discount * abs(copy_anchor_gen)


def copy_check(
    clip_margin: float,
    dino_margin: float,
    clip_copy_anchor_gen: float,
    dino_copy_anchor_gen: float,
    discount: float = DEFAULT_COPY_ANCHOR_DISCOUNT,
) -> CopyCheckResult:
    """Gate 1 ("not a copy"): score one image's CLIP + DINOv2 margins against their per-style
    copy-anchor thresholds.

    Args:
        clip_margin: The image's already-computed CLIP margin.
        dino_margin: The image's already-computed DINOv2 margin.
        clip_copy_anchor_gen: This style's `copy_anchor_gen` CLIP mean (task E1).
        dino_copy_anchor_gen: This style's `copy_anchor_gen` DINOv2 mean (task E1).
        discount: See `copy_anchor_threshold`.

    Returns:
        A `CopyCheckResult` with both per-space thresholds and pass booleans.
    """
    clip_threshold = copy_anchor_threshold(clip_copy_anchor_gen, discount)
    dino_threshold = copy_anchor_threshold(dino_copy_anchor_gen, discount)
    return CopyCheckResult(
        clip_margin=clip_margin,
        clip_copy_anchor_gen=clip_copy_anchor_gen,
        clip_copy_anchor_threshold=clip_threshold,
        clip_below_copy_anchor=clip_margin < clip_threshold,
        dino_margin=dino_margin,
        dino_copy_anchor_gen=dino_copy_anchor_gen,
        dino_copy_anchor_threshold=dino_threshold,
        dino_below_copy_anchor=dino_margin < dino_threshold,
    )


def copy_check_pass(
    result: CopyCheckResult, active_metrics: frozenset[str] = GATE1_METRICS
) -> bool:
    """Gate-1 verdict: every ACTIVE metric's margin must be below its copy-anchor threshold.

    Defaults to BOTH CLIP and DINOv2 (`GATE1_METRICS`) -- the original, strict joint-AND
    convention `MarginBandResult`'s old `margin_band_pass` used, kept for consistency (see
    `SKILL.md`'s "Design notes"). `active_metrics` lets calling code DROP a metric it has measured
    to be non-discriminative between the "genuinely a copy" and "genuinely unrelated" calibration
    endpoints, even AFTER correcting that metric's own anchor construction (a metric with no real
    separation between those two endpoints contributes noise, not signal, to an AND-of-both check
    -- see `SKILL.md`'s "Gate 1: dropping a non-discriminative metric" section for a worked
    example with real numbers).

    Args:
        result: Output of `copy_check`.
        active_metrics: Which metrics (`"clip"`, `"dinov2"`) must independently pass for Gate 1 to
            pass overall. Must be non-empty and contain only known metric names.

    Returns:
        `True` iff every metric named in `active_metrics` has its `*_below_copy_anchor` boolean
        `True` in `result`.

    Raises:
        ValueError: if `active_metrics` is empty (a Gate 1 with zero active metrics is not a gate,
            it is an unconditional pass, which must never happen silently -- see rule 98a) or
            contains an unrecognized metric name.
    """
    if not active_metrics:
        raise ValueError("active_metrics must be non-empty -- Gate 1 needs at least one metric")
    unknown = active_metrics - GATE1_METRICS
    if unknown:
        raise ValueError(f"unknown metric(s) in active_metrics: {sorted(unknown)}")
    checks: list[bool] = []
    if "clip" in active_metrics:
        checks.append(result["clip_below_copy_anchor"])
    if "dinov2" in active_metrics:
        checks.append(result["dino_below_copy_anchor"])
    return all(checks)


class JudgeResult(TypedDict):
    """One judge's result for one image: either a scored extraction, or a documented exclusion."""

    judge_name: str
    available: bool
    raw_extraction: dict[str, str] | None
    scores: dict[str, float] | None
    mean_score: float | None
    excluded_reason: str | None


class QCVerdict(TypedDict):
    """The combined copy-check + blind-VLM-panel QC verdict for one generated concept image.

    `overall_pass` is `copy_check_pass AND fidelity_pass` (task E2) -- `margin`/`margin_band_pass`
    (the OLD two-sided real-space band) are retained verbatim as a DIAGNOSTIC field only, reported
    for comparison but no longer part of the gate. See `SKILL.md`'s "Gate 1" section.
    """

    style_id: str
    margin: MarginBandResult
    margin_band_pass: bool
    copy_check: CopyCheckResult
    copy_check_pass: bool
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


def judge_fidelity_threshold(
    positive_control_mean: float, fraction: float = DEFAULT_FIDELITY_THRESHOLD_FRACTION
) -> float:
    """One judge's Gate-2 pass threshold: `fraction` of that judge's OWN calibration ceiling
    (task F2).

    Replaces a flat, judge-agnostic threshold (e.g. `DEFAULT_ATTRIBUTE_FIDELITY_THRESHOLD`, picked
    without reference to any calibration data) with a PER-JUDGE value derived from what that judge
    actually scores on a genuine positive control (a real image checked against its own TRUE
    attributes) -- different judges can have very different achievable ceilings even when both are
    doing a good job (verbose or hedging free-text extractions score lower under
    `score_attribute_match`'s exact/containment/Jaccard rules than terse, well-matched ones, for
    reasons that have nothing to do with attribute correctness). A flat threshold picked above one
    judge's own ceiling makes Gate 2 mathematically impossible for that judge to ever pass, even on
    genuine ground truth -- see `SKILL.md`'s "Gate 2" section for this project's real numbers.

    Args:
        positive_control_mean: This judge's mean `mean_score` across its positive-control runs
            (real images checked against their own true attributes) -- the judge's own calibration
            ceiling.
        fraction: Fraction of that ceiling required to pass (default `0.75`, i.e. "must reach at
            least three-quarters of what this judge scores on a genuine, correctly-labeled image"
            -- the same 75% figure the old flat threshold used, now correctly scaled per judge
            instead of applied as an absolute score).

    Returns:
        The per-judge Gate-2 threshold.
    """
    return fraction * positive_control_mean


def per_judge_fidelity_pass(
    available_judge_scores: dict[str, float], thresholds: dict[str, float]
) -> dict[str, bool]:
    """Per-judge Gate-2 pass/fail: each judge's mean_score >= its OWN calibrated threshold.

    Args:
        available_judge_scores: `{judge_name: mean_score}` for judges that actually produced a
            score this run -- an unavailable judge must simply be ABSENT from this dict, never
            included with a `None`/`0.0` placeholder (mirrors `combine_judges`'s convention of
            only ever summing over judges that actually contributed a score).
        thresholds: `{judge_name: threshold}` (typically `judge_fidelity_threshold`'s output, one
            call per judge).

    Returns:
        `{judge_name: bool}`, one entry per key in `available_judge_scores`.

    Raises:
        KeyError: if a judge present in `available_judge_scores` has no entry in `thresholds` --
            fail loud rather than silently skipping a judge whose threshold was never computed
            (e.g. missing calibration data for that judge), per rule 98a: "couldn't verify" must
            never be treated as a silent pass.
    """
    return {name: score >= thresholds[name] for name, score in available_judge_scores.items()}


def fidelity_pass_from_per_judge(
    available_judge_scores: dict[str, float], thresholds: dict[str, float]
) -> bool:
    """Gate-2 verdict under PER-JUDGE thresholds: every available judge must independently pass.

    Same strict joint-AND convention Gate 1 uses (`copy_check_pass`'s default `GATE1_METRICS`) --
    one judge's pass cannot outvote another's fail, and a judge that produced no score at all
    contributes neither a pass nor a fail (it is simply absent from `available_judge_scores`).

    Args:
        available_judge_scores: `{judge_name: mean_score}` for judges that actually scored this
            concept (see `per_judge_fidelity_pass`).
        thresholds: `{judge_name: threshold}`.

    Returns:
        `False` if `available_judge_scores` is empty (no fidelity signal at all is never an
        unconditional pass -- mirrors `combine_judges`'s `n_contributing_judges == 0` convention);
        otherwise `True` iff every available judge's own pass/fail (`per_judge_fidelity_pass`) is
        `True`.
    """
    if not available_judge_scores:
        return False
    return all(per_judge_fidelity_pass(available_judge_scores, thresholds).values())


def _available_judge_scores(judge_results: dict[str, JudgeResult]) -> dict[str, float]:
    """`{judge_name: mean_score}` for every judge in `judge_results` that actually scored."""
    return {
        name: r["mean_score"]
        for name, r in judge_results.items()
        if r["available"] and r["mean_score"] is not None
    }


def within_style_novelty_pass(
    concept_similarity: dict[str, float], benchmark_median: dict[str, float]
) -> bool:
    """Gate 1 (task H1): is the concept no more similar to its references than real siblings are?

    `concept_similarity[space]` is the concept's mean cosine to its style's reference images in an
    embedding space; `benchmark_median[space]` is the median pairwise cosine between DISTINCT real
    articles of that same style in the same space. Passes iff EVERY space is at or below its
    benchmark (joint AND, same convention as `copy_check_pass`).

    Raises:
        ValueError: if the two dicts are empty or do not name exactly the same spaces -- a gate that
            silently skips a space it has no benchmark for is an unconditional pass (rule 98a).
    """
    if not concept_similarity or set(concept_similarity) != set(benchmark_median):
        raise ValueError(
            "concept_similarity and benchmark_median must be non-empty and name the same spaces; "
            f"got {sorted(concept_similarity)} vs {sorted(benchmark_median)}"
        )
    return all(concept_similarity[s] <= benchmark_median[s] for s in concept_similarity)


def qc_verdict(
    style_id: str,
    margin: MarginBandResult,
    copy_check_result: CopyCheckResult,
    judge_results: dict[str, JudgeResult],
    attribute_fidelity_threshold: float = DEFAULT_ATTRIBUTE_FIDELITY_THRESHOLD,
    active_metrics: frozenset[str] = GATE1_METRICS,
    per_judge_fidelity_thresholds: dict[str, float] | None = None,
) -> QCVerdict:
    """Combine an already-computed copy-check result + judge panel into the final `QCVerdict`.

    `overall_pass` (task E2) is `copy_check_pass AND fidelity_pass` -- TWO one-sided tests: Gate 1
    (`copy_check_pass`, every ACTIVE metric's margin below its per-style copy-anchor threshold --
    see `active_metrics`/`copy_check_pass`) and Gate 2 (`fidelity_pass`, blind VLM attribute
    fidelity). The OLD two-sided `margin_band_pass` (BOTH `clip_in_band` AND `dino_in_band` against
    the real-space band) is still computed and reported as a DIAGNOSTIC field but no longer gates
    `overall_pass` -- see `SKILL.md` for the full rationale.

    Args:
        style_id: Opaque identifier, passed through unchanged.
        margin: Output of the (external) two-sided margin-band scoring step -- diagnostic only.
        copy_check_result: Output of `copy_check` -- the actual Gate-1 signal.
        judge_results: Output of `run_judge`, one entry per judge.
        attribute_fidelity_threshold: Minimum `consensus_mean_attribute_fidelity` to pass Gate 2
            UNDER THE OLD, flat-threshold convention -- only used when
            `per_judge_fidelity_thresholds` is `None` (the default, backward-compatible path).
        active_metrics: Which Gate-1 metrics (`copy_check_pass`'s `active_metrics`) must pass.
            Defaults to both (`GATE1_METRICS`) -- the original behavior.
        per_judge_fidelity_thresholds: `{judge_name: threshold}` (task F2, typically
            `judge_fidelity_threshold`'s output per judge). When supplied, Gate 2 uses
            `fidelity_pass_from_per_judge` (every available judge independently above ITS OWN
            threshold) INSTEAD OF the old flat-threshold-on-the-consensus-mean check --
            `attribute_fidelity_threshold` is then ignored. `None` (the default) preserves the old
            behavior unchanged, for backward compatibility.

    Returns:
        A `QCVerdict`. `consensus_mean_attribute_fidelity` is always the plain cross-judge mean
        (`combine_judges`'s output) for reporting/diagnostics, regardless of which threshold
        convention actually gates `fidelity_pass`.
    """
    consensus_fidelity, n_contributing = combine_judges(judge_results)
    margin_band_pass = margin["clip_in_band"] and margin["dino_in_band"]
    copy_pass = copy_check_pass(copy_check_result, active_metrics=active_metrics)
    if per_judge_fidelity_thresholds is not None:
        fidelity_pass = fidelity_pass_from_per_judge(
            _available_judge_scores(judge_results), per_judge_fidelity_thresholds
        )
    else:
        fidelity_pass = consensus_fidelity >= attribute_fidelity_threshold
    return QCVerdict(
        style_id=style_id,
        margin=margin,
        margin_band_pass=margin_band_pass,
        copy_check=copy_check_result,
        copy_check_pass=copy_pass,
        judges=judge_results,
        consensus_mean_attribute_fidelity=consensus_fidelity,
        n_contributing_judges=n_contributing,
        fidelity_pass=fidelity_pass,
        overall_pass=copy_pass and fidelity_pass,
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
