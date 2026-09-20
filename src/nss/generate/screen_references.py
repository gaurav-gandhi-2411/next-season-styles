"""Screen candidate IP-Adapter reference images by framing: full-garment product shot
vs. texture-crop/close-up, via blind VLM classification.

WHY THIS EXISTS: an earlier generation run found every Sweater candidate was a degenerate
fabric-texture close-up and traced the mechanism to `nss.generate.backends`'s DOCUMENTED
single-IP-Adapter-reference limit (that module's docstring note 1): `diffusers`'
`prepare_ip_adapter_image_embeds` requires exactly one image per *loaded* IP-Adapter, so
`_generate_local_sdxl` only ever conditions on `reference_images[0]` -- confirmed again here by
reading `backends.py` directly, not re-derived. That run reported this as "out of scope" because
building multi-reference IP-Adapter support (one adapter per reference image) is a real architecture
change, correctly out of scope for a QC-gate change.

ROOT CAUSE IS NOT MISSING DATA (confirmed, not assumed): the exemplar fetch got 8 candidate
reference images per final-three style (`reports/tables/exemplar_images_final_three.csv`,
`final_rank_*` rows) -- 23/24 fetched successfully (one T-shirt article, `0610776002`, failed fetch
and is correctly excluded downstream). The bug is in ORDERING: `nss.generate.final_concepts.
load_final_three_references` (via `nss.generate.derive_similarity_band.group_by_style`) sorts each
style's candidate paths by `Path.sort()`, i.e. lexicographically by
`data/images/<article_id>.jpg` -- a sort key with NO relationship to framing quality. For the
Sweater style that arbitrary sort put article `0673677023` (H&M's own texture-detail product shot
for that article) at index 0, which is exactly the image `_generate_local_sdxl` conditions on.

THE FIX (architectural limitation accepted, ordering/selection bug fixed): since multi-reference
IP-Adapter conditioning is out of scope (the same call as before, upheld here), this module makes
`reference_images[0]` the BEST available image instead of an arbitrary one, for EVERY final style
(not just the Sweater) --

1. Classify every candidate reference image, blind, as `"full_garment"` or `"texture_crop"` via
   both project VLM judges (Gemini always attempted, Groq if reachable -- same
   `nss.generate.vlm_judges` availability convention `concept_qc_pipeline.run_judge_panel` uses).
   A candidate is screened IN only if every AVAILABLE judge agrees `"full_garment"` (fail-closed:
   disagreement, or zero judges reachable, excludes the candidate rather than guessing).
2. Keep only full-garment survivors, ordered by `units_sold_last_26w` descending (best-selling
   full-garment shot first) -- `load_screened_references`'s ordering IS the fix for `local_sdxl`'s
   `reference_images[0]`-only limitation. `gemini`-backend generation (which DOES use every
   reference image, see `backends.py` note 2) also benefits: texture-crop references are no longer
   in the list to dilute/derail conditioning on ANY backend.
3. If fewer than `MIN_SURVIVING_BEFORE_FETCH` (3) full-garment images survive for a style, fetch
   next-best-selling constituent articles beyond the original top-8 (reusing
   `nss.data.select_exemplars`/`nss.data.fetch_images` verbatim) and screen those too, up to
   `MAX_FETCH_ROUNDS` rounds, until `TARGET_FULL_GARMENT_REFERENCES` (4) survive or the style's
   catalogue is genuinely exhausted (`exhausted=True` in the returned summary -- a real finding,
   reported, never silently padded).

CANONICAL SOURCE, POST-SCREENING (documented): this module writes a NEW file,
`reports/tables/exemplar_images_screened.csv` -- it does NOT overwrite `exemplar_images_final_three.
csv` (the fetch-provenance manifest stays exactly as it was written; this module's output is a
DERIVED, VLM-screening-provenance manifest layered on top, so both the original fetch record and
the screening decision are independently auditable). `nss.generate.final_concepts_v2.main` and
`scripts/run_pipeline.py`'s `generate`/`score` stages (the CURRENTLY ACTIVE generation pipeline)
were repointed from `final_concepts.load_final_three_references` to this module's
`load_screened_references` -- see those modules. `nss.generate.final_concepts.main` (a
historical/superseded artifact per `final_concepts_v2.py`'s own docstring) and
`nss.generate.concept_qc_pipeline.main` (same "superseded" status) are deliberately left
reading the UNSCREENED manifest -- they are not part of the active pipeline and rewriting a
superseded module's data source is out of scope here.

Usage:
    uv run python -m nss.generate.screen_references
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from nss.data.fetch_images import fetch_images, local_path
from nss.data.select_exemplars import (
    ARTICLES_PATH,
    LOOKBACK_WEEKS,
    PANEL_LAST_WEEK,
    compute_lookback_cutoff,
    select_top_selling_articles,
    style_key_values_from_panel,
)
from nss.features.style_panel import STYLE_KEY_COLS
from nss.generate import vlm_judges

FINAL_THREE_MANIFEST_PATH = Path("reports/tables/exemplar_images_final_three.csv")
SCREENED_MANIFEST_PATH = Path("reports/tables/reference_base_widened.csv")
PANEL_PATH = Path("data/processed/style_week_panel.parquet")
IMAGES_DIR = Path("data/images")

FRAMING_FULL_GARMENT = "full_garment"
FRAMING_TEXTURE_CROP = "texture_crop"
FRAMING_LABELS = (FRAMING_FULL_GARMENT, FRAMING_TEXTURE_CROP)

# Trigger a fetch-more round if fewer than this many full-garment images survive screening; keep
# fetching (up to MAX_FETCH_ROUNDS) until at least TARGET_FULL_GARMENT_REFERENCES survive. Per
# Specified thresholds: "<3 survive" triggers, ">=4" is the target.
MIN_SURVIVING_BEFORE_FETCH = 3
TARGET_FULL_GARMENT_REFERENCES = 4
FETCH_BATCH_SIZE = 8  # matches nss.data.select_exemplars.N_EXEMPLARS_PER_STYLE
MAX_FETCH_ROUNDS = 3

_SCREENED_MANIFEST_COLUMNS = [
    "style_id",
    "article_id",
    "image_path",
    "units_sold_last_26w",
    "newly_fetched",
    "gemini_available",
    "gemini_label",
    "gemini_excluded_reason",
    "groq_available",
    "groq_label",
    "groq_excluded_reason",
    "is_full_garment",
    "screening_note",
]


def build_framing_classification_prompt() -> str:
    """Build the blind framing-classification prompt sent to every judge.

    Deliberately separate from `vlm_judges.build_judge_prompt` (attribute-fidelity scoring) -- this
    is a simple binary classification, not a per-dimension free-text extraction.

    Returns:
        A prompt instructing the judge to classify ONLY the photograph's framing/composition.
    """
    return (
        "You are a blind fashion-catalogue QC analyst. You are shown ONE product photograph. You "
        "are NOT told what product, brand, or style this is -- judge only the FRAMING/composition "
        "of the photograph itself. Classify it into exactly one of two categories:\n"
        '- "full_garment": the ENTIRE garment is visible in frame (flat-lay, mannequin, '
        "ghost-mannequin, or worn full-body/product shot) -- the overall silhouette/shape is "
        "identifiable.\n"
        '- "texture_crop": a close-up/macro/cropped/zoomed shot showing ONLY fabric texture, a '
        "stitch, a button, a zipper, or another small detail -- the overall garment silhouette is "
        "NOT identifiable from this framing alone.\n\n"
        "Return STRICT JSON with exactly this shape (no other keys, no commentary, no markdown "
        'code fencing): {"framing": "full_garment" or "texture_crop"}'
    )


def _parse_framing_label(text: str | None) -> str:
    """Parse a judge's raw JSON response text into a validated framing label.

    Args:
        text: The judge's raw response text (expected to be `{"framing": "..."}`).

    Returns:
        One of `FRAMING_LABELS`.

    Raises:
        RuntimeError: if `text` is empty/`None`, not valid JSON, or the `framing` value is not one
            of `FRAMING_LABELS`.
    """
    if not text:
        raise RuntimeError("judge response was empty -- cannot parse a framing classification")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"judge response was not valid JSON: {text!r}") from exc
    if not isinstance(parsed, dict) or "framing" not in parsed:
        raise RuntimeError(f"judge response JSON missing 'framing' key: {text!r}")
    label = str(parsed["framing"]).strip()
    if label not in FRAMING_LABELS:
        raise RuntimeError(f"judge returned an unrecognized framing label: {label!r}")
    return label


def _unavailable_result(judge_name: str, reason: str) -> dict[str, Any]:
    """Build an `available=False` judge result dict with a documented reason (never a silent
    `None`)."""
    return {
        "judge_name": judge_name,
        "available": False,
        "label": None,
        "raw_response": None,
        "excluded_reason": reason,
    }


def classify_framing_gemini(image_path: Path, api_key: str | None = None) -> dict[str, Any]:
    """Judge A: blind framing classification via Gemini (`vlm_judges.GEMINI_JUDGE_MODEL_ID`).

    Any failure (missing key, network error, malformed response) is caught and converted to an
    `available=False` result rather than raised -- same "one judge's failure must not block
    screening" convention as `skills/concept-qc/run_qc.py`'s `run_judge` (this is a screening
    filter, not the fail-loud generation path).

    Args:
        image_path: The candidate reference image to classify.
        api_key: Overrides `GEMINI_API_KEY` from the environment/`.env` (mainly for tests).

    Returns:
        `{"judge_name": "gemini", "available": bool, "label": str | None,
        "raw_response": str | None, "excluded_reason": str | None}`.
    """
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types as genai_types
    from PIL import Image

    load_dotenv()
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        return _unavailable_result("gemini", "GEMINI_API_KEY is not set.")
    try:
        client = genai.Client(api_key=key)
        image = Image.open(image_path).convert("RGB")
        response = client.models.generate_content(
            model=vlm_judges.GEMINI_JUDGE_MODEL_ID,
            contents=[build_framing_classification_prompt(), image],
            config=genai_types.GenerateContentConfig(response_mime_type="application/json"),
        )
        label = _parse_framing_label(response.text)
    except Exception as exc:  # noqa: BLE001 -- screening judge, must not block the rest (see above)
        return _unavailable_result("gemini", f"Gemini framing classification failed: {exc}")
    return {
        "judge_name": "gemini",
        "available": True,
        "label": label,
        "raw_response": response.text,
        "excluded_reason": None,
    }


def classify_framing_groq(image_path: Path, api_key: str | None = None) -> dict[str, Any]:
    """Judge B: blind framing classification via Groq (`vlm_judges.GROQ_JUDGE_MODEL_ID`).

    Same soft-fail convention as `classify_framing_gemini` -- see that function's docstring.

    Args:
        image_path: The candidate reference image to classify.
        api_key: Overrides `GROQ_API_KEY` from the environment/`.env` (mainly for tests).

    Returns:
        Same shape as `classify_framing_gemini`, with `judge_name="groq"`.
    """
    import base64

    from dotenv import load_dotenv
    from groq import Groq

    load_dotenv()
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        return _unavailable_result("groq", "GROQ_API_KEY is not set.")
    try:
        client = Groq(api_key=key)
        image_bytes = image_path.read_bytes()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        response = client.chat.completions.create(
            model=vlm_judges.GROQ_JUDGE_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_framing_classification_prompt()},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{encoded}"},
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        label = _parse_framing_label(content)
    except Exception as exc:  # noqa: BLE001 -- screening judge, must not block the rest (see above)
        return _unavailable_result("groq", f"Groq framing classification failed: {exc}")
    return {
        "judge_name": "groq",
        "available": True,
        "label": label,
        "raw_response": content,
        "excluded_reason": None,
    }


def _consensus_is_full_garment(
    gemini_result: dict[str, Any], groq_result: dict[str, Any]
) -> tuple[bool, str]:
    """Fail-closed consensus: full-garment only if every AVAILABLE judge agrees.

    Args:
        gemini_result: Output of `classify_framing_gemini`.
        groq_result: Output of `classify_framing_groq` (or an `_unavailable_result` if Groq was
            never attempted this run).

    Returns:
        `(is_full_garment, note)` -- `note` always explains the verdict, so a screening decision is
        never opaque (rule 98a's fail-closed principle: "couldn't verify" -> excluded, not passed).
    """
    available_labels = [
        r["label"] for r in (gemini_result, groq_result) if r["available"] and r["label"]
    ]
    if not available_labels:
        return False, "no judge available -- screening inconclusive, excluded (fail-closed)"
    if any(label == FRAMING_TEXTURE_CROP for label in available_labels):
        return False, f"at least one available judge labeled texture_crop ({available_labels})"
    return True, f"all available judge(s) labeled full_garment ({available_labels})"


def classify_reference_image(
    image_path: Path, groq_available: bool, groq_detail: str = ""
) -> dict[str, Any]:
    """Classify one reference image's framing with both judges and combine (fail-closed).

    Gemini is always attempted; Groq only if `groq_available` (same one-time-check convention as
    `concept_qc_pipeline.run_judge_panel` -- never re-checked per image).

    Args:
        image_path: The candidate reference image to classify.
        groq_available: Result of a one-time `vlm_judges.check_groq_availability()` call.
        groq_detail: Detail string from that same check, folded into Groq's `excluded_reason` when
            `groq_available` is `False`.

    Returns:
        A flat dict: `image_path`, both judges' `available`/`label`/`excluded_reason`,
        `is_full_garment`, `screening_note`.
    """
    gemini_result = classify_framing_gemini(image_path)
    groq_result = (
        classify_framing_groq(image_path)
        if groq_available
        else _unavailable_result(
            "groq",
            f"GROQ_API_KEY set, but the vision judge model is unreachable: {groq_detail}"
            if groq_detail
            else "GROQ_API_KEY not set",
        )
    )
    is_full_garment, note = _consensus_is_full_garment(gemini_result, groq_result)
    return {
        "image_path": str(image_path),
        "gemini_available": gemini_result["available"],
        "gemini_label": gemini_result["label"],
        "gemini_excluded_reason": gemini_result["excluded_reason"],
        "groq_available": groq_result["available"],
        "groq_label": groq_result["label"],
        "groq_excluded_reason": groq_result["excluded_reason"],
        "is_full_garment": is_full_garment,
        "screening_note": note,
    }


def screen_candidates(
    style_id: str,
    candidates: Sequence[dict[str, Any]],
    groq_available: bool,
    groq_detail: str = "",
    classify_fn: Callable[[Path, bool, str], dict[str, Any]] = classify_reference_image,
    newly_fetched: bool = False,
) -> list[dict[str, Any]]:
    """Screen a batch of one style's candidate reference images, carrying metadata through.

    `classify_fn` is injected so this is fully unit-testable with a fake (no real API calls) --
    see `tests/test_screen_references.py`.

    Args:
        style_id: The style's composite `style_key`.
        candidates: `{"article_id": int, "local_image_path": Path, "units_sold_last_26w": int}`
            dicts (matches `exemplar_images_final_three.csv`'s schema, one per candidate).
        groq_available: One-time `vlm_judges.check_groq_availability()` result.
        groq_detail: Detail string from that same check.
        classify_fn: `(image_path, groq_available, groq_detail) -> classification dict`.
        newly_fetched: Tags every row with whether these candidates came from a fetch-more round
            (vs. the original top-8 fetch) -- purely provenance, never affects screening.

    Returns:
        One dict per candidate: `style_id`, `article_id`, `units_sold_last_26w`, `newly_fetched`,
        plus every field `classify_fn` returns.
    """
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        result = classify_fn(Path(candidate["local_image_path"]), groq_available, groq_detail)
        rows.append(
            {
                "style_id": style_id,
                "article_id": int(candidate["article_id"]),
                "units_sold_last_26w": int(candidate["units_sold_last_26w"]),
                "newly_fetched": newly_fetched,
                **result,
            }
        )
    return rows


def ensure_min_full_garment_references(
    style_id: str,
    style_values: dict[str, str],
    screened: list[dict[str, Any]],
    already_tried_article_ids: set[int],
    groq_available: bool,
    groq_detail: str,
    transactions: pl.DataFrame,
    articles: pl.DataFrame,
    window_start: date,
    window_end: date,
    images_dir: Path = IMAGES_DIR,
    classify_fn: Callable[[Path, bool, str], dict[str, Any]] = classify_reference_image,
    fetch_fn: Callable[[list[int], Path], dict[str, bool]] = fetch_images,
    min_before_fetch: int = MIN_SURVIVING_BEFORE_FETCH,
    target_after_fetch: int = TARGET_FULL_GARMENT_REFERENCES,
    fetch_batch_size: int = FETCH_BATCH_SIZE,
    max_fetch_rounds: int = MAX_FETCH_ROUNDS,
) -> dict[str, Any]:
    """Fetch + screen additional next-best-selling constituent articles until enough survive.

    A no-op (0 fetch rounds) if `screened` already has `>= min_before_fetch` full-garment
    survivors. Otherwise, re-ranks this style's constituent articles by units sold with a
    progressively larger `n` (reusing `select_top_selling_articles` verbatim -- same ranking logic
    the original top-8 fetch used), fetches only the NEW article_ids beyond
    `already_tried_article_ids`, screens them, and repeats until `target_after_fetch` survivors
    are reached or `max_fetch_rounds` is exhausted. `fetch_fn`/`classify_fn` are injected so this
    is fully unit-testable with fakes (no real network/API calls).

    Args:
        style_id: The style's composite `style_key`.
        style_values: This style's 5 `STYLE_KEY_COLS` values (for `select_top_selling_articles`).
        screened: This style's already-screened candidates (output of `screen_candidates`).
        already_tried_article_ids: article_ids already fetched/screened for this style (mutated in
            place as new articles are tried, so a caller inspecting it afterward sees every
            article_id this function ever attempted).
        groq_available: One-time `vlm_judges.check_groq_availability()` result.
        groq_detail: Detail string from that same check.
        transactions: Transaction rows (`article_id`, `t_dat`), already filtered to
            `>= window_start` (matches `select_exemplars.main`'s convention).
        articles: Article metadata with `article_id` and the 5 `STYLE_KEY_COLS` columns.
        window_start: Inclusive start of the sales lookback window.
        window_end: Inclusive end of the sales lookback window.
        images_dir: Local image cache directory.
        classify_fn: Injected for testing (see `screen_candidates`).
        fetch_fn: Injected for testing (`nss.data.fetch_images.fetch_images`'s signature).
        min_before_fetch: Trigger threshold (see module docstring).
        target_after_fetch: Stop-fetching threshold (see module docstring).
        fetch_batch_size: How many additional candidate articles to consider ranking further out
            per round.
        max_fetch_rounds: Maximum fetch-more rounds before giving up and reporting a shortfall.

    Returns:
        `{"screened": list[dict] (original + any newly fetched, screened rows), "rounds_used": int,
        "exhausted": bool, "n_survivors": int}`. `exhausted=True` means the style's catalogue ran
        out of distinct constituent articles with sales in the lookback window before reaching
        `target_after_fetch` -- a genuine finding, not a bug.
    """
    screened = list(screened)
    n_survivors = sum(1 for r in screened if r["is_full_garment"])
    if n_survivors >= min_before_fetch:
        return {
            "screened": screened,
            "rounds_used": 0,
            "exhausted": False,
            "n_survivors": n_survivors,
        }

    rounds_used = 0
    exhausted = False
    while n_survivors < target_after_fetch and rounds_used < max_fetch_rounds:
        rounds_used += 1
        n_needed = len(already_tried_article_ids) + fetch_batch_size
        ranked = select_top_selling_articles(
            transactions, articles, style_values, window_start, window_end, n=n_needed
        )
        new_articles = ranked.filter(~pl.col("article_id").is_in(list(already_tried_article_ids)))
        if new_articles.is_empty():
            exhausted = True
            break

        new_ids = [int(x) for x in new_articles["article_id"].to_list()]
        already_tried_article_ids.update(new_ids)
        fetch_results = fetch_fn(new_ids, images_dir)

        candidates = [
            {
                "article_id": int(row["article_id"]),
                "local_image_path": local_path(int(row["article_id"]), images_dir),
                "units_sold_last_26w": int(row["units_sold_last_26w"]),
            }
            for row in new_articles.iter_rows(named=True)
            if fetch_results.get(str(int(row["article_id"])), False)
        ]
        if not candidates:
            # every newly-ranked article failed to fetch -- try again next round (a fresh,
            # further-out ranking) rather than looping forever on the same failed ids.
            continue

        new_screened = screen_candidates(
            style_id, candidates, groq_available, groq_detail, classify_fn, newly_fetched=True
        )
        screened.extend(new_screened)
        n_survivors = sum(1 for r in screened if r["is_full_garment"])

    return {
        "screened": screened,
        "rounds_used": rounds_used,
        "exhausted": exhausted,
        "n_survivors": n_survivors,
    }


def load_screened_references(path: Path = SCREENED_MANIFEST_PATH) -> dict[str, list[Path]]:
    """Load this module's own screened manifest, keeping only full-garment survivors.

    THE FIX for `backends.py`'s single-IP-Adapter-reference limitation lives here: paths are
    ordered by `units_sold_last_26w` descending (ties broken by `image_path` ascending, for
    determinism) so index 0 -- the ONLY image `local_sdxl` ever conditions on -- is deliberately
    the best-selling full-garment shot, never an arbitrary alphabetically-first one.

    Args:
        path: Path to `exemplar_images_screened.csv` (this module's own `main()` output).

    Returns:
        Mapping of `style_id -> list[Path]` (full-garment survivors only, best-selling first).

    Raises:
        ValueError: if any style_id in the manifest has ZERO full-garment survivors -- diagnosed
            here (at reference-loading time) rather than several calls deeper inside
            `generate_concept`'s own "reference_images must be non-empty" check, with a message
            that names the actual root cause.
    """
    df = pl.read_csv(path)
    grouped: dict[str, list[Path]] = {}
    for style_id in df["style_id"].unique(maintain_order=True).to_list():
        style_df = df.filter((pl.col("style_id") == style_id) & pl.col("is_full_garment")).sort(
            ["units_sold_last_26w", "image_path"], descending=[True, False]
        )
        if style_df.is_empty():
            raise ValueError(
                f"no full-garment survivors for style_id={style_id!r} in {path} -- either this "
                "style's catalogue genuinely lacks a usable full-garment reference image, or "
                "screening was left incomplete by a VLM-provider quota exhaustion (check for "
                "'no judge available' rows via `retry_inconclusive_candidates` before concluding "
                "a genuine shortfall -- see the screening report for this style's finding)"
            )
        grouped[style_id] = [Path(p) for p in style_df["image_path"].to_list()]
    return grouped


def retry_inconclusive_candidates(
    path: Path = SCREENED_MANIFEST_PATH,
    groq_available: bool = True,
    groq_detail: str = "",
    classify_fn: Callable[[Path, bool, str], dict[str, Any]] = classify_reference_image,
    only_original_fetch: bool = True,
) -> pl.DataFrame:
    """Re-classify already-screened rows that were INCONCLUSIVE (no judge reachable) last run,
    without fetching any new candidate images -- a cheap, targeted retry for exactly the shared
    VLM-provider-quota-exhaustion finding documented in the screening report (Gemini's free-tier
    daily cap and Groq's TPD budget were both exhausted mid-run, largely by
    `ensure_min_full_garment_references`'s own aggressive over-fetching once a style fell short).

    Idempotent: a row that is no longer inconclusive after a retry keeps its new result; a row
    that is STILL inconclusive (quota not yet recovered) is written back unchanged -- safe to call
    repeatedly as quota recovers over time.

    Args:
        path: Path to an already-written `exemplar_images_screened.csv` (read AND rewritten in
            place).
        groq_available: `vlm_judges.check_groq_availability()` result for THIS retry attempt (may
            differ from the original run's -- quota recovers over time, never re-derived from the
            stale original-run value).
        groq_detail: Detail string from that same check.
        classify_fn: Injected for testing (see `screen_candidates`).
        only_original_fetch: If `True` (default), only retries rows with `newly_fetched == False`
            (the original top-selling candidates) -- deliberately does NOT retry every
            aggressively-over-fetched `newly_fetched == True` row from a shortfall round, so this
            retry cannot re-consume the same scarce shared quota that made it necessary in the
            first place. Set `False` to retry every inconclusive row, once quota headroom is
            confirmed ample.

    Returns:
        The rewritten `pl.DataFrame`.
    """
    df = pl.read_csv(path)
    rows = df.to_dicts()
    updated_rows: list[dict[str, Any]] = []
    for row in rows:
        is_inconclusive = (not row["gemini_available"]) and (not row["groq_available"])
        if only_original_fetch and row["newly_fetched"]:
            is_inconclusive = False
        if not is_inconclusive:
            updated_rows.append(row)
            continue
        result = classify_fn(Path(row["image_path"]), groq_available, groq_detail)
        updated_rows.append({**row, **result})

    out_df = pl.DataFrame(updated_rows).select(df.columns)
    out_df.write_csv(path)
    return out_df


def main() -> None:
    """Run the full screening pipeline over the current final-three styles' candidate
    reference images: screen -> fetch-more-if-needed -> write `exemplar_images_screened.csv`.

    VLM-PROVIDER QUOTA IS A REAL, RECURRING CONSTRAINT (see the screening report): both Gemini's
    free-tier daily request cap and Groq's shared TPD token budget can be exhausted mid-run --
    especially once `ensure_min_full_garment_references` triggers a fetch-more round, which can
    add dozens of new candidates to classify. A style whose printed summary says `SHORTFALL` at
    the END of this run may simply be quota-blocked, NOT genuinely short on full-garment images --
    do not conclude a genuine catalogue shortfall from this run's console output alone. Re-check
    with `retry_inconclusive_candidates(only_original_fetch=True)` (cheap: only retries the
    original, non-over-fetched candidates) once quota has had time to recover, BEFORE trusting a
    `SHORTFALL` verdict or invoking `load_screened_references`'s fail-closed `ValueError`.
    """
    manifest = pl.read_csv(FINAL_THREE_MANIFEST_PATH)
    final_rows = manifest.filter(
        pl.col("role").str.starts_with("final_rank_") & pl.col("fetch_success")
    )
    groq_available, groq_detail = vlm_judges.check_groq_availability()
    print(f"Groq judge availability: {groq_available} ({groq_detail})")

    panel = pl.read_parquet(PANEL_PATH, columns=["style_key", *STYLE_KEY_COLS])
    style_ids = final_rows["style_key"].unique(maintain_order=True).to_list()

    all_screened: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    transactions: pl.DataFrame | None = None
    articles: pl.DataFrame | None = None
    cutoff: date | None = None
    window_end: date | None = None

    for style_id in style_ids:
        style_rows = final_rows.filter(pl.col("style_key") == style_id)
        already_tried_ids = {int(x) for x in style_rows["article_id"].to_list()}
        candidates = [
            {
                "article_id": int(row["article_id"]),
                "local_image_path": row["local_image_path"],
                "units_sold_last_26w": int(row["units_sold_last_26w"]),
            }
            for row in style_rows.iter_rows(named=True)
        ]
        print(f"\nStyle: {style_id} ({len(candidates)} candidate reference(s))")
        screened = screen_candidates(style_id, candidates, groq_available, groq_detail)
        n_survivors = sum(1 for r in screened if r["is_full_garment"])
        print(f"  {n_survivors}/{len(screened)} full-garment survivor(s) (pre-fetch)")

        rounds_used = 0
        exhausted = False
        if n_survivors < MIN_SURVIVING_BEFORE_FETCH:
            if transactions is None:
                cutoff = compute_lookback_cutoff(PANEL_LAST_WEEK, LOOKBACK_WEEKS)
                articles = pl.read_csv(ARTICLES_PATH).select(["article_id", *STYLE_KEY_COLS])
                from nss.data.select_exemplars import TRANSACTIONS_DIR

                txn_lazy = pl.scan_parquet(str(TRANSACTIONS_DIR / "**" / "*.parquet")).select(
                    ["article_id", "t_dat"]
                )
                window_end = txn_lazy.select(pl.col("t_dat").max()).collect().item()
                transactions = txn_lazy.filter(pl.col("t_dat") >= cutoff).collect(
                    engine="streaming"
                )
            style_values = style_key_values_from_panel(panel, style_id)
            result = ensure_min_full_garment_references(
                style_id,
                style_values,
                screened,
                already_tried_ids,
                groq_available,
                groq_detail,
                transactions,
                articles,
                cutoff,
                window_end,
            )
            screened = result["screened"]
            rounds_used = result["rounds_used"]
            exhausted = result["exhausted"]
            n_survivors = result["n_survivors"]
            print(
                f"  fetch-more: rounds_used={rounds_used} exhausted={exhausted} "
                f"n_survivors_after={n_survivors}"
            )

        all_screened.extend(screened)
        summary_rows.append(
            {
                "style_id": style_id,
                "n_candidates_screened": len(screened),
                "n_full_garment_survivors": n_survivors,
                "fetch_rounds_used": rounds_used,
                "fetch_exhausted": exhausted,
            }
        )

    out_df = pl.DataFrame(all_screened).select(_SCREENED_MANIFEST_COLUMNS)
    SCREENED_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_csv(SCREENED_MANIFEST_PATH)
    print(f"\nWrote {SCREENED_MANIFEST_PATH} ({out_df.height} rows)")

    print("\n=== Summary ===")
    for row in summary_rows:
        status = (
            "OK" if row["n_full_garment_survivors"] >= MIN_SURVIVING_BEFORE_FETCH else "SHORTFALL"
        )
        print(
            f"  [{status}] {row['style_id']}: {row['n_full_garment_survivors']} full-garment "
            f"survivor(s) / {row['n_candidates_screened']} screened "
            f"(fetch_rounds_used={row['fetch_rounds_used']}, exhausted={row['fetch_exhausted']})"
        )


if __name__ == "__main__":
    main()
