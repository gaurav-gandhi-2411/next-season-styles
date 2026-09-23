"""Seasonal concept for the chosen season: the existing pipeline, conditioned on Summer.

The brief asks how the top styles OR generated concepts change for a chosen season. The seasonal
top-3 tables already show the styles; this module closes the concept half by running the SAME
pipeline primitives used for the three AW2020 finals on the Summer rank-1 style:

    prepare   pick the style, fetch its top-selling reference candidates (same selector + fetcher
              as `nss.data.select_final_three_exemplars`), and build a contact sheet for screening
    brief     the model's summer-origin forecast + local SHAP drivers -> `StyleProfile` ->
              `skills/style-brief` brief (same generator as the finals)
    generate  `final_concepts.generate_candidates_for_style` at `IP_ADAPTER_SCALE=0.45`, 4 seeds
    score     Gate 1 (p90) + Gate 1b + clone control on the candidates, same code as the finals

WHY A SUMMER-ORIGIN FORECAST: `top_styles_by_season_v2.csv`'s Summer table is a historical seasonal
mean, not a prediction. To give the Summer concept a genuine predicted intensity, the frozen model
config is trained ONLY on origins strictly before 2020-06-01 (exactly the walk-forward fit for the
last backtest origin, whose 13-week window is Jun-Aug 2020) and predicts that origin. Nothing is
saved from this model; no committed model or table is touched.

Usage:
    uv run python -m nss.generate.seasonal_concept prepare
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from nss.data.fetch_images import fetch_images
from nss.data.select_exemplars import (
    ARTICLES_PATH,
    LOOKBACK_WEEKS,
    PANEL_LAST_WEEK,
    compute_lookback_cutoff,
    select_top_selling_articles,
    style_key_values_from_panel,
)
from nss.data.select_final_three_exemplars import TRANSACTIONS_DIR
from nss.features.style_panel import STYLE_KEY_COLS
from nss.generate import build_design_briefs, final_concepts, final_concepts_v2
from nss.models import final_forecast, final_three_shap_verdict
from nss.models.backtest import generate_origin_schedule
from nss.models.lightgbm_model import (
    build_model_frame,
    feature_columns,
    predict_lightgbm,
    train_lightgbm,
)

SEASON = "summer"
SEASONAL_TABLE = Path("reports/tables/top_styles_by_season_v2.csv")
IMAGES_DIR = Path("data/images")
WORK_DIR = Path("data/generated/seasonal_summer")
CANDIDATES_PATH = Path("reports/tables/seasonal_summer_reference_candidates.csv")
N_CANDIDATES = 8
SCREENED_PATH = Path("reports/tables/seasonal_summer_screened.csv")
BRIEF_PATH = Path("reports/tables/seasonal_summer_brief.json")
FORECAST_PATH = Path("reports/tables/seasonal_summer_forecast.csv")
SUMMER_ORIGIN = date(2020, 6, 1)  # last backtest origin: its 13-week window is Jun-Aug 2020
SCREENING_NOTE = (
    "manual visual screening (task K8): full-garment product flat-lay, no pattern drift; VLM "
    "framing judges were quota-blocked (Gemini free tier 20/day, Groq daily token cap)"
)


def season_style(rank: int = 1, season: str = SEASON) -> dict[str, Any]:
    """The `rank`-th style of `season`'s seasonal top-3 table."""
    table = pl.read_csv(SEASONAL_TABLE).filter(pl.col("season") == season).sort("rank")
    return table.filter(pl.col("rank") == rank).to_dicts()[0]


def prepare(rank: int = 1) -> pl.DataFrame:
    """Rank the style's articles by last-26-week sales, fetch the top ones, write a manifest."""
    row = season_style(rank)
    style_key = row["style_key"]
    panel = pl.read_parquet("data/processed/style_week_panel.parquet")
    style_values = style_key_values_from_panel(panel, style_key)
    cutoff = compute_lookback_cutoff(PANEL_LAST_WEEK, LOOKBACK_WEEKS)
    articles = pl.read_csv(ARTICLES_PATH).select(["article_id", *STYLE_KEY_COLS])
    txn = pl.scan_parquet(str(TRANSACTIONS_DIR / "**" / "*.parquet")).select(
        ["article_id", "t_dat"]
    )
    window_end = txn.select(pl.col("t_dat").max()).collect().item()
    transactions = txn.filter(pl.col("t_dat") >= cutoff).collect(engine="streaming")
    ranked = select_top_selling_articles(
        transactions, articles, style_values, cutoff, window_end, n=N_CANDIDATES
    )
    ids = [int(x) for x in ranked["article_id"].to_list()]
    results = fetch_images(ids, IMAGES_DIR)
    manifest = ranked.with_columns(
        pl.lit(style_key).alias("style_id"),
        pl.Series("fetch_success", [bool(results.get(str(i), False)) for i in ids]),
        pl.Series("image_path", [str(IMAGES_DIR / f"{i:010d}.jpg") for i in ids]),
    )
    CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_csv(CANDIDATES_PATH)
    print(
        f"{style_key}: {manifest.height} candidates, fetched {int(manifest['fetch_success'].sum())}"
    )
    return manifest


def write_manual_screening(labels: dict[int, bool]) -> pl.DataFrame:
    """Screened manifest from a human's per-article full-garment verdicts (best-selling first).

    Same ordering contract as `screen_references.load_screened_references`: full-garment survivors
    sorted by units sold descending, so `reference_images[0]` (the only image `local_sdxl`
    conditions on) is the best-selling full-garment shot.
    """
    cand = pl.read_csv(CANDIDATES_PATH).filter(pl.col("fetch_success"))
    rows = [
        {
            "style_id": r["style_id"],
            "article_id": r["article_id"],
            "image_path": r["image_path"],
            "units_sold_last_26w": r["units_sold_last_26w"],
            "is_full_garment": bool(labels[r["article_id"]]),
            "screening_note": SCREENING_NOTE,
        }
        for r in cand.to_dicts()
    ]
    df = pl.DataFrame(rows)
    df.write_csv(SCREENED_PATH)
    return df


def screened_references() -> list[Path]:
    """Full-garment survivors, best-selling first."""
    df = pl.read_csv(SCREENED_PATH).filter(pl.col("is_full_garment"))
    df = df.sort(["units_sold_last_26w", "image_path"], descending=[True, False])
    return [Path(p) for p in df["image_path"].to_list()]


def summer_forecast(style_key: str) -> dict[str, Any]:
    """Frozen-config prediction for `style_key` from the summer origin, plus local SHAP drivers.

    Trains ONLY on origins strictly before `SUMMER_ORIGIN` (the backtest's last walk-forward fit);
    in memory only. Also returns the realised value of the same window for the record.
    """
    import numpy as np

    panel = pl.read_parquet("data/processed/style_week_panel.parquet")
    weeks = [o.origin_week for o in generate_origin_schedule(panel)]
    assert SUMMER_ORIGIN in weeks, "summer origin not in the backtest schedule"
    frame = build_model_frame(panel, weeks)
    cols = feature_columns(frame)
    train = frame.filter(pl.col("origin_week").is_in(weeks[: weeks.index(SUMMER_ORIGIN)]))
    at_origin = frame.filter(pl.col("origin_week") == SUMMER_ORIGIN)
    model = train_lightgbm(train, final_forecast.FINAL_MODEL_CONFIG, cols)
    row = at_origin.filter(pl.col("style_key") == style_key)
    pred = float(np.expm1(predict_lightgbm(model, row, cols))[0])
    ranks = np.expm1(predict_lightgbm(model, at_origin, cols))
    rank_of_style = int((ranks > pred).sum()) + 1
    shap_rows = final_forecast.compute_local_shap_drivers(model, [style_key], at_origin, cols)
    out = {
        "style_key": style_key,
        "origin_week": str(SUMMER_ORIGIN),
        "predicted_intensity": pred,
        "predicted_rank_among_eligible": rank_of_style,
        "n_eligible_styles": at_origin.height,
        "realised_intensity": float(np.expm1(row["y_true"].to_numpy()[0])),
        **shap_rows.to_dicts()[0],
    }
    pl.DataFrame([out]).write_csv(FORECAST_PATH)
    return out


def make_brief(forecast: dict[str, Any], references: list[Path]) -> dict[str, Any]:
    """`style-brief` brief for the seasonal style via the SAME generator the finals used."""
    skill = build_design_briefs._load_skill_module()
    drivers = [
        (forecast[f"shap_driver_{i}_feature"], float(forecast[f"shap_driver_{i}_value"]))
        for i in range(1, 6)
    ]
    style_key = forecast["style_key"]
    values = dict(zip(STYLE_KEY_COLS, style_key.split(" || "), strict=True))
    top_row = {
        "style_key": style_key,
        "product_type_name": values["product_type_name"],
        "garment_group_name": values["garment_group_name"],
        "perceived_colour_master_name": values["perceived_colour_master_name"],
        "graphical_appearance_name": values["graphical_appearance_name"],
        "predicted_intensity": forecast["predicted_intensity"],
        "growth_ratio": None,
    }
    verdict_row = {
        "dominant_mechanism": final_three_shap_verdict._dominant_mechanism(drivers),
        **{k: v for k, v in forecast.items() if k.startswith("shap_driver_")},
    }
    profile = build_design_briefs.build_style_profile(
        top_row, verdict_row, [str(p) for p in references]
    )
    brief = skill.generate_design_brief(profile)
    import json

    BRIEF_PATH.write_text(json.dumps(brief, indent=2), encoding="utf-8")
    return brief


def generate(brief: dict[str, Any], references: list[Path]) -> list[final_concepts.Candidate]:
    """4 seeds at scale 0.45 via the finals' own generation primitive (GPU)."""
    from nss.generate.scale_sweep import free_sdxl_pipeline

    style_id = brief["style_id"]
    prompt, negative = final_concepts.build_generation_spec(style_id, brief)
    prompt_2 = final_concepts.build_prompt_2(style_id)
    print("PROMPT:", prompt, "\nNEGATIVE:", negative, "\nPROMPT_2:", prompt_2)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    cands = final_concepts.generate_candidates_for_style(
        style_id,
        prompt,
        negative,
        references,
        seeds=tuple(final_concepts_v2.INITIAL_SEEDS),
        ip_adapter_scale=final_concepts_v2.IP_ADAPTER_SCALE,
        output_dir=WORK_DIR,
        prompt_2=prompt_2,
        negative_prompt_2=negative,
    )
    free_sdxl_pipeline()
    return cands


SCORED_PATH = Path("reports/tables/seasonal_summer_scored.csv")
JUDGE_LOG = Path("data/generated/judge_repeat_summer.jsonl")
JUDGE_OUT = Path("reports/tables/seasonal_summer_judge.csv")
SELECTED_SEED = 43  # visual inspection: coherent black ribbed high-waist bottom, no pattern drift
VISUAL_QC = {
    42: "reject: grey/white pinstripe panel -- pattern drift, not a plain black structure",
    43: "keep: coherent black ribbed high-waist bottom, folded waistband, picot edge; no drift",
    44: "usable but plain: smooth black bikini bottom, structure texture lost",
    45: "reject: straps attached to a bottom -- malformed garment",
}


def candidate_path(seed: int) -> Path:
    """Where `generate` wrote this seed's image."""
    return WORK_DIR / f"ladieswear_swimwear-bottom_swimwear_black_other-structure_seed{seed}.png"


def score_candidates() -> pl.DataFrame:
    """Gate 1 (p90 range), Gate 1b (nearest reference) and the clone control per candidate.

    Exactly the finals' code: `within_style_benchmark.style_benchmark`/`concept_similarity` for the
    statistics and `SKILL.within_style_novelty_pass` for the verdicts, references = the screened
    full-garment survivors.
    """
    from nss.generate import clip_scoring, dino_scoring
    from nss.generate.gate1b_nearest_reference import gate1b_pass
    from nss.generate.leave_one_out_control import SPACES
    from nss.generate.vlm_judges import SKILL
    from nss.generate.within_style_benchmark import concept_similarity, style_benchmark

    embedders = {"clip": clip_scoring.embed_image, "dinov2": dino_scoring.embed_image}
    refs = screened_references()
    ref_embs = {sp: [fn(p) for p in refs] for sp, fn in embedders.items()}
    bench = {sp: style_benchmark(ref_embs[sp]) for sp in SPACES}
    clone = gate1b_pass({sp: ref_embs[sp][0] for sp in SPACES}, ref_embs)
    rows = []
    for seed in final_concepts_v2.INITIAL_SEEDS:
        emb = {sp: embedders[sp](candidate_path(seed)) for sp in SPACES}
        sims = {sp: concept_similarity(emb[sp], ref_embs[sp])["mean"] for sp in SPACES}
        thr = {sp: bench[sp]["pair_p90"] for sp in SPACES}
        b = gate1b_pass(emb, ref_embs)
        rows.append(
            {
                "seed": seed,
                "visual_qc": VISUAL_QC[seed],
                **{f"{sp}_mean_sim": sims[sp] for sp in SPACES},
                **{f"{sp}_p90_threshold": thr[sp] for sp in SPACES},
                "gate1_pass": SKILL.within_style_novelty_pass(sims, thr),
                **{f"{sp}_max_sim": b[f"{sp}_max_sim"] for sp in SPACES},
                **{f"{sp}_gate1b_threshold": b[f"{sp}_threshold"] for sp in SPACES},
                "gate1b_pass": b["joint_pass"],
                "clone_gate1b_pass": clone["joint_pass"],
            }
        )
    df = pl.DataFrame(rows)
    df.write_csv(SCORED_PATH)
    return df


def judge_selected(n_repeats: int = 3) -> list[float]:
    """Blind Groq attribute-fidelity calls on the selected concept (stops at the first failure).

    Returns the successful mean scores (may be fewer than `n_repeats` if quota runs out -- never
    padded). Raw calls are appended to `JUDGE_LOG`; earlier successes are reused.
    """
    import json

    from nss.generate import judge_rescore, vlm_judges
    from nss.generate.concept_qc_pipeline import parse_style_attributes
    from nss.generate.vlm_judges import SKILL

    style_id = pl.read_csv(SCREENED_PATH)["style_id"][0]
    truth = {d: parse_style_attributes(style_id)[d] for d in vlm_judges.ATTRIBUTE_DIMENSIONS}
    path = candidate_path(SELECTED_SEED)
    scores: list[float] = []
    if JUDGE_LOG.exists():
        for line in JUDGE_LOG.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["available"] and rec["mean_score"] is not None:
                scores.append(rec["mean_score"])
    while len(scores) < n_repeats:
        res = SKILL.run_judge(
            "groq",
            vlm_judges.extract_attributes_groq,
            path,
            vlm_judges.ATTRIBUTE_DIMENSIONS,
            truth,
            "local_sdxl",
        )
        with JUDGE_LOG.open("a", encoding="utf-8") as fh:
            keep = ("available", "raw_extraction", "scores", "mean_score", "excluded_reason")
            fh.write(json.dumps({"image_path": str(path), **{k: res[k] for k in keep}}) + "\n")
        if not (res["available"] and res["mean_score"] is not None):
            print("judge unavailable:", str(res["excluded_reason"])[:200])
            break
        scores.append(res["mean_score"])
    _, _, thresholds = judge_rescore.recompute_calibration()
    print("judge scores:", scores, "threshold:", thresholds["groq"])
    # Committed record of the readings (the raw log lives under git-ignored data/).
    pl.DataFrame(
        {
            "reading": list(range(1, len(scores) + 1)),
            "mean_score": scores,
            "threshold": [thresholds["groq"]] * len(scores),
            "image": [str(path)] * len(scores),
        },
        schema={
            "reading": pl.Int64,
            "mean_score": pl.Float64,
            "threshold": pl.Float64,
            "image": pl.String,
        },
    ).write_csv(JUDGE_OUT)
    return scores


def main() -> None:
    """CLI dispatch."""
    cmd = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    if cmd == "prepare":
        prepare(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    elif cmd == "screen-all-full-garment":
        cand = pl.read_csv(CANDIDATES_PATH).filter(pl.col("fetch_success"))
        write_manual_screening({int(a): True for a in cand["article_id"].to_list()})
    elif cmd == "brief":
        refs = screened_references()
        style_key = pl.read_csv(SCREENED_PATH)["style_id"][0]
        fc = summer_forecast(style_key)
        print({k: v for k, v in fc.items() if not k.startswith("shap")})
        brief = make_brief(fc, refs)
        print(brief["silhouette"], "|", brief["colour_direction"])
    elif cmd == "score":
        import polars as _pl

        with _pl.Config(tbl_cols=-1, tbl_width_chars=240, fmt_str_lengths=30):
            print(score_candidates().drop("visual_qc").with_columns(_pl.col(_pl.Float64).round(4)))
    elif cmd == "judge":
        judge_selected()
    elif cmd == "generate":
        import json

        brief = json.loads(BRIEF_PATH.read_text(encoding="utf-8"))
        generate(brief, screened_references())
    else:
        raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # polars prints Unicode; avoid cp1252 crash
    _ = date  # imported for later phases
    main()
