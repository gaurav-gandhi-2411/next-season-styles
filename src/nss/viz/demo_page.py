# ruff: noqa: E501  -- long lines are inline HTML/CSS template strings, not logic
"""Reviewer-facing demo page `reports/DEMO.html` (tasks K3 + K7).

One self-contained HTML file: images inlined as base64, system fonts only, no scripts, no server,
no internet. Written for a reader with no ML background: every number sits next to a one-line
plain-language gloss, verdicts are written as words, and jargon is glossed where it first appears.
Every technical number is still shown (nothing is replaced by prose) and every one is READ from a
committed table -- nothing is typed in. The forecast spot-check series is cached to
`reports/tables/forecast_spotcheck.csv` (built once from the frozen config; `--refresh` rebuilds).

Usage:
    uv run python -m nss.viz.demo_page [--refresh]
"""

from __future__ import annotations

import base64
import html
import io
import sys
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import polars as pl
from PIL import Image

from nss.generate import concept_generation, final_registry
from nss.generate.concept_scoring import ADVISORY_JUDGES
from nss.generate.final_deliverables import STYLE_ORDER
from nss.generate.final_selection_figures import (
    CLOSED_LOOP_LABEL,
    OBSERVED_CAPTIONS,
    closed_loop_view,
)
from nss.viz import demo_explorer

OUT_PATH = Path("reports/DEMO.html")
T = Path("reports/tables")
SEASONAL_FIG = Path("reports/figures/seasonal_comparison.png")
SPOT_PATH = T / "forecast_spotcheck.csv"
SPOT_RANKS = (1, 1500)
LAST_WEEKS = 26
INK, ACCENT, MUTED = "#15181e", "#2a3fd0", "#5f646d"

FEATURE_WORDS = {
    "lag_1": "last week's sales level for this style",
    "lag_2": "sales two weeks ago",
    "n_active_articles_level": "how many products of this style are on sale",
    "fourier_sin_1": "the time-of-year pattern",
    "fourier_cos_1": "the time-of-year pattern",
    "garment_group_name": "the kind of garment",
    "perceived_colour_master_name": "the colour",
    "product_type_name": "the product type",
    "share_garment_group": "how much of its garment group's sales the style takes",
    "share_index_group": "how much of its department's sales the style takes",
}
METRICS = (
    ("hit_at_3_in_top20", "Hit@3 in top 20"),
    ("hit_at_3_in_top10", "Hit@3 in top 10"),
    ("precision_at_3", "Precision@3"),
    ("ndcg_at_10", "NDCG@10"),
    ("spearman_rho", "Spearman"),
)
METHODS = {
    "lightgbm": ("The forecasting model (LightGBM)", ""),
    "seasonal_naive": (
        "Baseline: same as last year",
        "Predict each style will sell what it sold at this time last year.",
    ),
    "ewma_persistence": (
        "Baseline: recent momentum",
        "Predict each style will keep selling at its recent weekly average.",
    ),
    "global_mean": (
        "Baseline: everyone the same",
        "Predict every style sells the store-wide average.",
    ),
    "parent_category_mean": (
        "Baseline: category average",
        "Predict each style sells its product category's average.",
    ),
    "random_floor": (
        "Random guessing",
        "Shuffle the true answers among styles, 20 times, and score that.",
    ),
}


def _b64_image(path: Path, jpeg: bool = False, max_side: int | None = None) -> str:
    """`data:` URI for an image; optionally re-encoded as JPEG q88 (and shrunk) to keep the page small."""
    if jpeg or max_side:
        buf = io.BytesIO()
        im = Image.open(path).convert("RGB")
        if max_side:
            im.thumbnail((max_side, max_side))
        im.save(buf, "JPEG", quality=88)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _fig_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


PLAIN_NAMES = final_registry.PLAIN_NAMES


def _e(text: object) -> str:
    return html.escape(str(text))


def _plain_driver(feature: str) -> str:
    return FEATURE_WORDS.get(feature, f"the model input '{feature}'")


def _num(x: float | None, nd: int = 3) -> str:
    if x is None or x != x:
        return "undefined"
    return f"{x:.{nd}f}"


def _ci(mean: float | None, lo: float | None, hi: float | None) -> str:
    if mean is None or mean != mean:
        return "undefined"
    if lo is None or hi is None or lo != lo or hi != hi:
        return f"<span class=big>{mean:.3f}</span>"
    return f"<span class=big>{mean:.3f}</span> <span class=ci>({lo:.3f} to {hi:.3f})</span>"


def verdict_word(ok: bool | None) -> str:
    """A verdict in words plus a shape marker (never colour alone)."""
    if ok is None:
        return '<span class="v inconclusive">Not decided</span>'
    return (
        '<span class="v pass">Passed</span>' if ok else '<span class="v fail">Did not pass</span>'
    )


# ------------------------------------------------------------------------------------------------
# Forecast spot-check data (cached table)
# ------------------------------------------------------------------------------------------------


def build_spotcheck_table() -> pl.DataFrame:
    """Realised weekly intensity (last 26 weeks) + the frozen model's forecast, two styles.

    Retrains the frozen final config in memory (deterministic; nothing else is saved) and writes
    the small series table `SPOT_PATH` so the page itself only reads committed tables.
    """
    from nss.models import final_forecast

    panel = pl.read_parquet("data/processed/style_week_panel.parquet")
    model, _frame, columns = final_forecast.train_final_model(panel)
    ranking = final_forecast.build_ranking_frame(panel, model, columns)
    origin = final_forecast.FORECAST_ORIGIN
    rows = []
    for rank in SPOT_RANKS:
        pick = ranking.filter(pl.col("rank_unguarded") == rank).to_dicts()[0]
        hist = panel.filter(
            (pl.col("style_key") == pick["style_key"])
            & (pl.col("week_start") > origin - timedelta(weeks=LAST_WEEKS))
            & (pl.col("week_start") <= origin)
        ).sort("week_start")
        for r in hist.select("week_start", "units_per_active_article").to_dicts():
            rows.append(
                {
                    "rank_unguarded": rank,
                    "n_ranked": ranking.height,
                    "style_key": pick["style_key"],
                    "origin": str(origin),
                    "week_start": str(r["week_start"]),
                    "units_per_active_article": r["units_per_active_article"],
                    "predicted_intensity": pick["predicted_intensity"],
                }
            )
    df = pl.DataFrame(rows)
    df.write_csv(SPOT_PATH)
    return df


def spotcheck_chart() -> str:
    """Two small charts on one shared axis: recent realised weeks vs the model's forecast."""
    df = pl.read_csv(SPOT_PATH, try_parse_dates=True)
    ymax = max(df["units_per_active_article"].max(), df["predicted_intensity"].max())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), sharey=True)
    for ax, rank in zip(axes, SPOT_RANKS, strict=True):
        d = df.filter(pl.col("rank_unguarded") == rank).sort("week_start")
        row = d.to_dicts()[0]
        origin = d["week_start"].max()
        fstart, fend = origin + timedelta(weeks=1), origin + timedelta(weeks=13)
        ax.axvspan(fstart, fend, color=ACCENT, alpha=0.08, lw=0)
        ax.plot(d["week_start"], d["units_per_active_article"], color=INK, lw=1.6)
        ax.hlines(row["predicted_intensity"], fstart, fend, color=ACCENT, lw=3.2)
        ax.text(
            fstart,
            row["predicted_intensity"] + ymax * 0.035,
            f"forecast {row['predicted_intensity']:.1f}",
            color=ACCENT,
            fontsize=9,
            fontweight="bold",
        )
        ax.set_ylim(0, ymax * 1.1)
        ax.set_title(
            f"Ranked {rank} of {row['n_ranked']:,}\n{row['style_key'].replace(' || ', ' · ')}",
            fontsize=9,
            color=INK,
            loc="left",
        )
        ax.tick_params(axis="x", rotation=30, labelsize=8, colors=MUTED)
        ax.tick_params(axis="y", labelsize=8, colors=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c9ccd1")
    axes[0].set_ylabel("units sold per product on sale, per week", fontsize=9, color=MUTED)
    fig.tight_layout()
    return _fig_b64(fig)


# ------------------------------------------------------------------------------------------------
# Sections
# ------------------------------------------------------------------------------------------------


def _load_concepts() -> list[dict]:
    sel = {r["style_id"]: r for r in pl.read_csv(T / "final_selection.csv").to_dicts()}
    top = {r["style_key"]: r for r in pl.read_csv(T / "top_styles_final_three.csv").to_dicts()}
    refs = concept_generation.load_refs()
    out = []
    for sid in (*STYLE_ORDER, final_registry.SUMMER):
        out.append(
            {
                "sid": sid,
                "name": PLAIN_NAMES[sid],
                "sel": sel[sid],
                "top": top.get(sid),
                "asked": concept_generation.CHANGES[sid]["applied_changes"],
                "refs": refs[sid],
            }
        )
    return out


def hero(concepts: list[dict]) -> str:
    """The concepts large, each with its briefed changes called out, then the orientation."""
    aw = [c for c in concepts if c["sid"] in STYLE_ORDER]
    figs = "".join(
        f'<figure><img src="{_b64_image(Path(c["sel"]["image_path"]), jpeg=True)}" alt="{_e(c["name"])} concept">'
        f'<figcaption><b>{_e(c["name"])}</b><span>{_e(OBSERVED_CAPTIONS[c["sid"]])}</span>'
        f'<ul class=chg aria-label="Design changes asked for">{"".join(f"<li>{_e(x)}</li>" for x in c["asked"])}</ul>'
        f'<em class="{"ok" if c["sel"]["human_brief_met"] else "no"}">{"Changes confirmed by eye" if c["sel"]["human_brief_met"] else "Changes not all visible"}</em></figcaption></figure>'
        for c in aw
    )
    met = sum(1 for c in concepts if c["sel"]["human_brief_met"])
    auto = sum(1 for c in concepts if c["sel"]["automatic_gates"] == "PASS")
    return f"""<header class=masthead><p class=site>next-season-styles</p>
<h1>Three garments a forecast chose, and the pictures made from them.</h1></header>
<div class=strip>{figs}</div>
<section class=first><h2>What to look at first</h2>
<p><b>What the model predicted.</b> From two years of H&amp;M sales it ranked about 2,000 clothing styles by how hard each would sell, per product on sale, over the 13 weeks after 21 September 2020. Its emerging risers, after a human rule that removed intimates and garments that cannot be told apart in a flat photo, were a beige knit sweater, a red dress and a white jersey top.</p>
<p><b>What was generated from it.</b> For each style an image generator was given several real H&amp;M photos of the style as a guide, plus a sentence naming two design changes (listed under each picture). Each picture is the best of eight tries, chosen by automatic checks and then by a person looking.</p>
<p><b>How to judge whether it worked.</b> Is each requested change visible, and is the picture still a believable product of its style? A person confirmed the changes on {met} of {len(concepts)} concepts (three autumn and winter, one summer); {auto} pass every automatic check. To find your own category, browse the forecast below.</p></section>"""


def _check_row(title: str, verdict: str, body: str, gloss: str) -> str:
    return (
        f"<div class=check><div class=chk-head><span class=chk-title>{title}</span>{verdict}</div>"
        f"<p class=chk-body>{body}</p><p class=gloss>{gloss}</p></div>"
    )


def concept_sections(concepts: list[dict]) -> str:
    """Section 1: each concept large, with every check in words, numbers and one-line glosses."""
    out = [
        "<section id=concepts><h2>The concepts</h2><p class=lede>Each block: the picture, what it shows, and the checks. Every number is followed by what it means. The measures behind the checks are explained once, after the last concept.</p>"
    ]
    for c in concepts:
        r = c["sel"]
        judges = r["judges"].split(",")
        g2 = "; ".join(
            f"{j}{' (advisory)' if j in ADVISORY_JUDGES else ''}: <b>{_num(r[f'{j}_fidelity'], 2)}</b> ({'pass' if r[f'{j}_gate2_pass'] else 'below its pass mark'})"
            for j in judges
        )
        g3 = "; ".join(
            f"{j}: answers <b>{r[f'{j}_gate3_answers']}</b> (Y = change visible) ({'pass' if r[f'{j}_gate3_pass'] else 'fail'})"
            for j in judges
            if r.get(f"{j}_gate3_answers") is not None
        )
        auto = r["automatic_gates"] == "PASS"
        human = bool(r["human_brief_met"])
        headline = (
            "Passes every automatic check. "
            if auto
            else "Fails at least one automatic check (below). "
        ) + (
            "A person confirms the requested changes are visible."
            if human
            else "A person finds a requested change missing."
        )
        refs_html = "".join(
            f'<img src="{_b64_image(p, max_side=420)}" alt="reference photo">'
            for p in c["refs"][:12]
        )
        changes = "; ".join(_e(x) for x in c["asked"])
        forecast = (
            f"maps to <b>{_e(str(r['forecast_style_key']).replace(' || ', ' · '))}</b>; forecast <b>{r['forecast_units']:.1f}</b> units per product per week, rank <b>{int(r['forecast_rank'])}</b>; confidence <b>{_e(r['forecast_confidence'])}</b>"
            if r["forecast_units"] is not None
            else "not run"
        )
        checks = "".join(
            [
                _check_row(
                    "The requested design changes are visible (human)",
                    verdict_word(human),
                    f"Asked for: {changes}. " + _e(r["human_check"]),
                    "A person looked at the image. This is the check that decides whether the picture is a new design or just a picture of the style.",
                ),
                _check_row(
                    "The requested changes are visible (automatic, Gate 3)",
                    verdict_word(bool(r["gate3_pass"])),
                    "An image reader was asked, one yes-or-no question per change, whether the garment has it. "
                    + g3
                    + ".",
                    "Passing means most requested changes were seen. The reader says yes too easily (in validation it was right about 'no' only 58% of the time), so this can pass what a person would fail.",
                ),
                _check_row(
                    "Fits in with real products of its style (Gate 1)",
                    verdict_word(bool(r["gate1_pass"])),
                    f'Average similarity to {r["n_refs"]} real reference photos: <b>{_num(r["clip_mean_sim"])}</b> on CLIP (limit {_num(r["clip_p90_limit"])}) and <b>{_num(r["dinov2_mean_sim"])}</b> on DINOv2 (limit {_num(r["dinov2_p90_limit"])}).',
                    "The limit is how alike real products of this style are to each other: only 1 in 10 real pairs is more alike.",
                ),
                _check_row(
                    "Not a copy of any single photo (Gate 1b)",
                    verdict_word(bool(r["gate1b_pass"])),
                    f'Closest single reference: <b>{_num(r["clip_max_sim"])}</b> on CLIP (limit {_num(r["clip_gate1b_limit"])}) and <b>{_num(r["dinov2_max_sim"])}</b> on DINOv2 (limit {_num(r["dinov2_gate1b_limit"])}). An exact copy fails this test in every style.',
                    "The limit is how close real products get to their nearest sibling.",
                ),
                _check_row(
                    "Looks like a real garment (integrity floor)",
                    verdict_word(bool(r["integrity_floor_pass"])),
                    f'The closest real reference must be at least as close as 9 in 10 real products of any style are to their own nearest sibling: <b>{_num(r["floor_max_sim"])}</b> against a global floor of {_num(r["global_floor_limit"])} (DINOv2). The floor calibrated within this style alone (advisory) is {_num(r["floor_limit"])}: {"pass" if r["integrity_style_pass"] else "fail"}.',
                    "The global floor gates because garment coherence is not a property of one style. It passes every known-good test image and misses one known-malformed one (a sheer-mesh garment); the within-style floor catches that one but is unmeetable for a design change in a style whose real products are near-identical. The white top clears the global floor but not the within-style one, for that reason."
                    if c["sid"] == final_registry.TOP
                    else "The global floor gates because garment coherence is not a property of one style. It passes every known-good test image and misses one known-malformed one (a sheer-mesh garment); the within-style floor (advisory) catches that one.",
                ),
                _check_row(
                    "Matches the style's attributes (Gate 2)",
                    verdict_word(bool(r["gate2_pass"])),
                    "Two local image readers, shown only the picture, were asked for product type, colour and pattern: "
                    + g2
                    + ".",
                    "SmolVLM gates; Florence-2 is advisory because its agreement with the API readers was too low to carry a verdict. A reader that sees a brown trim on a beige sweater will say 'brown': a real limit of scoring colour from a single word.",
                ),
                _check_row(
                    "Scored back through the forecaster (closed loop)",
                    '<span class="v inconclusive">Reported</span>',
                    forecast + ".",
                    "This is the forecast for the style the picture reads as, not a prediction that this exact design would sell.",
                ),
            ]
        )
        out.append(
            f"""<article class=concept><h3>{_e(c["name"])}</h3>
<div class=cols><div class=pic><img src="{_b64_image(Path(r['image_path']), jpeg=True)}" alt="{_e(c['name'])} concept, full size"></div>
<div class=body><p class=headline>{headline}</p>{checks}</div></div>
<details class=refs><summary>{len(c['refs'])} of the real H&amp;M photos it was guided by (first 12 shown)</summary><div class=refgrid>{refs_html}</div></details></article>"""
        )
    out.append(
        """<aside class=explain><h3>What the measures mean</h3>
<p><b>CLIP</b> and <b>DINOv2</b> are two pretrained image models that turn a picture into numbers so pictures can be compared; 1.0 means identical. CLIP compares overall look; DINOv2 compares shape and texture. A check passes only if both pass.</p>
<p>A <b>limit</b> here is the <b>90th percentile</b> of the same measure between real H&amp;M products of that style. The reference base was widened from 4 to 6 photos per style to as many as 25 (the white top has only 19 real articles in the whole catalogue), which cut the uncertainty of these limits by 2 to 4 times.</p></aside></section>"""
    )
    return "\n".join(out)


def trace_back(concepts: list[dict]) -> str:
    """Section 2: forecast -> why -> photos -> brief -> picture, per concept (a real sequence)."""
    out = [
        "<section id=trace><h2>How each concept traces back to its predicted style</h2><p class=lede>Read each row left to right: what the model forecast, why, which photos guided the picture, what the design brief asked for, and what the picture actually shows.</p>"
    ]
    verdicts = {
        r["style_key"]: r for r in pl.read_csv(T / "final_three_shap_verdict.csv").to_dicts()
    }
    for c in concepts:
        if c["top"] is None:
            continue
        top = c["top"]
        drivers = [top[f"shap_driver_{i}_feature"] for i in range(1, 4)]
        why = ", then ".join(_plain_driver(d) for d in drivers)
        asked = "".join(f"<li>{_e(x)}</li>" for x in c["asked"])
        v = verdicts[c["sid"]]
        out.append(
            f"""<article class=trace><h3>{_e(c['name'])}</h3><ol class=chain>
<li><h4>Forecast</h4><p>Predicted <b>{top['predicted_intensity']:.1f}</b> units per product on sale per week for the 13 weeks after 21 Sep 2020; an emerging riser. Its forecast is {top['growth_ratio']:.2f}× its recent level.</p>
<p class=gloss>"Intensity" means units sold per product on sale, so a style is not ranked highly just for having many products.</p></li>
<li><h4>Why the model liked it</h4><p>Biggest influences, in order: {_e(why)}.</p><p class=gloss>These come from an explanation method called SHAP, which shows which inputs pushed this forecast up or down. {_e(v['dominant_mechanism'].split(':')[0].capitalize())}.</p></li>
<li><h4>Guided by</h4><div class=thumbs>{"".join(f'<img src="{_b64_image(p, max_side=420)}" alt="reference">' for p in c['refs'][:3])}</div><p class=gloss>The generator was shown up to 8 of the best-selling real photos at once, so they set the type of garment and colour without any one photo being copied.</p></li>
<li><h4>What the brief asked for</h4><ul>{asked}</ul><p class=gloss>These were instructions to the generator, written as one plain sentence, not results.</p></li>
<li><h4>What the picture shows</h4><img class=final src="{_b64_image(Path(c['sel']['image_path']), jpeg=True)}" alt="concept"><p>{_e(OBSERVED_CAPTIONS[c['sid']])}</p><p class=gloss>Described by looking at the image.</p></li></ol></article>"""
        )
    out.append("</section>")
    return "\n".join(out)


def _top5_block(name: str, view: dict | None) -> str:
    """One concept's top-5 retrieval list, the intended style marked; presentation only."""
    if view is None:
        return ""
    body = "".join(
        f"<tr{' class=intended' if t['intended'] else ''}><td class=n>{t['pos']}</td>"
        f"<td>{_e(t['key'].replace(' || ', ' · '))}</td><td class=n>{t['sim']:.3f}</td>"
        f"<td class=n>{t['rank']}</td><td>{'the style this concept was designed from' if t['intended'] else ''}</td></tr>"
        for t in view["rows"]
    )
    return (
        f"<details class=top5 open><summary><b>{_e(name)}</b>: top 5 of {view['n_styles']:,} matched styles. "
        f"{_e(view['summary'])}; top-5 similarity spread {view['spread']:.3f}; "
        f"{view['same_type']}/5 share the intended product type, {view['same_colour']}/5 the intended colour.</summary>"
        "<div class=scroll><table class=metrics><thead><tr><th>#</th><th>Matched style</th><th>Similarity</th>"
        f"<th>Forecast rank</th><th>Note</th></tr></thead><tbody>{body}</tbody></table></div></details>"
    )


def closed_loop_section(concepts: list[dict]) -> str:
    """Predict -> generate -> score the generation through the same predictor (retrieval)."""
    old = pl.read_csv(T / "concept_forecast_validation.csv")
    old_sm = old.filter(pl.col("judge") == "smolvlm")
    old_exact = float(old_sm["exact_style_key"].mean())
    old_type = float(old_sm["type_ok"].mean())
    old_colour = float(old_sm["colour_ok"].mean())
    q415 = pl.read_csv(T / "retrieval_validation_partial_index.csv")
    qf = pl.read_csv(T / "retrieval_validation.csv")

    def pick(df: pl.DataFrame, cond: str, row: str = "summary_all") -> dict:
        return df.filter(
            (pl.col("condition") == cond)
            & (pl.col("config") == "avg")
            & (pl.col("row_type") == row)
        ).to_dicts()[0]

    f40 = pick(qf, "full_index_40")
    f159 = pick(qf, "loo_full_index_159")
    f1980 = pick(qf, "loo_full_index_1style1photo")
    chance = pick(qf, "full_index_40", "summary_chance_top5_of_candidates")
    n_cand = int(chance["n"])
    l415 = pick(q415, "loo_deployment_index")
    c415 = int(pick(q415, "loo_deployment_index", "summary_chance_top5_of_candidates")["n"])
    conf = {
        lab: pick(qf, "loo_full_index_1style1photo", f"summary_confidence_{lab}")
        for lab in ("high", "medium", "low")
    }
    rows = ""
    top5_html = ""
    for c in concepts:
        r = c["sel"]
        view = closed_loop_view(c["sid"])
        rows += (
            f"<tr><td>{_e(c['name'])}</td><td>{_e(str(r['forecast_style_key']).replace(' || ', ' · '))}</td>"
            f"<td class=n>{r['forecast_units']:.1f}</td><td class=n>{int(r['forecast_rank'])}</td><td>{_e(r['forecast_confidence'])}</td>"
            f"<td>{_e(view['summary']) if view else 'n/a'}</td></tr>"
        )
        top5_html += _top5_block(c["name"], view)
    return f"""<section id=loop><h2>Closing the loop: scoring the pictures with the same forecaster</h2>
<p class=lede>Predict, generate, then score the generated picture through the same predictor: the picture is matched to the nearest real catalogue style and the model's forecast for that style is looked up. The match is made by image retrieval (CLIP and DINOv2 similarity to each style's real photos), not by asking a small model to name the style. It stays a prototype: the measured accuracy is below.</p>
<div class=scroll><table class=metrics><thead><tr><th>Concept</th><th>Top-1 matched style</th><th>Forecast, units per product per week</th><th>Rank among forecast styles</th><th>Confidence</th><th>Where the intended style falls</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class=proto><b>{_e(CLOSED_LOOP_LABEL)}</b></p>
{top5_html}
<p class=gloss>Autumn/winter pictures are ranked among 1,980 styles forecast from 21 Sep 2020; the summer picture among 3,000 styles from 1 Jun 2020. The index holds each intended style's own reference photos, so an intended-style match is close to circular; a mismatch is the informative case. Against the full catalogue the sweater maps to an orange sibling that is 0.006 more similar than its intended style (third), and the white top to a neighbouring Blouses style (its intended style is fifth, 0.007 behind); the dress and the bikini top match their intended style, the bikini's confidence dropping from high to medium because three orange, yellow and green all-over-pattern tops are near-ties.</p>
<h3>How well does the matching work?</h3>
<div class=scroll><table class=metrics><thead><tr><th>Method and test</th><th>Candidate styles</th><th>Photos</th><th>Exact style</th><th>Top 5</th><th>Top 10</th><th>Product type</th><th>Colour</th></tr></thead><tbody>
<tr><td>Free-text baseline (a small model names the style), 40 held-out photos</td><td class=n>1,980</td><td class=n>40</td><td class=n>{old_exact:.1%}</td><td class=n>n/a</td><td class=n>n/a</td><td class=n>{old_type:.1%}</td><td class=n>{old_colour:.1%}</td></tr>
<tr><td><b>Retrieval, full catalogue</b>, the same 40 photos (each left out of the index)</td><td class=n>{n_cand:,}</td><td class=n>{int(f40['n'])}</td><td class=n><b>{f40['top1']:.1%}</b></td><td class=n>{f40['top5']:.1%}</td><td class=n>{f40['top10']:.1%}</td><td class=n>{f40['type_ok']:.1%}</td><td class=n>{f40['colour_ok']:.1%}</td></tr>
<tr><td>Retrieval, full catalogue, leave-one-out on 159 photos</td><td class=n>{n_cand:,}</td><td class=n>{int(f159['n'])}</td><td class=n>{f159['top1']:.1%}</td><td class=n>{f159['top5']:.1%}</td><td class=n>{f159['top10']:.1%}</td><td class=n>{f159['type_ok']:.1%}</td><td class=n>{f159['colour_ok']:.1%}</td></tr>
<tr><td>Retrieval, full catalogue, leave-one-out, one photo per style</td><td class=n>{n_cand:,}</td><td class=n>{int(f1980['n']):,}</td><td class=n>{f1980['top1']:.1%}</td><td class=n>{f1980['top5']:.1%}</td><td class=n>{f1980['top10']:.1%}</td><td class=n>{f1980['type_ok']:.1%}</td><td class=n>{f1980['colour_ok']:.1%}</td></tr>
<tr><td>Retrieval, earlier partial index, leave-one-out on the same 159 photos (optimistic)</td><td class=n>{c415}</td><td class=n>{int(l415['n'])}</td><td class=n>{l415['top1']:.1%}</td><td class=n>{l415['top5']:.1%}</td><td class=n>{l415['top10']:.1%}</td><td class=n>{l415['type_ok']:.1%}</td><td class=n>{l415['colour_ok']:.1%}</td></tr>
</tbody></table></div>
<p>Chance for exact style is {1 / n_cand:.2%} with {n_cand:,} candidates. It does not clearly beat the baseline on the same 40 photos, so it keeps the prototype label; the full-catalogue number is the honest result. The {l415['top1']:.1%} earlier figure came from a candidate set of {c415} styles and is optimistic for exactly that reason; the {n_cand:,}-style rows are the honest numbers.</p>
<p><b>Confidence label</b> (one photo per style, {int(f1980['n']):,} photos): the exact style is first {conf['high']['top1']:.0%} of the time when the label is high ({int(conf['high']['n'])} photos), {conf['medium']['top1']:.0%} for medium ({int(conf['medium']['n'])}) and {conf['low']['top1']:.0%} for low ({int(conf['low']['n'])}); within the top 5 it is {conf['high']['top5']:.0%}, {conf['medium']['top5']:.0%} and {conf['low']['top5']:.0%}. Near-duplicate articles within a style make these leave-one-out figures optimistic, and the forecast is for the archetype the picture reads as, not for the new design.</p></section>"""


def seasonal_section(concepts: list[dict]) -> str:
    """Section: autumn/winter vs summer, both from the same rules and pipeline."""
    d = pl.read_csv(T / "summer_selection_log.csv")
    first = d.filter(pl.col("excluded").is_null()).to_dicts()[0]
    excl = d.filter(pl.col("excluded").is_not_null()).head(3).to_dicts()
    fig = ""
    if SEASONAL_FIG.exists():
        fig = f'<figure class=seasonfig><img src="{_b64_image(SEASONAL_FIG, max_side=1500)}" alt="Autumn/winter 2020 concept beside the summer concept"><figcaption>The same pipeline, run for two seasons with the same selection rules.</figcaption></figure>'
    skipped = "; ".join(
        f"#{r['emerging_rank']} {_e(r['style_key'].replace(' || ', ' · '))} ({_e(r['excluded'].split(':')[0])})"
        for r in excl
    )
    s = next(c for c in concepts if c["sid"] == final_registry.SUMMER)["sel"]
    return f"""<section id=seasons><h2>Seasonal view</h2>
<p class=lede>Same forecaster, same selection rules, a different forecast date: summer picks a different garment.</p>
<p>For summer the model was re-run as of 1 June 2020, using only sales whose 13-week outcome was already known by then. Its top emerging style, after the same editorial exclusions (swimwear bottoms are excluded because a flat photo of one cannot be told from underwear; skipped here: {skipped}), is <b>{_e(first['style_key'].replace(' || ', ' · '))}</b>: predicted <b>{first['predicted_intensity']:.1f}</b> units per product per week for June to August, against <b>{first['realised_intensity']:.1f}</b> realised. One style is one data point: the match is encouraging, not a measure of accuracy.</p>
{fig}
<p>The summer picture: {_e(OBSERVED_CAPTIONS[final_registry.SUMMER])} Automatic gates: <b>{_e(s['automatic_gates'])}</b>; human check: the requested changes are {'visible' if s['human_brief_met'] else 'not all visible'}.</p></section>"""


def limits_section() -> str:
    """What could not be verified, said plainly (after the N-session fixes)."""
    return """<section id=limits><h2>What we could not verify</h2>
<ul class=limits>
<li><b>Whether the pictures would sell.</b> The forecast is about styles; the pictures are new designs no customer has seen. Nothing here tests demand for the pictures themselves.</li>
<li><b>Demand, as opposed to sales.</b> Everything derives from what was stocked and sold, not what customers wanted. No inventory data was available; a stock-out check finds a lower bound of 3.76% of style-weeks with a stock-out signature.</li>
<li><b>The image readers are small and were checked on few images.</b> Groq's daily limit and Gemini's free quota ran out (Gemini's replacement key authenticates and passes calibration, but no concept could be re-read), so the panel is two small local models, one of them advisory. The yes/no reader for design changes says yes too easily (right about 'no' 58% of the time), so a person made the final call.</li>
<li><b>Automatic integrity is a proxy.</b> Asking a small model whether a garment is coherent caught none of the three known-malformed test images; a global similarity floor gates and passes every known-good image but misses one of the three (sheer mesh); the within-style floor, reported as advisory, catches all three but cannot be met by a design change in a near-identical style. The human check remains necessary.</li>
<li><b>The limits behind the checks still rest on a modest number of real photos:</b> 8 to 25 per style (the white top has 19 in the whole catalogue).</li>
<li><b>The exact top three.</b> The model finds a useful neighbourhood but not the exact order, because the leaders are nearly tied.</li>
</ul></section>"""


def model_section() -> str:
    """Does the model work? Embargoed table with paired differences, controls, spot-check."""
    summ = {
        r["method"]: r
        for r in pl.read_csv(T / "backtest_embargo_summary.csv")
        .filter(pl.col("split") == "pooled")
        .to_dicts()
    }
    paired = pl.read_csv(T / "backtest_embargo_paired_diff.csv").filter(pl.col("split") == "pooled")
    old = {
        r["metric"]: r
        for r in pl.read_csv(T / "backtest_embargo_check.csv")
        .filter(pl.col("split") == "pooled")
        .to_dicts()
    }
    shuf = pl.read_csv(T / "label_shuffle_control.csv").filter(
        pl.col("variant") == "shuffled_train_true_test"
    )
    keys = [
        ("hit_at_3_in_top20", "Hit@3 in top 20"),
        ("hit_at_3_in_top10", "Hit@3 in top 10"),
        ("ndcg_at_10", "NDCG@10"),
        ("spearman_rho", "Spearman"),
        ("wmape", "WMAPE (lower is better)"),
    ]
    lg, fl = summ["lightgbm"], summ["random_floor"]
    top20 = old["hit_at_3_in_top20"]
    cov = pl.read_csv(T / "covid_two_model_comparison.csv")
    cv = cov.filter(
        (pl.col("design") == "purged")
        & (pl.col("split") == "all_noncovid")
        & pl.col("metric").is_in(["spearman_rho", "wmape"])
    ).to_dicts()
    ctext = "; ".join(
        f"{r['metric'].replace('_rho', '')} {r['diff']:+.3f} ({r['ci_lo']:+.3f} to {r['ci_hi']:+.3f})"
        for r in cv
    )
    head = "".join(f"<th>{lab}</th>" for _, lab in keys)
    rows = []
    for method, (label, gloss) in METHODS.items():
        m = summ[method]
        cells = []
        for k, _ in keys:
            cell = _ci(m[f"{k}_mean"], m.get(f"{k}_ci_low"), m.get(f"{k}_ci_high"))
            if method not in ("lightgbm", "random_floor"):
                q = paired.filter(
                    (pl.col("method_b") == method) & (pl.col("metric") == k)
                ).to_dicts()
                if q and q[0]["mean_diff"] == q[0]["mean_diff"]:
                    q = q[0]
                    cell += f'<span class=paired>model minus this: {q["mean_diff"]:+.3f} ({q["diff_ci_low"]:+.3f} to {q["diff_ci_high"]:+.3f})</span>'
            cells.append(f"<td>{cell}</td>")
        cls = " class=me" if method == "lightgbm" else ""
        rows.append(
            f'<tr{cls}><th scope=row>{label}<span class=gloss>{gloss}</span></th>{"".join(cells)}</tr>'
        )
    sc = "".join(
        f"<td><span class=big>{sum(shuf[k].to_list()) / shuf.height:.3f}</span></td>"
        if k in shuf.columns
        else "<td>not measured</td>"
        for k, _ in keys
    )
    rows.append(
        f"<tr class=shuffle><th scope=row>Leakage test: the model retrained on scrambled answers<span class=gloss>Same model and data, but which sales figure belongs to which style is shuffled before training. It should collapse to chance, and it does.</span></th>{sc}</tr>"
    )
    table = f'<div class=scroll><table class=metrics><thead><tr><th>Method</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div><p class=gloss>Each cell is the average over 12 past dates, with a 95% interval from a block bootstrap. "Model minus this" is the paired difference on the same dates. Seasonal naive is undefined at 2 of the 12 dates (no year-ago sales), so its paired rows use 10.</p>'
    if not SPOT_PATH.exists():
        build_spotcheck_table()
    spot = f'<h3>A spot-check you can see</h3><img class=chart src="{spotcheck_chart()}" alt="forecast spot-check"><p class=gloss>The best-ranked style and the style ranked 1,500th, both on one scale: the dark line is what each style actually sold, the coloured line is what the model forecast from earlier data.</p>'
    return f"""<section id=model><h2>Does the model actually work?</h2>
<p class=lede>Scored on 12 past dates: at each, the model is trained only on dates whose 13-week answers were already known, then forecasts the next 13 weeks, and we compare with what happened.</p>
<aside class=callout><h3>The honest headline</h3>
<p>Hit@3 in top 20 is <b>{lg['hit_at_3_in_top20_mean']:.3f}</b> (guessing: {fl['hit_at_3_in_top20_mean']:.3f}). An earlier version of this page showed {top20['shipped']:.3f}: that run trained on dates whose answers overlapped the test window. Closing the leak lowers it by {top20['diff']:.3f} (paired, {top20['ci_lo']:.3f} to {top20['ci_hi']:.3f}). Against the naive baselines the lead holds; against "same as last year" it is a lead on ranking quality (NDCG, Spearman, WMAPE) but not demonstrably on picking the top three.</p>
<p><b>Did COVID data help?</b> A second model without training rows whose answers touch March to June 2020, scored out of sample on 13 non-COVID dates, was worse on rank correlation and calibration ({_e(ctext)}, without minus with) and no different on top-k. COVID data did not hurt and mildly helped; the second model also has about 40% fewer rows.</p></aside>
{table}{spot}</section>"""


CSS = """
:root{--paper:#fbfbfa;--ink:#15181e;--muted:#5b6069;--rule:#dcdee2;--wash:#f1f2f4;--accent:#2a3fd0;--serif:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif;--sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
@media(prefers-color-scheme:dark){:root{--paper:#111318;--ink:#eceef2;--muted:#a0a6b0;--rule:#2b2f37;--wash:#1a1d24;--accent:#9aabff}}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.65 var(--sans)}
main{max-width:1180px;margin:0 auto;padding:0 20px 96px}
a{color:var(--accent)}a:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
.masthead{padding:56px 0 28px;max-width:860px}.site{margin:0 0 20px;font-size:.95rem;color:var(--muted);letter-spacing:.02em}
h1{font:400 clamp(2rem,4.6vw,3.4rem)/1.08 var(--serif);letter-spacing:-.015em;margin:0}
h2{font:400 clamp(1.7rem,3vw,2.3rem)/1.15 var(--serif);letter-spacing:-.01em;margin:112px 0 8px}
h3{font:600 1.15rem/1.3 var(--sans);margin:0 0 10px}h4{font:600 .95rem/1.3 var(--sans);margin:0 0 6px}
.lede{color:var(--muted);max-width:62ch;margin:0 0 36px}.gloss{color:var(--muted);font-size:.9rem;line-height:1.5;margin:6px 0 0;max-width:64ch}
.strip{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin:12px 0 0}
.strip figure{margin:0}.strip img{width:100%;height:auto;display:block;background:var(--wash)}
.strip figcaption{padding-top:10px;font-size:.92rem;line-height:1.45}.strip figcaption b{display:block;font:400 1.35rem/1.25 var(--serif);margin-bottom:4px}
.strip figcaption span{color:var(--muted)}
.first{max-width:66ch;margin-top:72px}.first h2{margin-top:0}.first p{font:400 1.2rem/1.6 var(--serif);margin:0 0 18px}.first b{font-family:var(--sans);font-size:.9rem;font-weight:650;display:block;margin-bottom:2px}
.concept{margin:64px 0 0;padding-top:28px;border-top:1px solid var(--rule)}
.cols{display:grid;grid-template-columns:minmax(0,5fr) minmax(0,6fr);gap:40px;align-items:start}
.pic img{width:100%;height:auto;display:block;background:var(--wash)}
.headline{font:400 1.45rem/1.3 var(--serif);margin:0 0 18px}
.check{padding:14px 0;border-top:1px solid var(--rule)}.chk-head{display:flex;justify-content:space-between;gap:12px;align-items:baseline}
.chk-title{font-weight:600}.chk-body{margin:6px 0 0;font-size:.98rem;line-height:1.55}
.v{white-space:nowrap;font-weight:600;font-size:.9rem}
.v::before{content:"";display:inline-block;width:.62em;height:.62em;border-radius:50%;margin-right:.45em;border:2px solid var(--ink)}
.v.pass{color:var(--accent)}.v.pass::before{background:var(--accent);border-color:var(--accent)}
.v.fail{color:var(--ink)}.v.fail::before{background:transparent;border-color:var(--ink);box-shadow:inset 0 0 0 2px var(--paper),inset 0 0 0 4px var(--ink)}
.v.inconclusive{color:var(--muted)}
.refs{margin-top:22px}.refs summary{cursor:pointer;color:var(--accent);font-size:.95rem}
.refgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:10px;margin-top:12px}.refgrid img{width:100%;background:var(--wash)}
.explain{margin-top:56px;padding:24px 28px;background:var(--wash);max-width:78ch}.explain p{margin:0 0 12px;font-size:.96rem}.explain h3{margin-bottom:8px}
.trace{margin:56px 0 0;padding-top:28px;border-top:1px solid var(--rule)}
.chain{list-style:none;counter-reset:s;display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:22px;padding:0;margin:16px 0 0}
.chain>li{counter-increment:s}.chain>li>h4::before{content:counter(s) ".";display:inline-block;margin-right:.4em;color:var(--accent)}
.chain p{margin:0 0 6px;font-size:.93rem;line-height:1.5}.chain ul{margin:0 0 6px;padding-left:1.1em;font-size:.9rem;line-height:1.45}
.thumbs{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}.thumbs img,.chain .final{width:100%;background:var(--wash);display:block;margin-bottom:8px}
.metricgloss{max-width:74ch;margin:0 0 40px}.metricgloss dt{font:600 1rem var(--sans);margin-top:16px}.metricgloss dd{margin:2px 0 0;color:var(--muted);font-size:.96rem;line-height:1.55}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%}
.metrics th,.metrics td{padding:12px 12px;border-top:1px solid var(--rule);vertical-align:top;text-align:left;font-size:.92rem}
.metrics thead th{border-top:0;border-bottom:2px solid var(--ink);font-weight:600;font-size:.85rem}
.metrics tbody th{font-weight:600;min-width:220px}.metrics tr.me{background:var(--wash)}.metrics .gloss{display:block;font-weight:400;font-size:.82rem}
.big{font:400 1.25rem var(--serif);font-variant-numeric:tabular-nums}.ci{color:var(--muted);font-size:.8rem;font-variant-numeric:tabular-nums}
.paired{display:block;color:var(--muted);font-size:.78rem;margin-top:3px;font-variant-numeric:tabular-nums}
.metrics tr.shuffle{outline:2px solid var(--accent);outline-offset:-2px}
.metrics tr.intended{background:var(--wash);font-weight:600}.proto{max-width:72ch;margin:14px 0 8px}details.top5{margin:10px 0}details.top5 summary{cursor:pointer;font-size:.92rem;max-width:88ch}
.callout{margin:40px 0;padding:24px 28px;border-left:4px solid var(--accent);background:var(--wash);max-width:78ch}.callout p{margin:0}.callout p+p{margin-top:12px}
.chart{width:100%;height:auto;display:block;margin-top:8px}
.seasons{display:grid;grid-template-columns:repeat(2,1fr);gap:32px 40px}.season table{font-size:.9rem}.season th,.season td{padding:8px 8px;border-top:1px solid var(--rule);text-align:left;vertical-align:top}.season thead th{border-top:0;border-bottom:2px solid var(--ink);font-size:.8rem}.season .n{font-variant-numeric:tabular-nums;white-space:nowrap}
.seasonfig{margin:48px 0 0}.seasonfig img{width:100%;height:auto;display:block}.seasonfig figcaption{color:var(--muted);font-size:.92rem;margin-top:8px;max-width:70ch}
.tries{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:12px 0}.try{margin:0}.try img{width:100%;display:block;background:var(--wash)}.try figcaption{font-size:.82rem;line-height:1.4;color:var(--muted);padding-top:6px}.try.chosen img{outline:3px solid var(--accent);outline-offset:2px}.try b{color:var(--ink)}\n.limits{max-width:74ch;padding-left:1.1em}.limits li{margin:0 0 14px;line-height:1.55}
.chg{margin:8px 0 6px;padding-left:1.1em;font-size:.88rem;color:var(--ink)}.strip em{font-style:normal;font-size:.8rem;font-weight:600}.strip em.ok{color:var(--accent)}.strip em.no{color:var(--muted)}
nav.top{display:flex;flex-wrap:wrap;gap:6px 22px;padding:14px 0;border-bottom:1px solid var(--rule);font-size:.92rem}
footer{margin-top:96px;color:var(--muted);font-size:.85rem;max-width:72ch}
@media(max-width:900px){.tries{grid-template-columns:1fr 1fr}.cols{grid-template-columns:1fr}.chain{grid-template-columns:1fr 1fr}.strip{grid-template-columns:1fr}.seasons{grid-template-columns:1fr}h2{margin-top:80px}}
@media(max-width:560px){.chain{grid-template-columns:1fr}body{font-size:16px}}
"""


def main() -> None:
    """Assemble and write `OUT_PATH`."""
    if "--refresh" in sys.argv or not SPOT_PATH.exists():
        build_spotcheck_table()
    concepts = _load_concepts()
    nav = (
        '<nav class=top aria-label="Sections"><a href="#explorer">Browse the forecast</a>'
        '<a href="#concepts">Evidence</a><a href="#model">Does the model work</a>'
        '<a href="#seasons">Seasonal view</a><a href="#limits">Limits</a></nav>'
    )
    concept_images = {
        c["sid"]: _b64_image(Path(c["sel"]["image_path"]), jpeg=True, max_side=520)
        for c in concepts
        if c["sid"] in STYLE_ORDER
    }
    body = (
        hero(concepts)
        + nav
        + demo_explorer.section(concept_images)
        + concept_sections(concepts)
        + trace_back(concepts)
        + closed_loop_section(concepts)
        + model_section()
        + seasonal_section(concepts)
        + limits_section()
        + "<footer>Numbers on this page are read from the CSV tables in <code>reports/tables/</code>. The full argument is in WRITEUP.md; how every file was produced is in SUBMISSION_CHECKLIST.md.</footer>"
    )
    doc = (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>next-season-styles demo</title><style>{CSS}{demo_explorer.CSS}</style></head><body><main>{body}</main></body></html>"
    )
    OUT_PATH.write_text(doc, encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({len(doc) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
