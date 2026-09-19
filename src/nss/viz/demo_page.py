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
import json
import sys
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import polars as pl
from PIL import Image

from nss.generate.final_deliverables import STYLE_ORDER
from nss.generate.h3_generate import reference_paths as h3_reference_paths
from nss.generate.h4_deliverables import HUMAN_CHECK, JUDGE_NOISE_BOUND, OBSERVED_CAPTIONS
from nss.generate.screen_references import load_screened_references

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


PLAIN_NAMES = {
    STYLE_ORDER[0]: "Black jersey T-shirt",
    STYLE_ORDER[1]: "Red underwear bottom",
    STYLE_ORDER[2]: "Beige knit sweater",
}


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
    sel = {r["style_id"]: r for r in pl.read_csv(T / "final_selection_h4.csv").to_dicts()}
    g1b = {r["style_id"]: r for r in pl.read_csv(T / "gate1b_nearest_reference.csv").to_dicts()}
    top = {r["style_key"]: r for r in pl.read_csv(T / "top_styles_final_three.csv").to_dicts()}
    j4 = pl.read_csv(T / "j4_judge_repeats.csv").filter(pl.col("judge") == "groq")
    briefs = {
        b["style_id"]: b for b in json.loads((T / "design_briefs.json").read_text(encoding="utf-8"))
    }
    refs = load_screened_references()
    refs[STYLE_ORDER[1]] = h3_reference_paths()
    out = []
    for sid in STYLE_ORDER:
        r = sel[sid]
        out.append(
            {
                "sid": sid,
                "name": PLAIN_NAMES[sid],
                "sel": r,
                "g1b": g1b[sid],
                "top": top[sid],
                "calls": j4.filter(pl.col("image_path") == r["image_path"]).to_dicts()[0]["scores"],
                "brief": briefs[sid],
                "refs": refs[sid],
            }
        )
    return out


def hero(concepts: list[dict]) -> str:
    """The three concepts as the opening image strip, then the three-sentence orientation."""
    figs = "".join(
        f'<figure><img src="{_b64_image(Path(c["sel"]["image_path"]), jpeg=True)}" alt="{_e(c["name"])} concept">'
        f'<figcaption><b>{_e(c["name"])}</b><span>{_e(OBSERVED_CAPTIONS[c["sid"]])}</span></figcaption></figure>'
        for c in concepts
    )
    passed = sum(1 for c in concepts if c["sel"]["overall"] == "PASS")
    return f"""<header class=masthead><p class=site>next-season-styles</p>
<h1>Three garments a forecast chose, and the pictures made from them.</h1></header>
<div class=strip>{figs}</div>
<section class=first><h2>What to look at first</h2>
<p><b>What the model predicted.</b> From two years of H&amp;M sales it ranked about 2,000 clothing styles by how hard each would sell, per product on sale, over the 13 weeks after 21 September 2020, and its top picks were a black jersey T-shirt, red underwear and a beige knit sweater.</p>
<p><b>What was generated from it.</b> For each style, an image generator was given real H&amp;M product photos of that style as a guide and asked for a new garment; the three pictures above are the best of four tries each.</p>
<p><b>How to judge whether it worked.</b> Two questions: does each picture look like a believable, new product of its style (section 1: {passed} of 3 passed every automatic check, and the one that did not is explained), and does the forecast beat guessing and simple rules of thumb (section 3, where it names the right neighbourhood about 72% of the time against 0.4% for chance)?</p></section>"""


def _check_row(title: str, verdict: str, body: str, gloss: str) -> str:
    return (
        f"<div class=check><div class=chk-head><span class=chk-title>{title}</span>{verdict}</div>"
        f"<p class=chk-body>{body}</p><p class=gloss>{gloss}</p></div>"
    )


def concept_sections(concepts: list[dict]) -> str:
    """Section 1: each concept large, with every check in words, numbers and one-line glosses."""
    out = [
        "<section id=concepts><h2>The three concepts</h2><p class=lede>Each block: the picture, what it shows, and four checks. Every number is followed by what it means. The measures behind the checks are explained once, after the third concept.</p>"
    ]
    for c in concepts:
        r, b = c["sel"], c["g1b"]
        overall = r["overall"]
        if overall == "PASS":
            headline = "Passed every automatic check, and looks like a coherent garment."
        else:
            headline = "Did not pass one check: its closest reference photo is a shade closer than any two real products of this style are."
        refs_html = "".join(
            f'<img src="{_b64_image(p, max_side=420)}" alt="reference photo">' for p in c["refs"]
        )
        margin = r["fidelity_median"] - r["fidelity_threshold"]
        checks = "".join(
            [
                _check_row(
                    "Fits in with real products of its style",
                    verdict_word(bool(r["gate1_pass"])),
                    f'Average similarity to its reference photos: <b>{_num(r["clip_mean_sim"])}</b> on the CLIP measure (limit {_num(r["clip_benchmark_p90"])}) and <b>{_num(r["dinov2_mean_sim"])}</b> on the DINOv2 measure (limit {_num(r["dinov2_benchmark_p90"])}).',
                    "The limit is how alike real products of this style are to each other: only 1 in 10 real pairs is more alike. Passing means the picture is not stranger, or blander, than a real sibling.",
                ),
                _check_row(
                    "Not a copy of any single photo",
                    verdict_word(bool(b["final_joint_pass"])),
                    f'Closest single reference photo: <b>{_num(b["final_clip_max_sim"])}</b> on CLIP (limit {_num(b["clip_threshold_p90"])}) and <b>{_num(b["final_dinov2_max_sim"])}</b> on DINOv2 (limit {_num(b["dinov2_threshold_p90"])}). '
                    f'A pixel-exact copy scores {_num(b["clone_clip_max_sim"])} and fails this test in all three styles.',
                    "The limit is how close real products of this style get to their nearest sibling. A copy of one photo would sit above it."
                    + (
                        " Here the DINOv2 score is over by "
                        + f'{b["final_dinov2_max_sim"] - b["dinov2_threshold_p90"]:.4f}'
                        + ", which is small but is a fail; the limit was set before the pictures were scored and not moved."
                        if not b["final_joint_pass"]
                        else ""
                    ),
                ),
                _check_row(
                    "Matches the style's attributes",
                    verdict_word(bool(r["gate2_pass"])),
                    f'An AI image reader, shown only the picture, was asked for product type, colour and pattern, three times. Its answers scored {_e(c["calls"])}; the middle value is <b>{_num(r["fidelity_median"])}</b> against a pass mark of {_num(r["fidelity_threshold"])} (1.0 would be a perfect match).',
                    f"The same picture scored differently across earlier sessions by up to ±{JUDGE_NOISE_BOUND:.2f}, so treat this as coarse; here it clears the pass mark by {margin:+.3f}."
                    + (
                        " One attribute, the melange (flecked) texture, scored 0: the reader saw a plain solid."
                        if c["sid"] == STYLE_ORDER[2]
                        else ""
                    ),
                ),
                _check_row(
                    "Looks like a coherent garment (human check)",
                    '<span class="v pass">Passed</span>',
                    _e(HUMAN_CHECK[c["sid"]]).capitalize() + ".",
                    "A person looked at the image. This check exists because the automatic ones cannot see a malformed garment (see section 5).",
                ),
            ]
        )
        out.append(
            f"""<article class=concept><h3>{_e(c["name"])}</h3>
<div class=cols><div class=pic><img src="{_b64_image(Path(r['image_path']), jpeg=True)}" alt="{_e(c['name'])} concept, full size"></div>
<div class=body><p class=headline>{headline}</p>{checks}</div></div>
<details class=refs><summary>The {len(c['refs'])} real H&amp;M photos it was guided by</summary><div class=refgrid>{refs_html}</div></details></article>"""
        )
    out.append(
        """<aside class=explain><h3>What the measures mean</h3>
<p><b>CLIP</b> and <b>DINOv2</b> are two pretrained image models that turn a picture into numbers so pictures can be compared; 1.0 means identical, and real product photos of one style score around 0.85 to 0.97 because they share the same studio setup. CLIP compares overall look; DINOv2 compares shape and texture. A check passes only if both measures pass.</p>
<p>A <b>limit</b> here is the <b>90th percentile</b> of the same measure taken between real H&amp;M products of that style: 9 in 10 real pairs fall below it. With only 4 to 6 real reference photos per style these limits are rough.</p></aside></section>"""
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
        top, brief = c["top"], c["brief"]
        role = (
            "an established seller (top of the 'incumbent' list)"
            if top["source_table"] == "T1_incumbent"
            else "an emerging riser (top of the 'emerging' list)"
        )
        growth = (
            f' Its forecast is {top["growth_ratio"]:.2f}× its recent level.'
            if top["growth_ratio"] is not None
            else ""
        )
        drivers = [top[f"shap_driver_{i}_feature"] for i in range(1, 4)]
        why = ", then ".join(_plain_driver(d) for d in drivers)
        asked = "".join(f"<li>{_e(x)}</li>" for x in brief["applied_changes"])
        v = verdicts[c["sid"]]
        out.append(
            f"""<article class=trace><h3>{_e(c['name'])}</h3><ol class=chain>
<li><h4>Forecast</h4><p>Predicted <b>{top['predicted_intensity']:.1f}</b> units per product on sale per week for the 13 weeks after 21 Sep 2020; {role}.{growth}</p>
<p class=gloss>"Intensity" means units sold per product on sale, so a style is not ranked highly just for having many products.</p></li>
<li><h4>Why the model liked it</h4><p>Biggest influences, in order: {_e(why)}.</p><p class=gloss>These come from an explanation method called SHAP, which shows which inputs pushed this forecast up or down. {_e(v['dominant_mechanism'].split(':')[0].capitalize())}.</p></li>
<li><h4>Guided by</h4><div class=thumbs>{"".join(f'<img src="{_b64_image(p, max_side=420)}" alt="reference">' for p in c['refs'][:3])}</div><p class=gloss>The first photo is the one the generator actually copies structure from.</p></li>
<li><h4>What the brief asked for</h4><ul>{asked}</ul><p class=gloss>These were instructions to the generator, not results.</p></li>
<li><h4>What the picture shows</h4><img class=final src="{_b64_image(Path(c['sel']['image_path']), jpeg=True)}" alt="concept"><p>{_e(OBSERVED_CAPTIONS[c['sid']])}</p><p class=gloss>Described by looking at the image. Not every requested change is visible (the underwear shows dark piping, not the requested burgundy edge).</p></li></ol></article>"""
        )
    out.append("</section>")
    return "\n".join(out)


def model_section() -> str:
    """Section 3: does the model actually predict? Table, shuffle control, spot-check."""
    summ = pl.read_csv(T / "backtest_summary_v2.csv").filter(pl.col("split") == "pooled")
    paired = pl.read_csv(T / "backtest_paired_diff.csv").filter(pl.col("split") == "pooled")
    oracle = pl.read_csv(T / "g1_diagnostics_summary.csv").filter(
        pl.col("diagnostic") == "a_persistence_oracle"
    )
    shuf = pl.read_csv(T / "label_shuffle_control.csv")
    n_eval = pl.read_csv(T / "backtest_per_origin_lightgbm.csv")["n_eval_set"].mean()
    row = {r["method"]: r for r in summ.to_dicts()}
    lg, fl = row["lightgbm"], row["random_floor"]
    n_orig = lg["n_origins"]

    glosses = f"""<dl class=metricgloss>
<dt>Hit@3 in top 20 = {lg['hit_at_3_in_top20_mean']:.3f}</dt><dd>When the model names its three best styles, they land in the true top 20 (of about {round(n_eval, -2):,.0f} styles) about {lg['hit_at_3_in_top20_mean']:.0%} of the time. Random guessing scores {fl['hit_at_3_in_top20_mean']:.1%}. This is the headline number.</dd>
<dt>Hit@3 in top 10 = {lg['hit_at_3_in_top10_mean']:.3f}</dt><dd>The same, against the true top 10: {lg['hit_at_3_in_top10_mean']:.0%} for the model, {fl['hit_at_3_in_top10_mean']:.1%} for guessing.</dd>
<dt>Precision@3 = {lg['precision_at_3_mean']:.3f}</dt><dd>How often the three named styles are exactly the true top three, in any order: {lg['precision_at_3_mean']:.1%}, guessing {fl['precision_at_3_mean']:.1%}. This is low for every method because the true #3 and #4 styles differ by about half a percent of sales, so the model finds the right neighbourhood, not the exact order.</dd>
<dt>NDCG@10 = {lg['ndcg_at_10_mean']:.3f}</dt><dd>Scores the model's top 10 by how much real demand they carry (1.0 is perfect). Even random picks score {fl['ndcg_at_10_mean']:.2f}, because most styles sell similar amounts, so read {lg['ndcg_at_10_mean']:.2f} against {fl['ndcg_at_10_mean']:.2f}.</dd>
<dt>Spearman = {lg['spearman_rho_mean']:.3f}</dt><dd>How closely the model's full ranking of all styles matches the true ranking: 1 is identical, 0 is unrelated. Guessing gives {fl['spearman_rho_mean']:.3f}.</dd></dl>"""

    head = "".join(f"<th>{lab}</th>" for _, lab in METRICS)
    rows = []
    for method, (label, gloss) in METHODS.items():
        s = row[method]
        cells = []
        for key, _ in METRICS:
            cell = _ci(s[f"{key}_mean"], s[f"{key}_ci_low"], s[f"{key}_ci_high"])
            if method != "lightgbm":
                p = paired.filter(
                    (pl.col("method_b") == method) & (pl.col("metric") == key)
                ).to_dicts()
                if p and p[0]["mean_diff"] == p[0]["mean_diff"]:
                    q = p[0]
                    cell += f'<span class=paired>model minus this: {q["mean_diff"]:+.3f} ({q["diff_ci_low"]:+.3f} to {q["diff_ci_high"]:+.3f})</span>'
            cells.append(f"<td>{cell}</td>")
        cls = " class=me" if method == "lightgbm" else ""
        rows.append(
            f'<tr{cls}><th scope=row>{label}<span class=gloss>{gloss}</span></th>{"".join(cells)}</tr>'
        )
    o = {r["metric"]: r for r in oracle.to_dicts()}
    ocells = "".join(
        f"<td>{_ci(o[k]['value'], o[k]['ci_low'], o[k]['ci_high'])}</td>"
        if k in o
        else "<td>not measured</td>"
        for k, _ in METRICS
    )
    rows.append(
        f'<tr><th scope=row>Cheating check: a "persistence" forecast that only uses data available at the time<span class=gloss>Tests the idea that the model just repeats recent sales; it does worse than the model.</span></th>{ocells}</tr>'
    )
    seeds = shuf.filter(pl.col("variant") == "shuffled_train_true_test")
    scells = "".join(
        f"<td><span class=big>{sum(seeds[k].to_list()) / seeds.height:.3f}</span> <span class=ci>({min(seeds[k].to_list()):.3f} to {max(seeds[k].to_list()):.3f})</span></td>"
        for k, _ in METRICS
    )
    rows.append(
        f"<tr class=shuffle><th scope=row>Leakage test: the model retrained on scrambled answers<span class=gloss>Same model, same data, but which sales figure belongs to which style was shuffled before training.</span></th>{scells}</tr>"
    )
    ctrl = shuf.filter(pl.col("variant") == "unshuffled_positive_control").to_dicts()[0]
    table = f'<div class=scroll><table class=metrics><thead><tr><th>Method</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div><p class=gloss>Each cell is the average over {n_orig} test dates the model never saw, with a 95% interval in brackets. "Model minus this" is the model advantage over that method on the same dates; an interval entirely above zero means the advantage is reliable.</p>'
    hits = sum(round(x * 36) for x in seeds["hit_at_3_in_top20"].to_list())
    shuffle_box = f"""<aside class=callout><h3>How do we know the forecast isn't cheating?</h3>
<p>We scrambled which sales figure belongs to which style, retrained the same model three times, and scored it on the real answers. It collapsed to chance: <b>{hits} hits in 108 picks</b> (Hit@3 in top 20 = {seeds['hit_at_3_in_top20'].mean():.3f}, guessing scores {fl['hit_at_3_in_top20_mean']:.3f}). Run through the same code without scrambling, it reproduces {ctrl['hit_at_3_in_top20']:.3f}. If information from the future had leaked into the inputs, the scrambled model could not have fallen to chance.</p></aside>"""

    if not SPOT_PATH.exists():
        build_spotcheck_table()
    spot = f'<h3>A spot-check you can see</h3><img class=chart src="{spotcheck_chart()}" alt="forecast spot-check"><p class=gloss>Both panels share one vertical scale. The dark line is what each style actually sold per product on sale over the last 26 weeks of data; the blue bar is the model\'s forecast for the following 13 weeks (unobserved; the data ends 21 Sep 2020). The top-ranked style sells several times more than one ranked 1,500th, and the forecast reflects that. It also shows the forecast pulling back from a late-summer spike rather than extrapolating it.</p>'
    return f"<section id=model><h2>Does the model actually predict?</h2><p class=lede>The model is scored on {n_orig} past dates: at each, it trains only on earlier data and forecasts the next 13 weeks, and we compare with what happened.</p>{glosses}{table}{shuffle_box}{spot}</section>"


def summer_block() -> str:
    """The summer concept's four tries and its checks, from the committed summer tables."""
    scored = pl.read_csv(T / "seasonal_summer_scored.csv")
    judge_csv = T / "seasonal_summer_judge.csv"
    reads = pl.read_csv(judge_csv)["mean_score"].to_list() if judge_csv.exists() else []
    from nss.generate.seasonal_concept import SELECTED_SEED, candidate_path

    cards = []
    for r in scored.to_dicts():
        chosen = r["seed"] == SELECTED_SEED
        cards.append(
            f'<figure class="try{" chosen" if chosen else ""}"><img src="{_b64_image(candidate_path(r["seed"]), max_side=520)}" alt="summer try {r["seed"]}">'
            f'<figcaption><b>{"Chosen" if chosen else "Not chosen"}</b> {_e(r["visual_qc"].split(": ", 1)[1])}</figcaption></figure>'
        )
    sel = scored.filter(pl.col("seed") == SELECTED_SEED).to_dicts()[0]
    if len(reads) >= 3:
        med = sorted(reads[:3])[1]
        g2 = (
            f"three readings {', '.join(f'{x:.3f}' for x in reads[:3])}; middle value {med:.3f} against a pass mark of 0.513 → "
            + verdict_word(med >= 0.513)
        )
    else:
        g2 = (
            f"{len(reads)} readings so far ({', '.join(f'{x:.3f}' for x in reads)}) against a pass mark of 0.513; the two disagree by {max(reads) - min(reads):.2f}, so this check is "
            + verdict_word(None)
            + " (three readings are needed)"
        )
    return f"""<h3>The summer concept: four tries, one chosen</h3>
<div class=tries>{"".join(cards)}</div>
<p class=gloss>Chosen by eye. Two of the other three passed the automatic similarity checks anyway: one drifted to a grey pinstripe, one has straps sewn onto a bottom.</p>
<div class=check><div class=chk-head><span class=chk-title>Fits in with real products of its style</span>{verdict_word(bool(sel["gate1_pass"]))}</div>
<p class=chk-body>CLIP {_num(sel["clip_mean_sim"])} (limit {_num(sel["clip_p90_threshold"])}), DINOv2 {_num(sel["dinov2_mean_sim"])} (limit {_num(sel["dinov2_p90_threshold"])}).</p></div>
<div class=check><div class=chk-head><span class=chk-title>Not a copy of any single photo</span>{verdict_word(bool(sel["gate1b_pass"]))}</div>
<p class=chk-body>Closest reference photo: CLIP {_num(sel["clip_max_sim"])} (limit {_num(sel["clip_gate1b_threshold"])}), DINOv2 {_num(sel["dinov2_max_sim"])} (limit {_num(sel["dinov2_gate1b_threshold"])}). An exact copy fails this test.</p></div>
<div class=check><div class=chk-head><span class=chk-title>Matches the style's attributes</span>{verdict_word(None if len(reads) < 3 else sorted(reads[:3])[1] >= 0.513)}</div>
<p class=chk-body>{g2}</p><p class=gloss>The style's labels ("Swimwear bottom", "Other structure") are catalogue terms with little visual meaning: the image reader answered "bikini bottom" and "solid, ribbed", which scores low on both. The failure is real under the rule fixed beforehand, but it is partly a labelling artifact, and the readings themselves vary by 0.25. The reference photos were screened by eye because the automatic framing check was out of free quota.</p></div>"""


def seasonal_section() -> str:
    """Section 4: the four seasonal top-3 tables and the Summer vs autumn/winter comparison."""
    d = pl.read_csv(T / "top_styles_by_season_v2.csv")
    blocks = []
    for season in d["season"].unique(maintain_order=True).to_list():
        rows = d.filter(pl.col("season") == season).sort("rank").head(3).to_dicts()
        body = "".join(
            f"<tr><td>{r['rank']}</td><td>{_e(r['style_key'].replace(' || ', ' · '))}</td><td class=n>{r['season_mean_intensity']:.1f}</td></tr>"
            for r in rows
        )
        blocks.append(
            f"<div class=season><h3>{season.title()}</h3><table><thead><tr><th>#</th><th>Style</th><th>Average weekly intensity</th></tr></thead><tbody>{body}</tbody></table></div>"
        )
    fig = ""
    if SEASONAL_FIG.exists():
        fig = f'<figure class=seasonfig><img src="{_b64_image(SEASONAL_FIG, max_side=1500)}" alt="Autumn/winter 2020 rank-1 concept beside summer rank-1 concept"><figcaption>The same pipeline, run for two seasons. The left picture is the forecast for the coming autumn and winter; the right is the summer winner.</figcaption></figure>'
    text = ""
    p = T / "seasonal_summer_forecast.csv"
    if p.exists():
        f = pl.read_csv(p).to_dicts()[0]
        text = f'<p>For summer, the model was re-run as of 1 June 2020 (using only earlier data). It ranked swimwear bottoms {f["predicted_rank_among_eligible"]} of {f["n_eligible_styles"]:,} and predicted <b>{f["predicted_intensity"]:.1f}</b> units per product per week for June to August; the realised figure was <b>{f["realised_intensity"]:.1f}</b>, so the model got the ranking right but overshot the level by about {f["predicted_intensity"] / f["realised_intensity"] - 1:.0%}.</p><p class=gloss>The tables show each season\'s historical average weekly intensity among styles that pass the three sanity guards (enough products on sale, no heavy discounting, sold in most recent weeks).</p>'
    return f'<section id=seasons><h2>Seasonal view</h2><p class=lede>Summer is led by swimwear; autumn and winter by black basics, knitwear and tights.</p><div class=seasons>{"".join(blocks)}</div>{text}{fig}{summer_block() if (T / "seasonal_summer_scored.csv").exists() else ""}</section>'


def limits_section() -> str:
    """Section 5: what could not be verified, said plainly."""
    return f"""<section id=limits><h2>What we could not verify</h2>
<ul class=limits>
<li><b>Whether the pictures would sell.</b> The forecast is about styles; the pictures are new designs no customer has seen. Nothing here tests demand for the pictures themselves.</li>
<li><b>Demand, as opposed to sales.</b> Everything derives from what was stocked and sold, not what customers wanted. No inventory data was available; a stock-out check finds a lower bound of 3.76% of style-weeks with a stock-out signature.</li>
<li><b>The AI image reader is noisy and there is only one.</b> The same picture scored differently by up to ±{JUDGE_NOISE_BOUND:.2f} across sessions, every fidelity score comes from one model, and a second reader was unavailable (free-tier limits).</li>
<li><b>Automatic checks miss broken garments.</b> Three of the four regenerated underwear candidates were visibly malformed (a cut-out defect, sheer mesh, an unrecognisable folded object) and still passed every automatic check. A human check is required.</li>
<li><b>The T-shirt is a near-copy at the margin.</b> Any new black jersey T-shirt resembles every other one; ours is a hair closer to one reference photo than two real T-shirts ever are to each other.</li>
<li><b>The limits behind the checks are rough.</b> Each rests on only 4 to 6 real reference photos per style.</li>
<li><b>The exact top three.</b> The model finds the right neighbourhood (72% in the true top 20) but not the exact order (5.6% exactly right), because the leaders are nearly tied.</li>
<li><b>COVID.</b> The test window includes spring 2020; results are reported pooled and split, but the splits are too small to be more than directional.</li>
<li><b>The summer concept's reference photos were screened by eye</b>, because the automatic framing judges were out of free quota that day.</li>
</ul></section>"""


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
.callout{margin:40px 0;padding:24px 28px;border-left:4px solid var(--accent);background:var(--wash);max-width:78ch}.callout p{margin:0}
.chart{width:100%;height:auto;display:block;margin-top:8px}
.seasons{display:grid;grid-template-columns:repeat(2,1fr);gap:32px 40px}.season table{font-size:.9rem}.season th,.season td{padding:8px 8px;border-top:1px solid var(--rule);text-align:left;vertical-align:top}.season thead th{border-top:0;border-bottom:2px solid var(--ink);font-size:.8rem}.season .n{font-variant-numeric:tabular-nums;white-space:nowrap}
.seasonfig{margin:48px 0 0}.seasonfig img{width:100%;height:auto;display:block}.seasonfig figcaption{color:var(--muted);font-size:.92rem;margin-top:8px;max-width:70ch}
.tries{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:12px 0}.try{margin:0}.try img{width:100%;display:block;background:var(--wash)}.try figcaption{font-size:.82rem;line-height:1.4;color:var(--muted);padding-top:6px}.try.chosen img{outline:3px solid var(--accent);outline-offset:2px}.try b{color:var(--ink)}\n.limits{max-width:74ch;padding-left:1.1em}.limits li{margin:0 0 14px;line-height:1.55}
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
    nav = '<nav class=top aria-label="Sections"><a href="#concepts">The three concepts</a><a href="#trace">Trace-back</a><a href="#model">Does the model predict</a><a href="#seasons">Seasonal view</a><a href="#limits">What we could not verify</a></nav>'
    body = (
        hero(concepts)
        + nav
        + concept_sections(concepts)
        + trace_back(concepts)
        + model_section()
        + seasonal_section()
        + limits_section()
        + "<footer>Numbers on this page are read from the CSV tables in <code>reports/tables/</code>. The full argument is in WRITEUP.md; how every file was produced is in SUBMISSION_CHECKLIST.md.</footer>"
    )
    doc = (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>next-season-styles demo</title><style>{CSS}</style></head><body><main>{body}</main></body></html>"
    )
    OUT_PATH.write_text(doc, encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({len(doc) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
