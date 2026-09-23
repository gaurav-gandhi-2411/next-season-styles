"""Phase B: four accuracy levers, quantile output, combination (PREREGISTRATION.md Section R).

Every rule here was committed before any lever was trained (Section R at `ed8d409`, amendments R8
at `9105766`). Two stages, so the levers can train in parallel processes:

    uv run python -m nss.models.phase_b run <lever>   # writes data/generated/phase_b_*.parquet
    uv run python -m nss.models.phase_b evaluate      # writes reports/tables/phase_b_*.csv

Levers: `repro` (the champion through this module's own training path; must equal the Phase A
frame), `b1`, `b1_shuffle`, `b1_random`, `b2`, `b3`, `b4_seed<S>` (S in 43..51), `b5_q<10|50|90>`,
`combo` (only if Section R7 calls for it).

All training reuses the champion's embargoed weekly-origin serving path: each weekly origin is
served by the model of its 4-week grid block, trained on origins whose 13-week label window closed
before it (`backtest_embargo_check.embargoed_train_origin_weeks`), with the locked
`FINAL_MODEL_CONFIG` and the champion's determinism parameters. Only the seed and objective vary,
and only where a lever says so.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from nss.features.model_features import build_features
from nss.features.style_panel import STYLE_KEY_COLS
from nss.features.targets import HORIZON_WEEKS, compute_forward_target
from nss.models.backtest import Origin
from nss.models.backtest_embargo_check import embargoed_train_origin_weeks
from nss.models.circular_bootstrap import bootstrap_p_two_sided, circular_ci, holm_decisions
from nss.models.final_forecast import FINAL_MODEL_CONFIG
from nss.models.lightgbm_model import (
    _categorical_indices,
    _to_lgb_matrix,
    build_model_frame,
    feature_columns,
)
from nss.models.metrics import METRIC_KEYS, score_predictions
from nss.models.phase_a_measure import (
    MARGIN_PREREGISTERED,
    NEW_METRICS,
    _stamp,
    new_metric_values,
    weekly_origins,
)

PANEL = "data/processed/style_week_panel.parquet"
GEN = Path("data/generated")
OUT = "reports/tables"
CHAMPION_SEED = 42
B4_SEEDS: tuple[int, ...] = tuple(range(42, 52))
QUANTILES: tuple[float, ...] = (0.10, 0.50, 0.90)
K_NEIGHBOURS = 10
PRIMARY = "demand_capture_at_20"
GUARDRAILS: dict[str, int] = {  # +1: higher is better, -1: lower is better
    "hit_at_3_in_top20": 1,
    "ndcg_at_10": 1,
    "spearman_rho": 1,
    "wmape": -1,
}
HOLM_LEVERS: tuple[str, ...] = ("b1", "b2", "b3", "b4")
METRICS: tuple[str, ...] = (*METRIC_KEYS, *NEW_METRICS)
PT_COLS = ["index_group_name", "product_type_name"]
IG_COLS = ["index_group_name"]


# ---------------------------------------------------------------------------
# training path
# ---------------------------------------------------------------------------


def fit(
    train: pl.DataFrame,
    columns: list[str],
    label: str,
    seed: int = CHAMPION_SEED,
    objective: str = "regression",
    alpha: float | None = None,
) -> lgb.LGBMRegressor:
    """`lightgbm_model.train_lightgbm` with the seed and objective exposed, nothing else changed."""
    extra = {"alpha": alpha} if alpha is not None else {}
    model = lgb.LGBMRegressor(
        objective=objective,
        random_state=seed,
        verbosity=-1,
        num_leaves=int(FINAL_MODEL_CONFIG["num_leaves"]),
        learning_rate=float(FINAL_MODEL_CONFIG["learning_rate"]),
        n_estimators=int(FINAL_MODEL_CONFIG["n_estimators"]),
        min_child_samples=int(FINAL_MODEL_CONFIG["min_child_samples"]),
        deterministic=True,
        force_row_wise=True,
        num_threads=1,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        data_random_seed=seed,
        **extra,
    )
    model.fit(
        _to_lgb_matrix(train, columns),
        train[label].to_numpy().astype(np.float64),
        categorical_feature=_categorical_indices(train, columns),
        feature_name=columns,
    )
    return model


def serve(
    frame: pl.DataFrame,
    columns: list[str],
    grid: Sequence[Origin],
    weeks: Sequence[date],
    label: str = "y_true",
    **fit_kw: object,
) -> pl.DataFrame:
    """Predictions at every weekly origin, each from its grid block's embargoed model."""
    grid_weeks = [o.origin_week for o in grid]
    models: dict[int, lgb.LGBMRegressor] = {}
    parts = []
    for t in sorted(weeks):
        k = max(i for i, w in enumerate(grid_weeks) if w <= t)
        if k not in models:
            train_weeks = list(embargoed_train_origin_weeks(list(grid), k))
            train = frame.filter(
                pl.col("origin_week").is_in(train_weeks) & pl.col(label).is_not_null()
            )
            models[k] = fit(train, columns, label, **fit_kw)  # type: ignore[arg-type]
        test = frame.filter(pl.col("origin_week") == t)
        preds = models[k].predict(_to_lgb_matrix(test, columns))
        parts.append(
            test.select("style_key", "origin_week").with_columns(pl.Series("y_pred", preds))
        )
    return pl.concat(parts)


def base_frame(panel: pl.DataFrame, grid: Sequence[Origin], weeks: Sequence[date]) -> pl.DataFrame:
    """The champion's model frame over every grid and weekly origin (one build, as in growth)."""
    all_weeks = sorted({*(o.origin_week for o in grid), *weeks})
    return build_model_frame(panel, all_weeks)


# ---------------------------------------------------------------------------
# B.1 visual momentum
# ---------------------------------------------------------------------------


def article_table() -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Embedded articles (style, first-sale date) sorted by first sale, plus their CLIP and DINOv2
    matrices in the same row order. Never-sold articles are dropped."""
    clip = np.load("data/retrieval_cache/emb_clip.npz")
    dino = np.load("data/retrieval_cache/emb_dino.npz")
    keys = sorted(clip.files)
    art = pl.read_csv("data/raw/articles.csv", schema_overrides={"article_id": pl.String}).select(
        "article_id", pl.concat_str(STYLE_KEY_COLS, separator=" || ").alias("style_key")
    )
    first = (
        pl.scan_parquet(
            "data/interim/transactions_train_parquet/**/*.parquet", hive_partitioning=True
        )
        .group_by("article_id")
        .agg(pl.col("t_dat").min().alias("first_sale"))
        .collect()
        .with_columns(pl.col("article_id").cast(pl.String).str.zfill(10))
    )
    table = (
        pl.DataFrame({"article_id": keys, "row": np.arange(len(keys))})
        .join(art, on="article_id", how="left")
        .join(first, on="article_id", how="left")
        .filter(pl.col("first_sale").is_not_null() & pl.col("style_key").is_not_null())
        .sort("first_sale", "article_id")
    )
    ids = table["article_id"].to_list()
    mat_c = np.empty((len(ids), clip[ids[0]].shape[0]), dtype=np.float32)
    mat_d = np.empty((len(ids), dino[ids[0]].shape[0]), dtype=np.float32)
    for i, k in enumerate(ids):
        mat_c[i] = clip[k]
        mat_d[i] = dino[k]
    return table, mat_c, mat_d


def _unit(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(n == 0, 1.0, n)


def vis_momentum(
    panel: pl.DataFrame, weeks: Sequence[date], mode: str, seed: int = CHAMPION_SEED
) -> pl.DataFrame:
    """`vis_nbr_momentum` for every style with a panel row at each week (R2 with R8).

    mode: "nearest" (the lever), "random" (negative control). The shuffle control permutes the
    "nearest" feature afterwards, in `add_vis_feature`.
    """
    arts, a_clip, a_dino = article_table()
    feats = build_features(panel, list(weeks)).select(
        "style_key",
        "origin_week",
        (pl.col("ewma_halflife_4w").log1p() - pl.col("ewma_halflife_13w").log1p()).alias("g"),
    )
    style_ids = {s: i for i, s in enumerate(sorted(arts["style_key"].unique().to_list()))}
    n_styles = len(style_ids)
    sum_c = np.zeros((n_styles, a_clip.shape[1]))
    sum_d = np.zeros((n_styles, a_dino.shape[1]))
    count = np.zeros(n_styles, dtype=int)
    a_style = np.array([style_ids[s] for s in arts["style_key"]])
    a_first = arts["first_sale"].to_numpy()
    ptr = 0
    rng = np.random.default_rng(seed)
    parts = []
    for w in sorted(weeks):
        while ptr < len(a_first) and a_first[ptr] < np.datetime64(w):  # strictly before origin
            sum_c[a_style[ptr]] += a_clip[ptr]
            sum_d[a_style[ptr]] += a_dino[ptr]
            count[a_style[ptr]] += 1
            ptr += 1
        present = feats.filter(pl.col("origin_week") == w)
        keys = present["style_key"].to_list()
        g = present["g"].to_numpy()
        idx = [style_ids.get(k, -1) for k in keys]
        has = np.array([i >= 0 and count[i] > 0 for i in idx])
        pool = np.flatnonzero(has)
        value = np.full(len(keys), np.nan)
        if pool.size > 1:
            rows = np.array([idx[p] for p in pool])
            vc, vd = _unit(sum_c[rows]), _unit(sum_d[rows])
            sim = (vc @ vc.T + vd @ vd.T) / 2.0
            np.fill_diagonal(sim, -np.inf)
            kk = min(K_NEIGHBOURS, pool.size - 1)
            for a in range(pool.size):
                if mode == "nearest":
                    nb = np.argpartition(-sim[a], kk - 1)[:kk]
                    wts = np.maximum(sim[a, nb], 0.0)
                else:
                    others = np.delete(np.arange(pool.size), a)
                    nb = rng.choice(others, size=kk, replace=False)
                    wts = np.ones(kk)
                gj = g[pool[nb]]
                ok = ~np.isnan(gj)
                if ok.any() and wts[ok].sum() > 0:
                    value[pool[a]] = float(np.sum(wts[ok] * gj[ok]) / wts[ok].sum())
        parts.append(
            pl.DataFrame(
                {"style_key": keys, "origin_week": [w] * len(keys), "vis_nbr_momentum": value}
            )
        )
    return pl.concat(parts)


def add_vis_feature(frame: pl.DataFrame, feat: pl.DataFrame, shuffle: bool) -> pl.DataFrame:
    """Attach the feature to model-frame rows; optionally permute it within each origin."""
    out = frame.join(feat, on=["style_key", "origin_week"], how="left", maintain_order="left")
    if not shuffle:
        return out
    rng = np.random.default_rng(CHAMPION_SEED)
    parts = []
    for w in out["origin_week"].unique().sort().to_list():
        sub = out.filter(pl.col("origin_week") == w)
        parts.append(
            sub.with_columns(
                pl.Series("vis_nbr_momentum", rng.permutation(sub["vis_nbr_momentum"].to_numpy()))
            )
        )
    return pl.concat(parts).sort("origin_week", maintain_order=True)


# ---------------------------------------------------------------------------
# B.2 shrunk-intensity label
# ---------------------------------------------------------------------------


def add_shrunk_label(frame: pl.DataFrame, panel: pl.DataFrame) -> pl.DataFrame:
    weeks = frame["origin_week"].unique().sort().to_list()
    shrunk = pl.concat(
        [
            compute_forward_target(panel, w, HORIZON_WEEKS, target_column="intensity_shrunk")
            for w in weeks
        ]
    ).select("style_key", "origin_week", pl.col("target").alias("y_shrunk"))
    return frame.join(shrunk, on=["style_key", "origin_week"], how="left", maintain_order="left")


# ---------------------------------------------------------------------------
# B.3 hierarchical reconciliation
# ---------------------------------------------------------------------------


def parent_panel(panel: pl.DataFrame, key_cols: list[str], tag: str) -> pl.DataFrame:
    """Aggregate the style panel to a parent level and densify it weekly (R4)."""
    agg = (
        panel.group_by([*key_cols, "week_start"])
        .agg(
            pl.col("units").sum().cast(pl.Int64),
            pl.col("revenue").sum(),
            pl.col("n_active_articles").sum().cast(pl.Int64),
            (
                (pl.col("price_index") * pl.col("n_active_articles")).sum()
                / pl.col("n_active_articles").filter(pl.col("price_index").is_not_null()).sum()
            ).alias("price_index"),
            pl.col("first_week_seen").min(),
        )
        .with_columns(pl.concat_str([pl.lit(tag), *key_cols], separator=" || ").alias("style_key"))
    )
    spans = agg.group_by("style_key").agg(
        pl.col("week_start").min().alias("lo"),
        pl.col("week_start").max().alias("hi"),
        *[pl.col(c).first() for c in key_cols],
        pl.col("first_week_seen").min(),
    )
    grid = spans.with_columns(
        pl.date_ranges("lo", "hi", interval="1w").alias("week_start")
    ).explode("week_start")
    dense = grid.select("style_key", *key_cols, "week_start", "first_week_seen").join(
        agg.select(
            "style_key", "week_start", "units", "revenue", "n_active_articles", "price_index"
        ),
        on=["style_key", "week_start"],
        how="left",
    )
    dense = dense.with_columns(
        pl.col("units").fill_null(0),
        pl.col("revenue").fill_null(0.0),
        pl.col("n_active_articles").fill_null(0),
    ).with_columns(
        pl.when(pl.col("n_active_articles") > 0)
        .then(pl.col("units") / pl.col("n_active_articles"))
        .otherwise(0.0)
        .alias("units_per_active_article"),
        *[pl.lit("ALL").alias(c) for c in STYLE_KEY_COLS if c not in key_cols],
    )
    return dense.with_columns(pl.col("units_per_active_article").alias("intensity_shrunk")).sort(
        "style_key", "week_start"
    )


def reconcile(styles: pl.DataFrame, pt: pl.DataFrame, ig: pl.DataFrame) -> pl.DataFrame:
    """OLS reconciliation at one origin, raw-intensity space (R4 with R8).

    styles: style_key, index_group_name, product_type_name, weight, y_pred (log1p).
    pt / ig: parent key columns and y_pred (log1p), one row per parent with a base forecast.
    """
    n = styles.height
    yb = np.expm1(styles["y_pred"].to_numpy())
    w = styles["weight"].to_numpy().astype(float)
    rows_a, yhat = [], []
    for parents, cols in ((pt, PT_COLS), (ig, IG_COLS)):
        for r in parents.iter_rows(named=True):
            mask = np.ones(n, dtype=bool)
            for c in cols:
                mask &= (styles[c] == r[c]).to_numpy()
            tot = w[mask].sum()
            if tot <= 0:
                continue
            a = np.zeros(n)
            a[mask] = w[mask] / tot
            rows_a.append(a)
            yhat.append(float(np.expm1(r["y_pred"])))
    if not rows_a:
        return styles.select("style_key", "y_pred")
    a_mat = np.vstack(rows_a)
    lhs = np.eye(n) + a_mat.T @ a_mat
    rhs = yb + a_mat.T @ np.asarray(yhat)
    b = np.linalg.solve(lhs, rhs)
    return styles.select("style_key").with_columns(
        pl.Series("y_pred", np.log1p(np.clip(b, 0.0, None)))
    )


def run_b3(panel: pl.DataFrame, grid: Sequence[Origin], weeks: Sequence[date]) -> pl.DataFrame:
    champion = load_champion()
    base = {}
    for tag, cols in (("PT", PT_COLS), ("IG", IG_COLS)):
        pp = parent_panel(panel, cols, tag)
        fr = base_frame(pp, grid, weeks)
        preds = serve(fr, feature_columns(fr), grid, weeks)
        base[tag] = preds.join(pp.select("style_key", *cols).unique(), on="style_key", how="left")
        base[tag].write_parquet(GEN / f"phase_b_b3_base_{tag}_seed42_{_stamp(weeks)}.parquet")
    art = pl.read_parquet(PANEL).select("style_key", *PT_COLS).unique()
    parts = []
    for t in sorted(weeks):
        s = (
            champion.filter(pl.col("origin_week") == t)
            .select("style_key", pl.col("n_active_articles_level").alias("weight"), "y_pred")
            .join(art, on="style_key", how="left")
        )
        rec = reconcile(
            s,
            base["PT"].filter(pl.col("origin_week") == t),
            base["IG"].filter(pl.col("origin_week") == t),
        )
        parts.append(rec.with_columns(pl.lit(t).alias("origin_week")))
    return pl.concat(parts).select("style_key", "origin_week", "y_pred")


# ---------------------------------------------------------------------------
# run / evaluate
# ---------------------------------------------------------------------------


def champion_path(weeks: Sequence[date]) -> Path:
    return GEN / f"phase_a_model_predictions_{_stamp(weeks)}.parquet"


def load_champion() -> pl.DataFrame:
    panel = pl.read_parquet(PANEL)
    _, weekly = weekly_origins(panel)
    weeks = [o.origin_week for o in weekly]
    return pl.read_parquet(champion_path(weeks)).rename({"y_pred_lightgbm": "y_pred"})


def run(lever: str) -> None:
    t0 = time.time()
    panel = pl.read_parquet(PANEL)
    grid, weekly = weekly_origins(panel)
    weeks = [o.origin_week for o in weekly]
    seed = CHAMPION_SEED
    frame = base_frame(panel, grid, weeks)
    columns = feature_columns(frame)
    if lever == "repro":
        preds = serve(frame, columns, grid, weeks)
    elif lever in ("b1", "b1_shuffle", "b1_random"):
        all_weeks = frame["origin_week"].unique().sort().to_list()
        feat = vis_momentum(panel, all_weeks, "random" if lever == "b1_random" else "nearest")
        f2 = add_vis_feature(frame, feat, shuffle=lever == "b1_shuffle")
        preds = serve(f2, [*columns, "vis_nbr_momentum"], grid, weeks)
        # Missing values are NaN (LightGBM's missing marker), which polars does not count as null.
        vals = f2.filter(pl.col("origin_week").is_in(weeks))["vis_nbr_momentum"]
        cov = float((~vals.fill_null(float("nan")).is_nan()).mean())
        print(f"vis_nbr_momentum non-missing share on eval rows: {cov:.4f}")
    elif lever == "b2":
        preds = serve(add_shrunk_label(frame, panel), columns, grid, weeks, label="y_shrunk")
    elif lever == "b3":
        preds = run_b3(panel, grid, weeks)
    elif lever.startswith("b4_seed"):
        seed = int(lever.removeprefix("b4_seed"))
        preds = serve(frame, columns, grid, weeks, seed=seed)
    elif lever.startswith("b5_q"):
        alpha = int(lever.removeprefix("b5_q")) / 100
        preds = serve(frame, columns, grid, weeks, objective="quantile", alpha=alpha)
    elif lever == "combo":
        preds = run_combo(panel, grid, weeks, frame, columns)
    else:
        raise SystemExit(f"unknown lever {lever}")
    path = GEN / f"phase_b_{lever}_seed{seed}_{_stamp(weeks)}.parquet"
    preds.write_parquet(path)
    secs = time.time() - t0
    (GEN / f"phase_b_{lever}_timing.json").write_text(json.dumps({"lever": lever, "seconds": secs}))
    print(f"{lever}: {preds.height} rows -> {path} in {secs:.0f}s")


def run_combo(*_: object) -> pl.DataFrame:
    """Section R7 combination. Built only if two or more levers are adopted."""
    decisions = pl.read_csv(f"{OUT}/phase_b_decisions.csv")
    adopted = decisions.filter(pl.col("adopted"))["lever"].to_list()
    raise SystemExit(f"combination requested with adopted={adopted}; not built (see R7)")


def lever_path(lever: str, weeks: Sequence[date]) -> Path:
    seed = int(lever.removeprefix("b4_seed")) if lever.startswith("b4_seed") else CHAMPION_SEED
    return GEN / f"phase_b_{lever}_seed{seed}_{_stamp(weeks)}.parquet"


def score_frame(eval_set: pl.DataFrame, preds: pl.DataFrame, name: str) -> pl.DataFrame:
    """Per-origin metrics of `preds` on the champion's eval set (same rows, same y_true)."""
    j = eval_set.join(preds, on=["style_key", "origin_week"], how="left")
    if j["y_pred"].null_count():
        raise SystemExit(f"{name}: {j['y_pred'].null_count()} eval rows without a prediction")
    rows = []
    for w in j["origin_week"].unique().sort().to_list():
        s = j.filter(pl.col("origin_week") == w)
        y, p = s["y_true"].to_numpy(), s["y_pred"].to_numpy()
        m = score_predictions(y, p, s["n_active_articles_level"].to_numpy())
        m.pop("n_eval")
        rows.append(
            {"origin_week": w, "method": name, **m, **new_metric_values(y, p, MARGIN_PREREGISTERED)}
        )
    return pl.DataFrame(rows)


def paired(per_origin: pl.DataFrame, challenger: str) -> list[dict[str, object]]:
    a = per_origin.filter(pl.col("method") == challenger).sort("origin_week")
    b = per_origin.filter(pl.col("method") == "champion").sort("origin_week")
    j = a.join(b, on="origin_week", suffix="_c")
    out = []
    for m in METRICS:
        d = (j[m] - j[f"{m}_c"]).fill_nan(None).drop_nulls().to_numpy()
        c = circular_ci(d)
        out.append(
            {
                "challenger": challenger,
                "metric": m,
                "n_paired": c["n"],
                "mean_diff": c.get("mean"),
                "ci_lo": c.get("ci_lo"),
                "ci_hi": c.get("ci_hi"),
                "se": c.get("se"),
                "p_two_sided": bootstrap_p_two_sided(d) if np.any(d != 0) else 1.0,
                "identical": bool(np.all(d == 0)),
            }
        )
    return out


def guardrail_failures(pairs: pl.DataFrame, challenger: str) -> list[str]:
    fails = []
    for m, sign in GUARDRAILS.items():
        r = pairs.filter((pl.col("challenger") == challenger) & (pl.col("metric") == m)).row(
            0, named=True
        )
        if (sign > 0 and r["ci_hi"] < 0) or (sign < 0 and r["ci_lo"] > 0):
            fails.append(m)
    return fails


def b4_ensemble(eval_set: pl.DataFrame, weeks: Sequence[date]) -> tuple[pl.DataFrame, dict]:
    frames = [
        pl.read_parquet(champion_path(weeks)).select(
            "style_key", "origin_week", pl.col("y_pred_lightgbm").alias("s42")
        )
    ]
    for s in B4_SEEDS[1:]:
        frames.append(pl.read_parquet(lever_path(f"b4_seed{s}", weeks)).rename({"y_pred": f"s{s}"}))
    j = frames[0]
    for f in frames[1:]:
        j = j.join(f, on=["style_key", "origin_week"], how="inner")
    mat = j.select([f"s{s}" for s in B4_SEEDS]).to_numpy()
    identical = bool(np.array_equal(mat, np.repeat(mat[:, [0]], mat.shape[1], axis=1)))
    # With bit-identical seeds the SD is exactly 0; np.std would return ~1e-16 float noise and
    # turn "no seed variance" into a spurious 90% reduction, so the identical case is explicit.
    sd = np.zeros(mat.shape[0]) if identical else mat.std(axis=1, ddof=1)
    var_single = float(np.mean(sd**2))
    var_ens = var_single / len(B4_SEEDS)  # variance of the mean of independent seed components
    stats = {
        "n_rows": int(mat.shape[0]),
        "seeds": list(B4_SEEDS),
        "seeds_bit_identical": identical,
        "max_abs_diff_vs_seed42": float(np.max(np.abs(mat - mat[:, [0]]))),
        "mean_across_seed_sd": float(sd.mean()),
        "var_single_seed_component": var_single,
        "var_ensemble_seed_component": var_ens,
        "variance_reduction": None if var_single == 0 else 1 - var_ens / var_single,
    }
    ens_pred = mat[:, 0] if identical else mat.mean(axis=1)
    ens = j.select("style_key", "origin_week", pl.Series("y_pred", ens_pred))
    return ens, stats


def evaluate() -> None:
    panel = pl.read_parquet(PANEL)
    _, weekly = weekly_origins(panel)
    weeks = [o.origin_week for o in weekly]
    champ = load_champion()
    eval_set = champ.select("style_key", "origin_week", "y_true", "n_active_articles_level")

    repro = pl.read_parquet(lever_path("repro", weeks))
    jr = champ.join(repro, on=["style_key", "origin_week"], suffix="_r")
    diff = float(np.max(np.abs(jr["y_pred"].to_numpy() - jr["y_pred_r"].to_numpy())))
    if jr.height != champ.height or diff > 1e-9:
        raise SystemExit(f"training path does not reproduce the champion: max diff {diff}")
    print(f"training path reproduces champion: {jr.height} rows, max |diff| {diff:.2e}")

    per = [score_frame(eval_set, champ.select("style_key", "origin_week", "y_pred"), "champion")]
    phase_a = pl.read_csv("reports/tables/phase_a_per_origin.csv", try_parse_dates=True).filter(
        pl.col("method") == "lightgbm"
    )
    chk = per[0].join(phase_a, on="origin_week", suffix="_a")
    assert all(np.allclose(chk[m], chk[f"{m}_a"], atol=1e-12, equal_nan=True) for m in METRICS)

    ens, b4_stats = b4_ensemble(eval_set, weeks)
    (Path(OUT) / "phase_b4_seed_variance.json").write_text(json.dumps(b4_stats, indent=2))
    challengers = {
        "b1": pl.read_parquet(lever_path("b1", weeks)),
        "b1_shuffle": pl.read_parquet(lever_path("b1_shuffle", weeks)),
        "b1_random": pl.read_parquet(lever_path("b1_random", weeks)),
        "b2": pl.read_parquet(lever_path("b2", weeks)),
        "b3": pl.read_parquet(lever_path("b3", weeks)),
        "b4": ens,
        "b5_q50": pl.read_parquet(lever_path("b5_q50", weeks)),
    }
    for name, p in challengers.items():
        per.append(score_frame(eval_set, p.select("style_key", "origin_week", "y_pred"), name))
    per_origin = pl.concat(per)
    per_origin.write_csv(f"{OUT}/phase_b_per_origin.csv")

    pairs = pl.DataFrame([r for c in challengers for r in paired(per_origin, c)])
    pairs.write_csv(f"{OUT}/phase_b_paired.csv")

    prim = pairs.filter(pl.col("metric") == PRIMARY)
    pvals = {c: prim.filter(pl.col("challenger") == c)["p_two_sided"][0] for c in HOLM_LEVERS}
    holm = holm_decisions(pvals)
    order = sorted(pvals, key=lambda k: pvals[k])
    thresholds = dict(zip(order, [0.05 / (4 - i) for i in range(4)], strict=True))
    controls_ok = all(
        prim.filter(pl.col("challenger") == c)["ci_lo"][0] <= 0 for c in ("b1_shuffle", "b1_random")
    )
    rows = []
    for c in HOLM_LEVERS:
        r = prim.filter(pl.col("challenger") == c).row(0, named=True)
        fails = guardrail_failures(pairs, c)
        adopted = holm[c] and r["mean_diff"] > 0 and not fails and (c != "b1" or controls_ok)
        rows.append(
            {
                "lever": c,
                "mean_diff": r["mean_diff"],
                "ci_lo": r["ci_lo"],
                "ci_hi": r["ci_hi"],
                "p_two_sided": r["p_two_sided"],
                "holm_rank": order.index(c) + 1,
                "holm_threshold": thresholds[c],
                "holm_significant": holm[c],
                "guardrail_failures": ",".join(fails),
                "controls_ok": controls_ok if c == "b1" else None,
                "adopted": adopted,
            }
        )
    decisions = pl.DataFrame(rows)
    decisions.write_csv(f"{OUT}/phase_b_decisions.csv")

    b5 = b5_coverage(eval_set, weeks)
    q50 = prim.filter(pl.col("challenger") == "b5_q50").row(0, named=True)
    pooled = float(b5["covered"].mean())
    per_o = b5.group_by("origin_week").agg(pl.col("covered").mean()).sort("origin_week")
    per_o.write_csv(f"{OUT}/phase_b5_coverage_per_origin.csv")
    ci = circular_ci(per_o["covered"].to_list())
    b5_summary = {
        "pooled_coverage": pooled,
        "per_origin_mean": ci["mean"],
        "per_origin_ci": [ci["ci_lo"], ci["ci_hi"]],
        "per_origin_min": float(per_o["covered"].min()),
        "per_origin_max": float(per_o["covered"].max()),
        "crossing_rate": float(b5["crossed"].mean()),
        "coverage_in_band": 0.75 <= pooled <= 0.85,
        "q50_capture20_diff": [q50["mean_diff"], q50["ci_lo"], q50["ci_hi"]],
        "q50_non_inferior": not (q50["ci_hi"] < 0),
    }
    b5_summary["accepted"] = b5_summary["coverage_in_band"] and b5_summary["q50_non_inferior"]
    (Path(OUT) / "phase_b5_summary.json").write_text(json.dumps(b5_summary, indent=2))

    with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=240, float_precision=4):
        print(decisions)
        print(
            pairs.filter(pl.col("metric").is_in([PRIMARY, *GUARDRAILS])).select(
                "challenger", "metric", "mean_diff", "ci_lo", "ci_hi", "p_two_sided", "identical"
            )
        )
    print(json.dumps(b4_stats, indent=2))
    print(json.dumps(b5_summary, indent=2))


def b5_coverage(eval_set: pl.DataFrame, weeks: Sequence[date]) -> pl.DataFrame:
    lo = pl.read_parquet(lever_path("b5_q10", weeks)).rename({"y_pred": "q10"})
    hi = pl.read_parquet(lever_path("b5_q90", weeks)).rename({"y_pred": "q90"})
    j = eval_set.join(lo, on=["style_key", "origin_week"]).join(hi, on=["style_key", "origin_week"])
    return j.with_columns(
        (pl.col("q10") > pl.col("q90")).alias("crossed"),
        ((pl.col("q10") <= pl.col("y_true")) & (pl.col("y_true") <= pl.col("q90"))).alias(
            "covered"
        ),
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        run(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "evaluate":
        evaluate()
    else:
        raise SystemExit(__doc__)
