"""Label-shuffle control: how do we know the headline is not leakage?

The forward target is randomly permuted ACROSS styles WITHIN each origin (the target's marginal
distribution per origin is preserved; only the style<->target correspondence is destroyed). The L2
LightGBM model is then retrained through the SAME expanding-window walk-forward path as the
committed model (`nss.models.lightgbm_model`: same features, same locked config, same 12 test
origins) and scored with the SAME `score_predictions`.

Three rows per seed, so the control can be read without trusting any single design choice:

* `shuffled_train_true_test` (primary): TRAINING labels shuffled, test origins scored against their
  TRUE labels. A model that learned nothing must land on the random floor; if it does not, either
  the features/harness leak label information or the metric machinery favours some ordering.
* `shuffled_train_shuffled_test`: the literal reading -- the test labels are permuted as well.
* `unshuffled` (positive control, seed-independent): the same scratch code with real labels must
  reproduce the committed headline (0.722), proving this path is the committed harness.

Everything is computed in memory; the ONLY file written is `OUT_PATH`. No committed model or table
is touched (the committed LightGBM / random-floor numbers are read back for comparison only).

Usage:
    uv run python -m nss.models.label_shuffle_control
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from nss.models.backtest import WMAPE_WEIGHT_COL, generate_origin_schedule
from nss.models.final_forecast import FINAL_MODEL_CONFIG
from nss.models.lightgbm_model import (
    INITIAL_POOL_SIZE,
    build_model_frame,
    feature_columns,
    predict_lightgbm,
    train_lightgbm,
)
from nss.models.metrics import score_predictions

PANEL_PATH = Path("data/processed/style_week_panel.parquet")
OUT_PATH = Path("reports/tables/label_shuffle_control.csv")
COMMITTED_SUMMARY = Path("reports/tables/backtest_summary_v2.csv")
SHUFFLE_SEEDS = (42, 43, 44)  # repo-wide seed 42, then its two successors
REPORT_METRICS = (
    "hit_at_3_in_top20",
    "hit_at_3_in_top10",
    "precision_at_3",
    "spearman_rho",
    "ndcg_at_10",
)
N_PICKS_PER_ORIGIN = 3  # Hit@3: each origin contributes 3 top-3 picks


def shuffle_within_origin(frame: pl.DataFrame, seed: int) -> np.ndarray:
    """`y_true` permuted across styles within each origin (row order of `frame` preserved).

    Each origin gets its own generator (`seed`, origin ordinal) so the permutation is reproducible
    and independent across origins. The multiset of targets per origin is unchanged.
    """
    y = frame["y_true"].to_numpy().astype(np.float64).copy()
    weeks = frame["origin_week"].to_list()
    out = y.copy()
    for ordinal, week in enumerate(sorted(set(weeks))):
        idx = np.flatnonzero(np.array([w == week for w in weeks]))
        rng = np.random.default_rng([seed, ordinal])
        out[idx] = rng.permutation(y[idx])
    return out


def walk_forward_scores(
    frame: pl.DataFrame,
    train_y: np.ndarray,
    test_y_sets: dict[str, np.ndarray],
    origin_weeks: list[date],
    columns: list[str],
) -> dict[str, list[dict[str, float]]]:
    """One expanding-window pass: train on `train_y` labels, score against each `test_y_sets` entry.

    Mirrors `nss.models.lightgbm_model.run_lightgbm_walk_forward` (train on every origin strictly
    before the test origin, test origins are `origin_weeks[INITIAL_POOL_SIZE:]`), but with the
    labels swapped in. Returns `{test_label_name: [per-origin metric dict, ...]}`.
    """
    trainable = frame.with_columns(pl.Series("y_true", train_y))
    results: dict[str, list[dict[str, float]]] = {k: [] for k in test_y_sets}
    for t in range(INITIAL_POOL_SIZE, len(origin_weeks)):
        train_weeks = set(origin_weeks[:t])
        test_week = origin_weeks[t]
        train_mask = frame["origin_week"].is_in(train_weeks).to_numpy()
        test_mask = (frame["origin_week"] == test_week).to_numpy()
        model = train_lightgbm(trainable.filter(pl.Series(train_mask)), FINAL_MODEL_CONFIG, columns)
        test_frame = frame.filter(pl.Series(test_mask))
        preds = predict_lightgbm(model, test_frame, columns)
        weight = test_frame[WMAPE_WEIGHT_COL].to_numpy()
        for name, y_full in test_y_sets.items():
            m = score_predictions(y_full[test_mask], preds, weight)
            m.pop("n_eval")
            results[name].append(m)
    return results


def _pooled(per_origin: list[dict[str, float]]) -> dict[str, float]:
    return {k: float(np.mean([r[k] for r in per_origin])) for k in REPORT_METRICS}


def committed_reference() -> list[dict[str, object]]:
    """Committed pooled LightGBM and random-floor rows (read only), for side-by-side comparison."""
    summary = pl.read_csv(COMMITTED_SUMMARY).filter(pl.col("split") == "pooled")
    rows: list[dict[str, object]] = []
    for method in ("lightgbm", "random_floor"):
        r = summary.filter(pl.col("method") == method).to_dicts()[0]
        rows.append(
            {
                "variant": f"committed_{method}",
                "seed": None,
                **{m: r[f"{m}_mean"] for m in REPORT_METRICS},
            }
        )
    return rows


def main() -> None:
    """Run the control (3 seeds + positive control) and write `OUT_PATH`."""
    panel = pl.read_parquet(PANEL_PATH)
    origins = generate_origin_schedule(panel)
    origin_weeks = [o.origin_week for o in origins]
    frame = build_model_frame(panel, origin_weeks)
    columns = feature_columns(frame)
    y_true = frame["y_true"].to_numpy().astype(np.float64)

    rows: list[dict[str, object]] = []
    real = walk_forward_scores(frame, y_true, {"true": y_true}, origin_weeks, columns)["true"]
    rows.append({"variant": "unshuffled_positive_control", "seed": None, **_pooled(real)})
    print("unshuffled:", _pooled(real))

    per_seed_hits: list[float] = []
    for seed in SHUFFLE_SEEDS:
        y_shuf = shuffle_within_origin(frame, seed)
        out = walk_forward_scores(
            frame, y_shuf, {"true": y_true, "shuffled": y_shuf}, origin_weeks, columns
        )
        rows.append({"variant": "shuffled_train_true_test", "seed": seed, **_pooled(out["true"])})
        rows.append(
            {"variant": "shuffled_train_shuffled_test", "seed": seed, **_pooled(out["shuffled"])}
        )
        per_seed_hits.append(_pooled(out["true"])["hit_at_3_in_top20"])
        print(seed, "true-test:", _pooled(out["true"]))

    primary = [r for r in rows if r["variant"] == "shuffled_train_true_test"]
    rows.append(
        {
            "variant": "shuffled_train_true_test_MEAN_OF_SEEDS",
            "seed": None,
            **{m: float(np.mean([r[m] for r in primary])) for m in REPORT_METRICS},
        }
    )
    rows.extend(committed_reference())
    pl.DataFrame(rows, infer_schema_length=None).write_csv(OUT_PATH)
    n_hits = round(sum(per_seed_hits) * len(origin_weeks[INITIAL_POOL_SIZE:]) * N_PICKS_PER_ORIGIN)
    print(f"Wrote {OUT_PATH}; hits across all seeds = {n_hits} of 108 picks")


if __name__ == "__main__":
    main()
