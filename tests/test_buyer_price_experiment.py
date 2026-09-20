from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from nss.models.buyer_price_experiment import (
    HYPOTHESIS_FEATURES,
    decide,
    decision_table,
    hypothesis_table,
)
from nss.models.embargoed_retune import config_label, select_config


def _paired(top20: tuple[float, float], ndcg: tuple[float, float], rho: tuple[float, float]):
    """Pooled treatment-minus-control rows: each tuple is (mean_diff, ci_low)."""
    rows = [
        {"split": "pooled", "metric": m, "mean_diff": d, "diff_ci_low": lo, "diff_ci_high": lo + 1}
        for m, (d, lo) in (
            ("hit_at_3_in_top20", top20),
            ("ndcg_at_10", ndcg),
            ("spearman_rho", rho),
        )
    ]
    return pl.DataFrame(rows)


def test_rule_a_adopts_on_top20_ci_above_zero() -> None:
    v = decide(_paired((0.1, 0.02), (-0.01, -0.05), (0.0, -0.01)))
    assert v["rule_a"] and not v["rule_b"] and v["decision"] == "ADOPT"


def test_rule_b_adopts_when_ndcg_and_spearman_cis_exclude_zero_and_top20_not_worse() -> None:
    v = decide(_paired((0.0, -0.1), (0.02, 0.005), (0.01, 0.002)))
    assert not v["rule_a"] and v["rule_b"] and v["decision"] == "ADOPT"


def test_rule_b_needs_top20_point_estimate_not_worse() -> None:
    v = decide(_paired((-0.03, -0.1), (0.02, 0.005), (0.01, 0.002)))
    assert v["decision"] == "REJECT"


def test_rule_b_needs_both_ndcg_and_spearman() -> None:
    assert decide(_paired((0.05, -0.05), (0.02, 0.005), (0.01, -0.001)))["decision"] == "REJECT"
    assert decide(_paired((0.05, -0.05), (0.02, -0.001), (0.01, 0.002)))["decision"] == "REJECT"


def test_ci_touching_zero_is_not_excluding_zero() -> None:
    assert decide(_paired((0.1, 0.0), (0.02, 0.0), (0.01, 0.0)))["decision"] == "REJECT"


def test_decision_table_carries_the_rule_text_and_verdict_on_every_row() -> None:
    from nss.models.metrics import METRIC_KEYS

    rows = []
    for metric in METRIC_KEYS:
        rows.append(
            {
                "split": "pooled",
                "metric": metric,
                "mean_diff": 0.0,
                "diff_ci_low": -1.0,
                "diff_ci_high": 1.0,
            }
        )
    paired = pl.DataFrame(rows)
    per_origin = pl.DataFrame(
        [
            {"method": m, **{k: 0.5 for k in METRIC_KEYS}}
            for m in ("lightgbm_control", "lightgbm_treatment_u")
        ]
    )
    table = decision_table(paired, per_origin)
    assert table.height == len(METRIC_KEYS)
    assert set(table["decision"]) == {"REJECT"}
    assert table["rule_text"].n_unique() == 1 and "ADOPT iff" in table["rule_text"][0]


def _grid(rmses: dict[int, list[float]]) -> pl.DataFrame:
    rows = []
    for grid_index, values in rmses.items():
        for val_index, rmse in zip((4, 5, 6, 7), values, strict=True):
            rows.append(
                {
                    "config": f"c{grid_index}",
                    "grid_index": grid_index,
                    "val_index": val_index,
                    "val_rmse": rmse,
                }
            )
    return pl.DataFrame(rows)


def test_select_config_picks_lowest_mean_and_breaks_ties_by_grid_order() -> None:
    out = select_config(_grid({0: [1, 1, 1, 1], 1: [0.5, 1.5, 1, 1], 2: [2, 2, 2, 2]}))
    assert out["config"].to_list() == ["c0", "c1", "c2"]  # c0 and c1 tie at 1.0 -> grid order
    assert out["rank"].to_list() == [1, 2, 3]
    assert out.filter(pl.col("config") == "c1")["strict_single_origin_rmse"].item() == 0.5


def test_config_label_is_stable() -> None:
    label = config_label(
        {"num_leaves": 63, "learning_rate": 0.05, "n_estimators": 200, "min_child_samples": 50}
    )
    assert label == "nl63_lr0.05_n200_mcs50"


def _shap_table(rho_by_feature: dict[str, float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature": list(rho_by_feature),
            "rank": list(range(1, len(rho_by_feature) + 1)),
            "mean_abs_shap": [0.1] * len(rho_by_feature),
            "shap_spearman": list(rho_by_feature.values()),
        }
    )


def _test_frame(seed: int = 0) -> tuple[pl.DataFrame, list[date]]:
    rng = np.random.default_rng(seed)
    weeks = [date(2019, 8, 1) + timedelta(weeks=4 * i) for i in range(12)]
    n = 60
    rows = []
    for week in weeks:
        base = {name: rng.normal(size=n) for name in HYPOTHESIS_FEATURES}
        rows.append(
            pl.DataFrame(
                {
                    "origin_week": [week] * n,
                    "y_true": rng.normal(size=n).clip(min=0),
                    "ewma_halflife_13w": rng.uniform(0.5, 2, n),
                    **base,
                }
            )
        )
    return pl.concat(rows), weeks


def test_hypothesis_verdict_thresholds() -> None:
    frame, weeks = _test_frame()
    agree = {f: 0.3 * s for f, s in HYPOTHESIS_FEATURES.items()}
    table, verdict = hypothesis_table(_shap_table(agree), frame, weeks)
    assert verdict == "SUPPORTED" and int(table["shap_consistent"].sum()) == 10
    flipped = {f: -0.3 * s for f, s in HYPOTHESIS_FEATURES.items()}
    assert hypothesis_table(_shap_table(flipped), frame, weeks)[1] == "CONTRADICTED"
    names = list(HYPOTHESIS_FEATURES)
    mixed = {f: 0.3 * HYPOTHESIS_FEATURES[f] * (1 if i < 5 else -1) for i, f in enumerate(names)}
    assert hypothesis_table(_shap_table(mixed), frame, weeks)[1] == "MIXED"
    weak = {f: 0.01 * s for f, s in HYPOTHESIS_FEATURES.items()}  # right sign, |rho| < 0.05
    assert hypothesis_table(_shap_table(weak), frame, weeks)[1] == "CONTRADICTED"
