"""C4 verification: read-only SHAP-driver + commercial-scale report for the final three styles
(`reports/tables/top_styles_final_three.csv`, Phase 2.5's A7 diversity-constrained reselection).

READ-ONLY, NO MODEL CHANGES: every value this module reports (top-5 SHAP drivers, `predicted_
intensity`) is already persisted in `top_styles_final_three.csv` by `nss.models.diversity_forecast`
-- nothing here retrains a model, recomputes SHAP, or re-runs any inference. This module only reads
that CSV, classifies each style's already-saved top-5 drivers into a dominant-mechanism verdict
(seasonal-recovery vs. short-term persistence vs. neither), and writes a small report table.

DOMINANT-MECHANISM CLASSIFICATION: compares the SHAP value of `lag_1` (short-term persistence)
against the largest-magnitude SHAP value among `SEASONAL_FEATURES` (`fourier_sin_1`, `fourier_
cos_1`, `fourier_sin_2`, `fourier_cos_2`, `lag_52`), restricted to whichever of those actually
appear in that style's saved top-5. If neither `lag_1` nor any seasonal feature appears in the
top-5 at all, or if the single largest-magnitude driver overall is neither, the verdict says so
plainly rather than forcing a seasonal-vs-persistence answer where the data doesn't support one --
see module docstring of the C4 task instructions this implements.

RANK AMONG ALL GUARD-PASSING STYLES: exactly derivable for a T1_incumbent-sourced row ONLY, by a
pure code-logic argument (no data needed beyond `source_table` and `rank`==1): `nss.models.
diversity_forecast.select_t1_incumbent` sorts every guard-passing style descending by `predicted_
intensity`, then diversity-walks top-down, skipping only rows whose `DIVERSITY_KEY_COLS` pair
collides with an ALREADY-KEPT row. The walk's first kept row (T1 rank 1) is, by construction,
the single highest-`predicted_intensity` row in the entire guard-passing population -- nothing has
been kept yet when it's considered, so it can never be skipped. `T2_emerging` rows are ranked by
`growth_ratio`, not `predicted_intensity`, so no such proof applies to them; their exact rank would
require the full guard-passing population's `predicted_intensity` ranking, which is not persisted
anywhere on disk (`top_styles.csv`/`top_styles_t1_incumbent.csv` are each only a top-10 slice, and
`top_styles.csv` additionally predates the A5 determinism fix -- see that commit -- so it reflects
a DIFFERENT, non-reproducible model realization and cannot be validly combined with this run's
numbers). Computing it exactly would require re-deriving `build_ranking_frame`, which needs a
trained model; no trained model artifact is persisted to disk, so doing so would mean retraining --
out of scope under this task's HARD CONSTRAINT (modelling frozen). This module instead reports the
provable BOUND for T2 rows (>= the guard-passing population median, per `nss.models.
diversity_forecast.t2_absolute_intensity_floor`'s eligibility rule) and states the gap explicitly
rather than silently omitting it or retraining to fill it in.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

SEASONAL_FEATURES: tuple[str, ...] = (
    "fourier_sin_1",
    "fourier_cos_1",
    "fourier_sin_2",
    "fourier_cos_2",
    "lag_52",
)
PERSISTENCE_FEATURE = "lag_1"
N_SHAP_DRIVERS = 5

DEFAULT_FINAL_THREE_PATH = Path("reports/tables/top_styles_final_three.csv")
DEFAULT_OUT_PATH = Path("reports/tables/final_three_shap_verdict.csv")

# See module docstring RANK AMONG ALL GUARD-PASSING STYLES.
_T1_RANK_NOTE = (
    "1 -- proven by selection-code logic (select_t1_incumbent's diversity walk keeps its first "
    "row unconditionally, so T1 rank 1 is always the single highest predicted_intensity among ALL "
    "guard-passing styles); zero computation/retraining required"
)
_T2_RANK_NOTE = (
    ">= guard-passing-population median (top 50%) by construction (t2_absolute_intensity_floor); "
    "exact rank not computable from persisted artifacts -- would require retraining the final "
    "model to re-derive the full guard-passing ranking, out of scope under the modelling freeze"
)


def _overall_driver_note(top_feature: str, top_value: float) -> str:
    """A short parenthetical: confirms `lag_1` is also the single largest driver overall, or, when
    it ISN'T, names the actual largest driver explicitly (so a lag_1-vs-seasonal verdict never
    silently hides a THIRD feature -- e.g. `n_active_articles_level` -- being the real top driver).
    """
    if top_feature == PERSISTENCE_FEATURE:
        return " (also the single largest driver overall)"
    return f" (though the single largest driver overall is {top_feature}, SHAP={top_value:.4f})"


def _dominant_mechanism(drivers: list[tuple[str, float]]) -> str:
    """Classify a style's top-5 SHAP drivers as persistence-dominated, seasonal-dominated, or
    neither. See module docstring DOMINANT-MECHANISM CLASSIFICATION.

    Args:
        drivers: `[(feature_name, shap_value), ...]` in the style's saved top-5 rank order
            (`shap_driver_1_feature`/`_value` first).

    Returns:
        A human-readable verdict sentence naming the top overall driver, whether `lag_1` or a
        seasonal feature is present, and which (if either) dominates.
    """
    by_feature = dict(drivers)
    top_feature, top_value = drivers[0]

    lag_1_value = by_feature.get(PERSISTENCE_FEATURE)
    seasonal_present = [(f, v) for f, v in drivers if f in SEASONAL_FEATURES]
    seasonal_top = max(seasonal_present, key=lambda fv: abs(fv[1])) if seasonal_present else None

    if lag_1_value is None and seasonal_top is None:
        return (
            f"neither lag_1 (persistence) nor any seasonal feature appears in the top-5; top "
            f"driver is {top_feature} (SHAP={top_value:.4f})"
        )
    if lag_1_value is not None and seasonal_top is None:
        overall_note = _overall_driver_note(top_feature, top_value)
        return (
            f"persistence (lag_1) dominates: SHAP={lag_1_value:.4f}{overall_note}; no seasonal "
            f"fourier/lag_52 term appears in the top-5 at all"
        )
    if lag_1_value is None and seasonal_top is not None:
        s_feature, s_value = seasonal_top
        overall_note = (
            "" if top_feature == s_feature else _overall_driver_note(top_feature, top_value)
        )
        return (
            f"seasonal recovery ({s_feature}) dominates: SHAP={s_value:.4f}{overall_note}; "
            f"lag_1 not in top-5"
        )

    assert lag_1_value is not None and seasonal_top is not None
    s_feature, s_value = seasonal_top
    if abs(lag_1_value) >= abs(s_value):
        overall_note = _overall_driver_note(top_feature, top_value)
        return (
            f"persistence (lag_1) dominates over seasonal: lag_1 SHAP={lag_1_value:.4f} vs. "
            f"{s_feature} SHAP={s_value:.4f}{overall_note}"
        )
    overall_note = "" if top_feature == s_feature else _overall_driver_note(top_feature, top_value)
    return (
        f"seasonal recovery ({s_feature}) dominates over persistence: {s_feature} "
        f"SHAP={s_value:.4f} vs. lag_1 SHAP={lag_1_value:.4f}{overall_note}"
    )


def build_verdict_table(final_three: pl.DataFrame) -> pl.DataFrame:
    """Build the C4 verdict table from `top_styles_final_three.csv`'s already-saved columns.

    Args:
        final_three: `top_styles_final_three.csv` read as-is (must have `style_key`,
            `source_table`, `predicted_intensity`, and `shap_driver_{1..5}_{feature,value}`).

    Returns:
        One row per input style, columns: `style_key`, `source_table`, `shap_driver_{1..5}_feature`,
        `shap_driver_{1..5}_value`, `dominant_mechanism`, `absolute_predicted_intensity`,
        `rank_among_guard_passing_styles`.
    """
    rows: list[dict[str, object]] = []
    for row in final_three.iter_rows(named=True):
        drivers = [
            (row[f"shap_driver_{i}_feature"], float(row[f"shap_driver_{i}_value"]))
            for i in range(1, N_SHAP_DRIVERS + 1)
        ]
        out: dict[str, object] = {
            "style_key": row["style_key"],
            "source_table": row["source_table"],
        }
        for i, (feature, value) in enumerate(drivers, start=1):
            out[f"shap_driver_{i}_feature"] = feature
            out[f"shap_driver_{i}_value"] = value
        out["dominant_mechanism"] = _dominant_mechanism(drivers)
        out["absolute_predicted_intensity"] = float(row["predicted_intensity"])
        out["rank_among_guard_passing_styles"] = (
            _T1_RANK_NOTE if row["source_table"] == "T1_incumbent" else _T2_RANK_NOTE
        )
        rows.append(out)
    return pl.DataFrame(rows)


def main() -> None:
    """CLI entry point: read `top_styles_final_three.csv`, write `final_three_shap_verdict.csv`."""
    final_three = pl.read_csv(DEFAULT_FINAL_THREE_PATH)
    verdict = build_verdict_table(final_three)
    DEFAULT_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    verdict.write_csv(DEFAULT_OUT_PATH)
    print(f"Wrote {DEFAULT_OUT_PATH} ({verdict.height} rows)")
    for row in verdict.iter_rows(named=True):
        print(f"  {row['style_key']} [{row['source_table']}]: {row['dominant_mechanism']}")


if __name__ == "__main__":
    main()
