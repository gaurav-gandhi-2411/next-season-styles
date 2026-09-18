from __future__ import annotations

import polars as pl

from nss.models.final_three_shap_verdict import build_verdict_table

_SHAP_COLS = [
    c for i in range(1, 6) for c in (f"shap_driver_{i}_feature", f"shap_driver_{i}_value")
]


def _row(
    style_key: str,
    source_table: str,
    predicted_intensity: float,
    drivers: list[tuple[str, float]],
) -> dict[str, object]:
    """Build one `top_styles_final_three.csv`-shaped row from a top-5 `(feature, value)` list."""
    assert len(drivers) == 5
    row: dict[str, object] = {
        "style_key": style_key,
        "source_table": source_table,
        "predicted_intensity": predicted_intensity,
    }
    for i, (feature, value) in enumerate(drivers, start=1):
        row[f"shap_driver_{i}_feature"] = feature
        row[f"shap_driver_{i}_value"] = value
    return row


def test_dominant_mechanism_persistence_wins_when_lag_1_exceeds_seasonal() -> None:
    """lag_1 (0.75) vs. the largest seasonal term fourier_sin_1 (0.10) -- lag_1 must win, and the
    verdict text must name both values (hand-computed from the T1 style's real saved SHAP row)."""
    final_three = pl.DataFrame(
        [
            _row(
                "STYLE_A",
                "T1_incumbent",
                33.69,
                [
                    ("lag_1", 0.7451),
                    ("n_active_articles_level", 0.2130),
                    ("perceived_colour_master_name", 0.1196),
                    ("fourier_sin_1", -0.1043),
                    ("garment_group_name", 0.1011),
                ],
            )
        ]
    )

    verdict = build_verdict_table(final_three)
    row = verdict.row(0, named=True)

    assert "persistence (lag_1) dominates" in row["dominant_mechanism"]
    assert "0.7451" in row["dominant_mechanism"]
    assert "fourier_sin_1" in row["dominant_mechanism"]
    assert row["absolute_predicted_intensity"] == 33.69
    assert row["rank_among_guard_passing_styles"].startswith("1 --")


def test_dominant_mechanism_seasonal_wins_when_it_exceeds_lag_1() -> None:
    """Synthetic case where a seasonal term (fourier_cos_2=0.9) outmagnitudes lag_1 (0.2) --
    verdict must say seasonal recovery dominates, not persistence."""
    final_three = pl.DataFrame(
        [
            _row(
                "STYLE_B",
                "T2_emerging",
                10.0,
                [
                    ("fourier_cos_2", 0.9),
                    ("lag_1", 0.2),
                    ("n_active_articles_level", 0.15),
                    ("slope_13w", 0.05),
                    ("share_garment_group", 0.02),
                ],
            )
        ]
    )

    verdict = build_verdict_table(final_three)
    row = verdict.row(0, named=True)

    assert row["dominant_mechanism"].startswith("seasonal recovery (fourier_cos_2) dominates")
    assert "0.9000" in row["dominant_mechanism"]
    assert row["rank_among_guard_passing_styles"].startswith(">= guard-passing-population median")


def test_dominant_mechanism_neither_when_lag_1_and_seasonal_both_absent() -> None:
    """No lag_1, no seasonal feature in the top-5 -- verdict must name the true top driver plainly
    rather than forcing a seasonal-vs-persistence answer."""
    final_three = pl.DataFrame(
        [
            _row(
                "STYLE_C",
                "T2_emerging",
                18.97,
                [
                    ("n_active_articles_level", 0.2696),
                    ("perceived_colour_master_name", 0.0656),
                    ("n_active_articles_trend_13w", 0.0618),
                    ("slope_13w", 0.03),
                    ("share_garment_group", 0.02),
                ],
            )
        ]
    )

    verdict = build_verdict_table(final_three)
    row = verdict.row(0, named=True)

    assert "neither lag_1" in row["dominant_mechanism"]
    assert "n_active_articles_level" in row["dominant_mechanism"]


def test_dominant_mechanism_persistence_wins_but_a_third_feature_is_the_true_top_driver() -> None:
    """Hand-computed from the real T2-sweater style row: lag_1 (0.1771) beats the largest seasonal
    term fourier_sin_1 (0.0595), but n_active_articles_level (0.2696) is the actual top-1 driver --
    the verdict must say persistence wins the lag_1-vs-seasonal comparison AND separately name
    n_active_articles_level as the true overall top driver, not silently drop it."""
    final_three = pl.DataFrame(
        [
            _row(
                "STYLE_D",
                "T2_emerging",
                18.97,
                [
                    ("n_active_articles_level", 0.2696),
                    ("lag_1", 0.1771),
                    ("perceived_colour_master_name", 0.0656),
                    ("n_active_articles_trend_13w", 0.0618),
                    ("fourier_sin_1", 0.0595),
                ],
            )
        ]
    )

    verdict = build_verdict_table(final_three)
    row = verdict.row(0, named=True)

    assert "persistence (lag_1) dominates over seasonal" in row["dominant_mechanism"]
    assert "n_active_articles_level" in row["dominant_mechanism"]
    assert "0.2696" in row["dominant_mechanism"]


def test_build_verdict_table_preserves_input_row_order_and_shap_columns() -> None:
    """Multi-row input: output preserves row order and copies every shap_driver_{i}_{feature,value}
    column through unchanged."""
    drivers = [
        ("lag_1", 0.5),
        ("lag_2", 0.4),
        ("lag_3", 0.3),
        ("lag_4", 0.2),
        ("lag_5", 0.1),
    ]
    final_three = pl.DataFrame(
        [
            _row("FIRST", "T1_incumbent", 1.0, drivers),
            _row("SECOND", "T2_emerging", 2.0, drivers),
        ]
    )

    verdict = build_verdict_table(final_three)

    assert verdict["style_key"].to_list() == ["FIRST", "SECOND"]
    for i in range(1, 6):
        assert verdict[f"shap_driver_{i}_feature"].to_list() == [drivers[i - 1][0]] * 2
        assert verdict[f"shap_driver_{i}_value"].to_list() == [drivers[i - 1][1]] * 2
