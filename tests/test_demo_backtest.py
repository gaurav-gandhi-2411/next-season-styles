from __future__ import annotations

import polars as pl
import pytest

from nss import demo_backtest


def test_committed_fixture_matches_generator() -> None:
    """The committed fixture is what `make_toy_sparse_panel(42)` produces (no silent drift).

    Floats are compared approximately: exp/sin can differ in the last ulp across platforms, which
    is not a reason to fail the test.
    """
    committed = pl.read_parquet(demo_backtest.FIXTURE_PATH)
    fresh = demo_backtest.make_toy_sparse_panel()
    assert committed.columns == fresh.columns
    assert committed.height == fresh.height
    for column in committed.columns:
        if committed[column].dtype.is_float():
            assert committed[column].to_numpy() == pytest.approx(fresh[column].to_numpy())
        else:
            assert committed[column].to_list() == fresh[column].to_list()


def test_toy_panel_has_the_real_panels_shape() -> None:
    """Same extent as the real panel, so the origin schedule is the reported one (20 / 12)."""
    dense = demo_backtest.build_dense_panel(pl.read_parquet(demo_backtest.FIXTURE_PATH))
    origins = demo_backtest.generate_origin_schedule(dense)
    assert len(origins) == 20
    assert len(origins[demo_backtest.INITIAL_POOL_SIZE :]) == 12
    assert dense["style_key"].n_unique() == demo_backtest.N_STYLES


def test_demo_runs_end_to_end_and_beats_the_random_floor(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drives the real harness through `main()`: every method is reported, floor is beaten."""
    monkeypatch.setattr("sys.argv", ["demo_backtest"])
    demo_backtest.main()
    out = capsys.readouterr().out

    assert "SYNTHETIC DATA" in out
    assert "OK for all 12 test origins" in out
    for method in (*demo_backtest.BASELINE_METHODS, "lightgbm", "random_floor"):
        assert method in out

    # The floor comparison, recomputed rather than scraped from the printed table.
    dense = demo_backtest.build_dense_panel(pl.read_parquet(demo_backtest.FIXTURE_PATH))
    origins = demo_backtest.generate_origin_schedule(dense)
    test_origins = origins[demo_backtest.INITIAL_POOL_SIZE :]
    embargoed, skipped = demo_backtest.run_embargoed_walk_forward(
        dense, origins, demo_backtest.DEMO_LGBM_CONFIG
    )
    assert not skipped
    predictions = demo_backtest.build_predictions_frame(
        dense, [o.origin_week for o in test_origins]
    )
    floor = demo_backtest.aggregate_random_floor_over_seeds(
        demo_backtest.score_random_floor_per_origin(predictions, test_origins)
    )
    assert embargoed["hit_at_3_in_top20"].mean() > floor["hit_at_3_in_top20"].mean()
