"""A synthetic registry that exercises the champion/challenger gate without any dataset.

Used by the unit tests and by CI (`python -m nss.prod.cli gate-fixture`). Per-origin metric
tables over 48 weekly origins are generated with a fixed seed. Challengers:

- `improves`:          capture@20 +0.05, guardrails unchanged        -> must be ACCEPTED
- `breaks_guardrail`:  capture@20 +0.05, Spearman -0.05 every origin -> must be REFUSED
- `no_effect`:         capture@20 unchanged up to noise              -> must be REFUSED
"""

from __future__ import annotations

import json
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from nss.prod.contracts import STYLE_KEY_COLS
from nss.prod.registry import PromotionRefused, Registry

N_ORIGINS = 48
METRICS = ("demand_capture_at_20", "hit_at_3_in_top20", "ndcg_at_10", "spearman_rho", "wmape")


def synthetic_per_origin(seed: int = 42) -> pl.DataFrame:
    """A champion-like per-origin table with autocorrelated noise (AR(1), 0.8)."""
    rng = np.random.default_rng(seed)
    weeks = [date(2019, 7, 29) + timedelta(weeks=i) for i in range(N_ORIGINS)]
    base = {"demand_capture_at_20": 0.6, "hit_at_3_in_top20": 0.43, "ndcg_at_10": 0.87,
            "spearman_rho": 0.71, "wmape": 0.17}  # fmt: skip
    cols = {}
    for m, level in base.items():
        e = np.zeros(N_ORIGINS)
        for i in range(1, N_ORIGINS):
            e[i] = 0.8 * e[i - 1] + rng.normal(0, 0.02)
        cols[m] = level + e
    return pl.DataFrame({"origin_week": weeks, **cols})


def _centred(rng: np.random.Generator, sd: float, n: int) -> np.ndarray:
    """Zero-mean noise. Uncentred noise has a small random mean, and with variance this small the
    gate rightly calls even a -0.0002 guardrail shift significant: the Section 6 rule has no
    minimum effect size. "Unchanged" must mean an exact zero mean difference."""
    x = rng.normal(0, sd, size=n)
    return x - x.mean()


def challenger(champion: pl.DataFrame, kind: str, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = champion.height
    noise = _centred(rng, 0.004, n)
    out = champion.with_columns(pl.col(m) + _centred(rng, 0.001, n) for m in METRICS)
    if kind in ("improves", "breaks_guardrail"):
        out = out.with_columns(pl.col("demand_capture_at_20") + 0.05 + noise)
    if kind == "breaks_guardrail":
        out = out.with_columns(pl.col("spearman_rho") - 0.05)
    if kind == "no_effect":
        out = out.with_columns(pl.col("demand_capture_at_20") + noise)
    return out


def make_artifact(root: Path, version: str, per_origin: pl.DataFrame, dirty: bool = False) -> Path:
    """The minimum an artifact needs to be registered and gated (no model files)."""
    art = root / version
    art.mkdir(parents=True)
    meta = {"version": version, "git": {"sha": "0" * 40, "dirty": dirty}, "config_hash": "fixture",
            "panel_sha256": "fixture", "checks": {"point.demand_capture_at_20": 0.0}}  # fmt: skip
    (art / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    for name in ("calibrator.json", "feature_spec.json", "reference.json"):
        (art / name).write_text("{}", encoding="utf-8")
    per_origin.write_csv(art / "backtest_per_origin.csv")
    return art


def build_fixture_registry(workdir: Path) -> Registry:
    """A registry with a promoted champion and three registered, unpromoted challengers."""
    reg = Registry(str(workdir / "registry"))
    champ = synthetic_per_origin()
    reg.register(make_artifact(workdir / "artifacts", "champion", champ))
    reg.promote("champion", initial=True, actor="fixture")
    for kind in ("breaks_guardrail", "no_effect", "improves"):
        reg.register(make_artifact(workdir / "artifacts", kind, challenger(champ, kind)))
    return reg


TOY_PANEL = Path(__file__).resolve().parents[1] / "demo_data" / "toy_sparse_panel.parquet"


def toy_panel() -> pl.DataFrame:
    """The committed 200-style demo panel, completed to the panel contract (no dataset needed).

    It has no `price_index` or `intensity_shrunk`; for tests the raw intensity stands in for the
    shrunk one and the price index is 1.0.
    """
    return pl.read_parquet(TOY_PANEL).with_columns(
        pl.col("units_per_active_article").alias("intensity_shrunk"),
        pl.lit(1.0).alias("price_index"),
    )


def build_toy_artifact(out_dir: Path, version: str = "toy") -> Path:
    """A real, loadable artifact trained on the toy panel with a small model. For API and
    inference tests in CI; its numbers mean nothing about the H&M champion."""
    import lightgbm as lgb

    from nss.models.backtest import generate_origin_schedule
    from nss.models.lightgbm_model import (
        _categorical_indices,
        _to_lgb_matrix,
        build_model_frame,
        feature_columns,
    )
    from nss.prod import INTERVAL_SEMANTICS
    from nss.prod.training import feature_spec, psi_reference

    panel = toy_panel()
    weeks = [o.origin_week for o in generate_origin_schedule(panel)][-8:]
    frame = build_model_frame(panel, weeks)
    columns = feature_columns(frame)
    X = _to_lgb_matrix(frame, columns)
    y = frame["y_true"].to_numpy()
    art = out_dir / version
    art.mkdir(parents=True)
    for name, obj, alpha in (
        ("point", "regression", None),
        ("q10", "quantile", 0.1),
        ("q90", "quantile", 0.9),
    ):
        m = lgb.LGBMRegressor(objective=obj, n_estimators=20, num_leaves=7, min_child_samples=5,
                              verbosity=-1, deterministic=True, force_row_wise=True, num_threads=1,
                              **({"alpha": alpha} if alpha else {}))  # fmt: skip
        m.fit(X, y, categorical_feature=_categorical_indices(frame, columns), feature_name=columns)
        m.booster_.save_model(str(art / f"model_{name}.txt"))
    numeric = [c for c in columns if c not in STYLE_KEY_COLS]
    files = {
        "calibrator.json": {"method": "cqr_asymmetric", "alpha": 0.2, "Q_lo": 0.1, "Q_hi": 0.1},
        "feature_spec.json": feature_spec(panel, columns),
        "reference.json": {
            "features": {c: psi_reference(frame[c].cast(pl.Float64).to_numpy()) for c in numeric},
            "prediction": psi_reference(y),
        },
        "metadata.json": {
            "version": version,
            "as_of": str(weeks[-1]),
            "git": {"dirty": False},
            "interval_semantics": INTERVAL_SEMANTICS,
            "backtest": {},
        },  # fmt: skip
    }
    for name, obj in files.items():
        (art / name).write_text(json.dumps(obj, default=str), encoding="utf-8")
    synthetic_per_origin().write_csv(art / "backtest_per_origin.csv")
    return art


def run_gate_fixture() -> int:
    """Exit 0 iff the gate refuses the two bad challengers, accepts the good one, audits all."""
    expected = {"breaks_guardrail": False, "no_effect": False, "improves": True}
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        reg = build_fixture_registry(Path(tmp))
        for kind, should_pass in expected.items():
            try:
                reg.promote(kind, actor="ci")
                accepted = True
            except PromotionRefused as e:
                accepted = False
                print(f"  refused {kind}: {e}")
            status = "OK" if accepted == should_pass else "WRONG"
            ok &= accepted == should_pass
            print(f"[{status}] {kind}: {'accepted' if accepted else 'refused'}")
        promotes = [e for e in reg.audit_log() if e["event"] == "promote"]
        audited = len(promotes) == 4 and [e["accepted"] for e in promotes] == [
            True,
            False,
            False,
            True,
        ]
        print(f"[{'OK' if audited else 'WRONG'}] audit log: {len(promotes)} promote events")
        champion_ok = reg.champion() == "improves"
        print(f"[{'OK' if champion_ok else 'WRONG'}] champion pointer: {reg.champion()}")
        ok &= audited and champion_ok
    print("gate fixture PASSED" if ok else "gate fixture FAILED")
    return 0 if ok else 1
