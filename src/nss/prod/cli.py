"""Command line for the production path.

    python -m nss.prod.cli validate --kind panel --path data/processed/style_week_panel.parquet
    python -m nss.prod.cli train    --config configs/champion.yaml --out data/artifacts
    python -m nss.prod.cli register --artifact data/artifacts/<version> [--registry ROOT]
    python -m nss.prod.cli promote  <version> [--registry ROOT] [--initial]
    python -m nss.prod.cli score    --panel PATH --as-of YYYY-MM-DD --out PATH [--registry ROOT]
    python -m nss.prod.cli gate-fixture

The registry root defaults to `$NSS_REGISTRY_ROOT`, else the config's `registry_root`. It can be
a local path or an fsspec URL (`gs://...` in C2).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date

import polars as pl

DEFAULT_CONFIG = "configs/champion.yaml"


def _registry_root(arg: str | None) -> str:
    if arg:
        return arg
    if os.environ.get("NSS_REGISTRY_ROOT"):
        return os.environ["NSS_REGISTRY_ROOT"]
    from nss.prod.config import load_config

    return load_config(DEFAULT_CONFIG).registry_root


def cmd_validate(a: argparse.Namespace) -> int:
    from nss.prod import contracts as c

    readers = {
        "articles": c.read_articles,
        "customers": c.read_customers,
        "transactions": lambda p: c.read_transactions(p).collect(),
        "panel": pl.read_parquet,
    }
    frame = readers[a.kind](a.path)
    try:
        c.validate(frame, "style_week_panel" if a.kind == "panel" else a.kind)
        if a.kind == "transactions" and a.articles:
            c.check_references(
                frame,
                c.read_articles(a.articles),
                c.read_customers(a.customers) if a.customers else None,
            )
    except c.ContractViolation as e:
        print(e, file=sys.stderr)
        if a.report:
            e.report.write_csv(a.report)
            print(f"full row-level report: {a.report}", file=sys.stderr)
        return 1
    print(f"{a.kind}: {frame.height} rows pass the contract")
    return 0


def cmd_train(a: argparse.Namespace) -> int:
    from nss.prod.config import load_config
    from nss.prod.training import ReproductionError, train

    cfg = load_config(a.config)
    try:
        art = train(cfg, a.out, date.fromisoformat(a.as_of) if a.as_of else None)
    except ReproductionError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    print(f"artifact: {art}")
    return 0


def cmd_register(a: argparse.Namespace) -> int:
    from nss.prod.registry import Registry

    version = Registry(_registry_root(a.registry)).register(a.artifact)
    print(f"registered {version}")
    return 0


def cmd_promote(a: argparse.Namespace) -> int:
    from nss.prod.registry import PromotionRefused, Registry

    try:
        decision = Registry(_registry_root(a.registry)).promote(
            a.version, initial=a.initial, actor=a.actor
        )
    except PromotionRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    print(f"promoted {a.version}: {decision['reasons']}")
    return 0


def cmd_score(a: argparse.Namespace) -> int:
    from nss.prod.inference import Predictor
    from nss.prod.monitoring import ScoringLog

    predictor = (
        Predictor.from_dir(a.artifact)
        if a.artifact
        else Predictor.from_registry(_registry_root(a.registry))
    )
    out = predictor.score(pl.read_parquet(a.panel), date.fromisoformat(a.as_of))
    out.write_parquet(a.out) if a.out.endswith(".parquet") else out.write_csv(a.out)
    if a.log_dir:
        ScoringLog(a.log_dir).append(out)
    print(f"scored {out.height} styles as of {a.as_of} with {predictor.version} -> {a.out}")
    return 0


def cmd_gate_fixture(_: argparse.Namespace) -> int:
    from nss.prod.fixture import run_gate_fixture

    return run_gate_fixture()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nss.prod.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="check a raw input or the panel against its contract")
    v.add_argument(
        "--kind", required=True, choices=["articles", "customers", "transactions", "panel"]
    )
    v.add_argument("--path", required=True)
    v.add_argument("--articles", help="articles.csv, for transactions referential integrity")
    v.add_argument("--customers", help="customers.csv, for transactions referential integrity")
    v.add_argument("--report", help="write the full row-level violation report here (CSV)")
    v.set_defaults(fn=cmd_validate)

    t = sub.add_parser("train", help="train from the config to a versioned artifact")
    t.add_argument("--config", default=DEFAULT_CONFIG)
    t.add_argument("--out", default="data/artifacts")
    t.add_argument("--as-of", help="deployable model's as-of week (default: the panel's last week)")
    t.set_defaults(fn=cmd_train)

    r = sub.add_parser("register", help="copy an artifact into the registry")
    r.add_argument("--artifact", required=True)
    r.add_argument("--registry")
    r.set_defaults(fn=cmd_register)

    pr = sub.add_parser("promote", help="promote a version to champion if it passes the gate")
    pr.add_argument("version")
    pr.add_argument("--registry")
    pr.add_argument("--initial", action="store_true", help="first champion of an empty registry")
    pr.add_argument("--actor", default=os.environ.get("USERNAME") or os.environ.get("USER"))
    pr.set_defaults(fn=cmd_promote)

    s = sub.add_parser("score", help="batch-score every style in a panel at an as-of week")
    s.add_argument("--panel", required=True)
    s.add_argument("--as-of", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--artifact", help="score with this artifact directory, not the champion")
    s.add_argument("--registry")
    s.add_argument("--log-dir", help="append every prediction to this scoring log")
    s.set_defaults(fn=cmd_score)

    g = sub.add_parser("gate-fixture", help="champion/challenger gate on a synthetic registry")
    g.set_defaults(fn=cmd_gate_fixture)

    a = p.parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    raise SystemExit(main())
