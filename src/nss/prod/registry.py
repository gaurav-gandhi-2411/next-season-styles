"""File-based model registry with gated promotion and an append-only audit log.

Layout under the root (a local path or any fsspec URL; C2 points it at `gs://...` unchanged):

    manifest.json               every registered version with its provenance
    champion.json               the champion pointer
    versions/<version>/...      the artifact, copied as-is
    audit/<utc>-<uuid>.json     one object per event (register, promote accepted or refused)

The audit log is append-only by construction: every event is a new object whose name is unique,
and an existing name is never overwritten. That also works on object stores, which cannot append
to an object.

`promote` applies SPEC Section 6 and refuses unless it passes:

1. demand capture@20 (the primary): the challenger-minus-champion paired 95% circular block
   bootstrap interval (L = 13) lies entirely above zero;
2. no guardrail is significantly worse: Hit@3-in-top20, NDCG@10 and Spearman fail if their
   interval lies entirely below zero, WMAPE if entirely above.

Both are paired over the same backtest origins; mismatched origins are refused, not guessed at.
The first champion of an empty registry needs `initial=True`, and its artifact must carry passing
reproduction checks (every recorded check <= 1e-9).
"""

from __future__ import annotations

import io
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fsspec
import polars as pl

from nss.models.circular_bootstrap import circular_ci

PRIMARY = "demand_capture_at_20"
GUARDRAILS: dict[str, int] = {  # +1 higher is better, -1 lower is better
    "hit_at_3_in_top20": 1,
    "ndcg_at_10": 1,
    "spearman_rho": 1,
    "wmape": -1,
}
MIN_ORIGINS = 8
REPRO_ATOL = 1e-9


class PromotionRefused(Exception):
    """The champion/challenger rule (or a precondition) refused the promotion."""


class RegistrationRefused(Exception):
    """The artifact cannot be registered (dirty tree, duplicate version, missing files)."""


REQUIRED_FILES = (
    "metadata.json",
    "calibrator.json",
    "feature_spec.json",
    "reference.json",
    "backtest_per_origin.csv",
)


def gate(champion: pl.DataFrame, challenger: pl.DataFrame) -> dict[str, Any]:
    """Section 6 on two per-origin metric tables. Returns passed, reasons and the statistics."""
    a = challenger.sort("origin_week")
    b = champion.sort("origin_week")
    if a["origin_week"].to_list() != b["origin_week"].to_list():
        return {
            "passed": False,
            "reasons": ["backtest origins differ; not comparable"],
            "stats": {},
        }
    if a.height < MIN_ORIGINS:
        return {"passed": False, "reasons": [f"only {a.height} origins"], "stats": {}}
    stats: dict[str, Any] = {}
    reasons: list[str] = []
    for m in (PRIMARY, *GUARDRAILS):
        d = (a[m] - b[m]).to_numpy()
        c = circular_ci(d)
        stats[m] = {k: c[k] for k in ("n", "mean", "ci_lo", "ci_hi")}
    p = stats[PRIMARY]
    primary_ok = p["ci_lo"] > 0
    reasons.append(
        f"primary {PRIMARY}: {p['mean']:+.4f} [{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}] "
        + ("excludes zero in the challenger's favour" if primary_ok else "does not exclude zero")
    )
    failed = []
    for m, sign in GUARDRAILS.items():
        s = stats[m]
        worse = s["ci_hi"] < 0 if sign > 0 else s["ci_lo"] > 0
        if worse:
            failed.append(m)
            reasons.append(
                f"guardrail {m} significantly worse: {s['mean']:+.4f} [{s['ci_lo']:+.4f}, "
                f"{s['ci_hi']:+.4f}]"
            )
    if not failed:
        reasons.append("no guardrail significantly worse")
    return {
        "passed": primary_ok and not failed,
        "reasons": reasons,
        "stats": stats,
        "guardrail_failures": failed,
    }


class Registry:
    def __init__(self, root: str) -> None:
        self.root = root
        self.fs, self.base = fsspec.core.url_to_fs(root)
        self.base = self.base.rstrip("/")

    # -- paths and small json IO ------------------------------------------------------------
    def _p(self, *parts: str) -> str:
        return "/".join([self.base, *parts])

    def _read_json(self, *parts: str) -> Any:
        with self.fs.open(self._p(*parts), "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_json(self, obj: Any, *parts: str) -> None:
        self.fs.makedirs(self._p(*parts[:-1]) if len(parts) > 1 else self.base, exist_ok=True)
        with self.fs.open(self._p(*parts), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)

    def _audit(self, event: dict[str, Any]) -> str:
        """Write one immutable audit event; never overwrites an existing object."""
        now = datetime.now(UTC)
        name = f"{now:%Y%m%dT%H%M%S%fZ}-{uuid.uuid4().hex[:12]}.json"
        if self.fs.exists(self._p("audit", name)):  # pragma: no cover - uuid collision
            raise RuntimeError(f"audit event {name} already exists")
        self._write_json({"utc": now.isoformat(), **event}, "audit", name)
        return name

    # -- reads ------------------------------------------------------------------------------
    def manifest(self) -> list[dict[str, Any]]:
        if not self.fs.exists(self._p("manifest.json")):
            return []
        return list(self._read_json("manifest.json")["versions"])

    def champion(self) -> str | None:
        if not self.fs.exists(self._p("champion.json")):
            return None
        return str(self._read_json("champion.json")["version"])

    def audit_log(self) -> list[dict[str, Any]]:
        if not self.fs.exists(self._p("audit")):
            return []
        names = sorted(Path(p).name for p in self.fs.ls(self._p("audit"), detail=False))
        return [self._read_json("audit", n) for n in names]

    def metadata(self, version: str) -> dict[str, Any]:
        return dict(self._read_json("versions", version, "metadata.json"))

    def per_origin(self, version: str) -> pl.DataFrame:
        with self.fs.open(self._p("versions", version, "backtest_per_origin.csv"), "rb") as f:
            return pl.read_csv(io.BytesIO(f.read()), try_parse_dates=True)

    def version_path(self, version: str) -> str:
        """The artifact location in fsspec form (a local path for a local registry)."""
        return self._p("versions", version)

    # -- writes -----------------------------------------------------------------------------
    def register(self, artifact_dir: str | Path, allow_dirty: bool = False) -> str:
        art = Path(artifact_dir)
        missing = [f for f in REQUIRED_FILES if not (art / f).exists()]
        if missing:
            raise RegistrationRefused(f"artifact is missing {missing}")
        meta = json.loads((art / "metadata.json").read_text(encoding="utf-8"))
        version = meta["version"]
        if meta.get("git", {}).get("dirty") and not allow_dirty:
            raise RegistrationRefused(
                f"{version} was trained from an uncommitted tree (git dirty); retrain from a commit"
            )
        if any(v["version"] == version for v in self.manifest()):
            raise RegistrationRefused(f"{version} is already registered")
        dest = self._p("versions", version)
        self.fs.makedirs(dest, exist_ok=True)
        for f in sorted(art.iterdir()):
            if f.is_file():
                self.fs.put_file(str(f), f"{dest}/{f.name}")
        entry = {
            "version": version,
            "registered_utc": datetime.now(UTC).isoformat(),
            "config_hash": meta.get("config_hash"),
            "git_sha": meta.get("git", {}).get("sha"),
            "panel_sha256": meta.get("panel_sha256"),
        }
        self._write_json({"versions": [*self.manifest(), entry]}, "manifest.json")
        self._audit({"event": "register", "version": version, "entry": entry})
        return str(version)

    def promote(
        self, version: str, initial: bool = False, actor: str | None = None
    ) -> dict[str, Any]:
        """Promote `version` if the gate passes. Every attempt is audited, accepted or refused."""
        current = self.champion()
        decision: dict[str, Any] = {"version": version, "champion_before": current}
        try:
            if not any(v["version"] == version for v in self.manifest()):
                raise PromotionRefused(f"{version} is not registered")
            if current == version:
                raise PromotionRefused(f"{version} is already the champion")
            if current is None:
                if not initial:
                    raise PromotionRefused("no champion yet; the first promotion needs --initial")
                checks = self.metadata(version).get("checks", {})
                bad = {k: v for k, v in checks.items() if not v <= REPRO_ATOL}
                if not checks or bad:
                    raise PromotionRefused(
                        f"initial champion lacks passing reproduction checks: {bad}"
                    )
                decision |= {
                    "passed": True,
                    "reasons": ["initial champion; reproduction checks pass"],
                }
            else:
                if initial:
                    raise PromotionRefused("--initial is only for an empty registry")
                decision |= gate(self.per_origin(current), self.per_origin(version))
                if not decision["passed"]:
                    raise PromotionRefused("; ".join(decision["reasons"]))
        except PromotionRefused as e:
            decision |= {"passed": False, "refusal": str(e)}
            self._audit({"event": "promote", "accepted": False, "actor": actor, **decision})
            raise
        event = self._audit({"event": "promote", "accepted": True, "actor": actor, **decision})
        self._write_json(
            {
                "version": version,
                "promoted_utc": datetime.now(UTC).isoformat(),
                "audit_event": event,
            },
            "champion.json",
        )
        return decision
