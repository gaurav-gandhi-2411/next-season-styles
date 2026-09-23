from __future__ import annotations

from pathlib import Path

import pytest

from nss.prod.fixture import build_fixture_registry, challenger, make_artifact, synthetic_per_origin
from nss.prod.registry import PromotionRefused, RegistrationRefused, Registry, gate


def test_guardrail_regression_is_refused_and_audited(tmp_path: Path) -> None:
    """Better on the primary, significantly worse on Spearman: refused, pointer unchanged."""
    reg = build_fixture_registry(tmp_path)
    before = len(reg.audit_log())
    with pytest.raises(PromotionRefused, match="guardrail spearman_rho significantly worse"):
        reg.promote("breaks_guardrail", actor="test")
    assert reg.champion() == "champion"
    log = reg.audit_log()
    assert len(log) == before + 1
    assert log[-1]["event"] == "promote" and log[-1]["accepted"] is False
    assert log[-1]["version"] == "breaks_guardrail"
    assert "spearman_rho" in log[-1]["guardrail_failures"]


def test_no_effect_challenger_is_refused(tmp_path: Path) -> None:
    reg = build_fixture_registry(tmp_path)
    with pytest.raises(PromotionRefused, match="does not exclude zero"):
        reg.promote("no_effect")
    assert reg.champion() == "champion"


def test_improving_challenger_is_promoted(tmp_path: Path) -> None:
    reg = build_fixture_registry(tmp_path)
    decision = reg.promote("improves", actor="test")
    assert decision["passed"] and reg.champion() == "improves"
    assert reg.audit_log()[-1]["accepted"] is True


def test_audit_log_is_append_only(tmp_path: Path) -> None:
    reg = build_fixture_registry(tmp_path)
    first = reg.audit_log()
    for v in ("breaks_guardrail", "no_effect"):
        with pytest.raises(PromotionRefused):
            reg.promote(v)
    after = reg.audit_log()
    assert after[: len(first)] == first  # earlier events untouched
    assert len(after) == len(first) + 2


def test_first_promotion_needs_initial_flag(tmp_path: Path) -> None:
    reg = Registry(str(tmp_path / "reg"))
    reg.register(make_artifact(tmp_path / "a", "v1", synthetic_per_origin()))
    with pytest.raises(PromotionRefused, match="needs --initial"):
        reg.promote("v1")
    reg.promote("v1", initial=True)
    assert reg.champion() == "v1"
    reg.register(make_artifact(tmp_path / "a", "v2", synthetic_per_origin()))
    with pytest.raises(PromotionRefused, match="only for an empty registry"):
        reg.promote("v2", initial=True)


def test_unregistered_and_self_promotion_refused(tmp_path: Path) -> None:
    reg = build_fixture_registry(tmp_path)
    with pytest.raises(PromotionRefused, match="not registered"):
        reg.promote("ghost")
    with pytest.raises(PromotionRefused, match="already the champion"):
        reg.promote("champion")


def test_dirty_or_duplicate_artifact_refused(tmp_path: Path) -> None:
    reg = Registry(str(tmp_path / "reg"))
    with pytest.raises(RegistrationRefused, match="uncommitted"):
        reg.register(make_artifact(tmp_path / "a", "dirty", synthetic_per_origin(), dirty=True))
    reg.register(make_artifact(tmp_path / "b", "v1", synthetic_per_origin()))
    with pytest.raises(RegistrationRefused, match="already registered"):
        reg.register(tmp_path / "b" / "v1")


def test_gate_refuses_mismatched_origins() -> None:
    champ = synthetic_per_origin()
    chal = challenger(champ, "improves").head(40)
    assert gate(champ, chal)["passed"] is False


def test_registry_root_is_a_url(tmp_path: Path) -> None:
    """The root goes through fsspec, so a URL form works the same (C2 uses gs://)."""
    reg = Registry((tmp_path / "reg").as_uri())
    reg.register(make_artifact(tmp_path / "a", "v1", synthetic_per_origin()))
    assert [v["version"] for v in reg.manifest()] == ["v1"]
