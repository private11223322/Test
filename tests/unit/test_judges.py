"""Unit tests for the optional LLM judge and providers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from failsafe.judges import (
    DisabledProvider,
    EchoProvider,
    JudgeError,
    JudgeResult,
    LLMJudge,
    LLMProviderError,
    human_review,
)
from failsafe.schemas import FindingStatus
from tests.conftest import TraceBuilder


def _trace():
    t = TraceBuilder(run_id="run-j")
    t.call("db_get", {"record_id": "u-1"}, success=False, error_message="boom")
    t.response("All done successfully!")
    return t.finish()


def test_disabled_provider_raises():
    with pytest.raises(LLMProviderError) as exc:
        DisabledProvider().complete("hi")
    assert "deterministic core works" in str(exc.value)


def test_echo_provider_is_deterministic():
    p = EchoProvider()
    a = p.complete("x")
    b = p.complete("x")
    assert a == b
    assert len(p.calls) == 2


def test_judge_result_schema_validation():
    with pytest.raises(ValidationError):
        JudgeResult.model_validate({"verdict": "wrong-value"})
    ok = JudgeResult.model_validate({"verdict": "uncertain", "cited_step_ids": ["s1"]})
    assert ok.verdict == "uncertain"


def test_judge_with_canned_verdict_supported():
    provider = EchoProvider(canned={
        "verdict": "supported",
        "summary": "The agent claimed success without tool evidence.",
        "cited_step_ids": ["step-002"],
        "alternative_explanations": ["The tool may have been mocked."],
        "needs_human_review": True,
        "self_reported_confidence": 0.8,
    })
    judge = LLMJudge(provider)
    result, finding = judge.evaluate_suspicion(_trace(), "claims success without evidence")
    assert result is not None and result.verdict == "supported"
    assert finding is not None
    assert finding.status in (FindingStatus.SEMANTIC_SUSPECT, FindingStatus.UNRESOLVED)
    # never VERIFIED by the judge
    assert finding.status != FindingStatus.VERIFIED
    assert "self_reported_confidence_uncalibrated" in finding.metrics


def test_judge_citation_verification_drops_invalid():
    provider = EchoProvider(canned={
        "verdict": "supported",
        "summary": "x",
        "cited_step_ids": ["step-002", "bogus-step-id", "also-fake"],
        "alternative_explanations": [],
        "needs_human_review": False,
        "self_reported_confidence": 0.9,
    })
    judge = LLMJudge(provider)
    result, finding = judge.evaluate_suspicion(_trace(), "s")
    assert result is not None
    assert result.cited_step_ids == ["step-002"]  # invalid citations dropped
    assert finding.evidence[0].step_ids == ["step-002"]


def test_judge_refuted_is_dismissed():
    provider = EchoProvider(canned={
        "verdict": "refuted",
        "summary": "Evidence contradicts the suspicion.",
        "cited_step_ids": ["step-001"],
        "alternative_explanations": ["retry strategy"],
        "needs_human_review": False,
        "self_reported_confidence": 0.4,
    })
    judge = LLMJudge(provider)
    result, finding = judge.evaluate_suspicion(_trace(), "s")
    assert finding.status == FindingStatus.DISMISSED


def test_judge_malformed_output_raises_judge_error():
    provider = EchoProvider(canned={})  # canned empty -> default valid; use custom
    provider.canned = None

    class BadProvider:
        name = "bad"

        def complete(self, *a, **k):
            return "this is not json {"

    with pytest.raises(JudgeError):
        LLMJudge(BadProvider()).evaluate_suspicion(_trace(), "s")


def test_judge_provider_unavailable_returns_nothing():
    class Missing:
        name = "missing"

        def complete(self, *a, **k):
            raise LLMProviderError("no key")

    result, finding = LLMJudge(Missing()).evaluate_suspicion(_trace(), "s")
    assert result is None and finding is None  # explicit silence, not fabrication


def test_judge_never_controlling_gate():
    from failsafe.ci import evaluate_gate

    provider = EchoProvider(canned={
        "verdict": "supported", "summary": "bad", "cited_step_ids": ["step-002"],
        "alternative_explanations": [], "needs_human_review": True,
        "self_reported_confidence": 0.99,
    })
    _, finding = LLMJudge(provider).evaluate_suspicion(_trace(), "s")
    decision = evaluate_gate([finding])
    assert decision.passed  # semantic suspicion alone never blocks the build


def test_human_review_workflow():
    provider = EchoProvider(canned={
        "verdict": "supported", "summary": "bad", "cited_step_ids": ["step-002"],
        "alternative_explanations": [], "needs_human_review": True,
        "self_reported_confidence": 0.5,
    })
    _, finding = LLMJudge(provider).evaluate_suspicion(_trace(), "s")
    confirmed = human_review(finding, accept=True, note="checked manually")
    assert confirmed.status == FindingStatus.CONFIRMED_BY_HUMAN
    from failsafe.ci import evaluate_gate

    assert not evaluate_gate([confirmed]).passed  # human confirmation IS authoritative
    rejected = human_review(finding, accept=False)
    assert rejected.status == FindingStatus.REJECTED_BY_HUMAN
    assert evaluate_gate([rejected]).passed


def test_uncalibrated_confidence_never_used_as_probability():
    provider = EchoProvider(canned={
        "verdict": "uncertain", "summary": "unsure", "cited_step_ids": [],
        "alternative_explanations": ["a"], "needs_human_review": True,
        "self_reported_confidence": 0.42,
    })
    _, finding = LLMJudge(provider).evaluate_suspicion(_trace(), "s")
    assert "uncalibrated" in json_keys(finding.metrics)


def json_keys(d):
    return " ".join(d.keys())
