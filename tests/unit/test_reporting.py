"""Unit tests for report generation."""

from __future__ import annotations

from failsafe.agents.sample import get_agent
from failsafe.evaluation import compute_run_metrics, execute_suite, get_builtin_tasks
from failsafe.reporting import build_report_payload, render_html, render_markdown, write_reports


def test_report_payload_structure():
    tasks = [t for t in get_builtin_tasks() if t.id == "task-db-lookup"]
    res = execute_suite(get_agent("buggy"), tasks)[0]
    metrics = compute_run_metrics([res])
    payload = build_report_payload(
        title="Test report", trace=res.trace, findings=res.findings,
        run_result=res, metrics=metrics,
    )
    assert payload["title"] == "Test report"
    assert payload["trace"]["run_id"] == res.run_id
    assert payload["run_result"]["task_success"] is False
    assert payload["findings_by_status"]
    assert payload["known_limitations"]


def test_html_report_contains_evidence():
    tasks = [t for t in get_builtin_tasks() if t.id == "task-update-record"]
    res = execute_suite(get_agent("buggy"), tasks)[0]
    payload = build_report_payload(title="R", trace=res.trace, findings=res.findings, run_result=res)
    html = render_html(payload)
    assert "<html" in html
    assert res.run_id in html
    for f in res.findings:
        for sid in f.step_ids:
            assert sid in html  # exact evidence step IDs present
    assert "Verified" in html or "Rule-triggered" in html
    assert "Known limitations" in html


def test_html_report_clean_control():
    tasks = [t for t in get_builtin_tasks() if t.id == "task-db-lookup"]
    res = execute_suite(get_agent("good"), tasks)[0]
    payload = build_report_payload(title="Clean", trace=res.trace, findings=res.findings, run_result=res)
    html = render_html(payload)
    assert "clean control stays clean" in html
    assert "VERIFIED SUCCESS" in html


def test_markdown_report_no_html_escaping():
    tasks = [t for t in get_builtin_tasks() if t.id == "task-db-lookup"]
    res = execute_suite(get_agent("buggy"), tasks)[0]
    payload = build_report_payload(title="R", trace=res.trace, findings=res.findings, run_result=res)
    md = render_markdown(payload)
    assert "&#34;" not in md
    assert "&quot;" not in md
    assert "# R" in md


def test_markdown_report_sections():
    tasks = get_builtin_tasks()[:2]
    res = execute_suite(get_agent("good"), tasks)
    metrics = compute_run_metrics(res)
    payload = build_report_payload(
        title="Suite", trace=res[0].trace, findings=res[0].findings,
        run_result=res[0], metrics=metrics,
    )
    md = render_markdown(payload)
    for section in ("Run metadata", "Task outcome", "Oracle results", "Detected failures",
                    "Uncertainty & human review", "Known limitations"):
        assert section in md
    assert "task_success_rate" in md


def test_report_gate_and_investigation_sections():
    payload = build_report_payload(
        title="G", gate={"mode": "strict", "passed": False, "reasons": ["bad thing"]},
        investigation={"hypothesis": "H", "alternatives": ["A"], "evidence_step_ids": ["s1"],
                       "proposed_test": "T", "note": "n"},
    )
    html = render_html(payload)
    assert "CI gate" in html
    assert "Investigation report" in html
    assert "bad thing" in html


def test_write_reports_files(tmp_path):
    payload = build_report_payload(title="X")
    paths = write_reports(payload, tmp_path, basename="my-run")
    assert paths["html"].is_file()
    assert paths["markdown"].is_file()
    assert "X" in paths["markdown"].read_text()


def test_report_never_presents_suspicion_as_confirmed():
    from failsafe.schemas import DetectorFinding, Evidence, FindingStatus, Severity

    f = DetectorFinding(
        run_id="r", detector_id="llm_judge", category="semantic", summary="suspicious",
        severity=Severity.LOW, status=FindingStatus.SEMANTIC_SUSPECT,
        evidence=[Evidence(kind="k", step_ids=["s1"])],
    )
    payload = build_report_payload(title="S", findings=[f])
    html = render_html(payload)
    assert "Semantic suspicion (unverified)" in html
    assert "never presented as a confirmed failure" in html
