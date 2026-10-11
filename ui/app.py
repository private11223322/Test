"""Failsafe Streamlit trace viewer.

A lightweight UI over the existing backend services (repository, detectors,
metrics, reporting) — no business logic is duplicated here.

Run with:

    FAILSAFE_UI_DB_URL=sqlite:///failsafe.db streamlit run ui/app.py

Features: run listing & filtering, timeline, tool I/O inspection, injected
faults, oracle outcomes, run comparison, investigation reports, and recovery
metrics.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# Make the src layout importable when running from a checkout.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from failsafe.config import load_config  # noqa: E402
from failsafe.reporting import build_report_payload  # noqa: E402
from failsafe.schemas import Trace  # noqa: E402
from failsafe.storage import Repository  # noqa: E402

st.set_page_config(page_title="Failsafe Trace Viewer", page_icon="🛟", layout="wide")


@st.cache_resource
def get_repo() -> Repository:
    cfg = load_config()
    db_url = os.environ.get("FAILSAFE_UI_DB_URL") or cfg.db_url
    return Repository(db_url)


def _load_trace(run_id: str) -> Trace | None:
    return get_repo().get_trace(run_id)


def _timeline(trace: Trace) -> None:
    rows = []
    for s in trace.ordered_steps():
        tool = ""
        if s.tool_call is not None:
            tool = f"{s.tool_call.name}({_short(s.tool_call.arguments)})"
        err = s.error_message or ""
        rows.append({
            "step_id": s.id,
            "kind": s.kind.value,
            "status": s.status.value,
            "name": s.name or "",
            "tool": tool,
            "error": _short(err, 60),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _short(v, n: int = 100) -> str:
    text = str(v)
    return text if len(text) <= n else text[: n - 1] + "…"


def _step_detail(trace: Trace) -> None:
    ids = [s.id for s in trace.ordered_steps()]
    sid = st.selectbox("Select a step to inspect", ids)
    step = trace.step_by_id(sid)
    if step is None:
        return
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Kind:** `{step.kind.value}` · **Status:** `{step.status.value}`")
        if step.tool_call is not None:
            st.markdown("**Tool call**")
            st.json(step.tool_call.model_dump(mode="json"))
        if step.input is not None:
            st.markdown("**Input**")
            st.json(step.input)
    with c2:
        if step.tool_result is not None:
            st.markdown("**Tool result**")
            st.json(step.tool_result.model_dump(mode="json"))
        if step.output is not None:
            st.markdown("**Output**")
            st.json(step.output if not isinstance(step.output, str) else {"text": step.output})
        if step.error_message:
            st.error(step.error_message)


def _findings(run_id: str) -> None:
    repo = get_repo()
    findings = repo.list_findings(run_id=run_id)
    if not findings:
        st.success("No findings — clean control stays clean.")
        return
    for f in findings:
        color = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵", "info": "⚪"}.get(f.severity, "⚪")
        label = f.status.replace("_", " ")
        with st.expander(f"{color} [{label}] {f.detector_id}: {_short(f.summary, 90)}"):
            st.markdown(f"**Category:** `{f.category}` · **Severity:** `{f.severity}`")
            st.write(f.description)
            st.markdown("**Evidence step IDs**")
            steps = [sid for e in f.evidence for sid in e.step_ids]
            st.code(", ".join(steps) or "(none)")
            st.json([dict(e) for e in f.evidence])
            st.caption(
                "Verified = confirmed against evidence/oracle · Rule-triggered = read the evidence · "
                "Semantic suspicion = unverified LLM output · Unresolved = needs human review"
            )


def _faults(oracle_rows, trace: Trace) -> None:
    if trace.fault_injections:
        st.markdown("**Injected faults**")
        st.dataframe(
            [
                {
                    "tool": fi.tool_name,
                    "kind": fi.kind.value,
                    "triggered": fi.triggered,
                    "seed": fi.seed,
                    "details": _short(fi.details, 60),
                }
                for fi in trace.fault_injections
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No faults were injected in this run.")
    st.markdown("**Oracle outcomes**")
    if not oracle_rows:
        st.info("No oracle results recorded for this run.")
        return
    for o in oracle_rows:
        icon = "✅" if o.success else "❌"
        st.markdown(f"{icon} **{o.task_name or o.task_id}** — {o.detail}")
        checks = []
        for c in o.checks:
            checks.append({
                "check": c.get("name"),
                "passed": c.get("passed"),
                "detail": _short(c.get("detail"), 80),
                "expected": _short(c.get("expected"), 40),
                "actual": _short(c.get("actual"), 40),
            })
        st.dataframe(checks, use_container_width=True, hide_index=True)


def _metrics(run_id: str) -> None:
    from failsafe.evaluation.metrics import MetricsReport, RunResult
    from failsafe.evaluation.oracles import run_checks
    from failsafe.evaluation.world import make_demo_world

    repo = get_repo()
    trace = repo.get_trace(run_id)
    if trace is None:
        return
    oracle_rows = repo.list_oracles(run_id=run_id)
    findings = repo.list_findings(run_id=run_id)
    oracle_success = bool(oracle_rows) and all(o.success for o in oracle_rows)
    from failsafe.schemas import DetectorFinding, Evidence, OracleCheck, OracleOutcome

    outcome = OracleOutcome(
        run_id=run_id, task_id=trace.metadata.task_id or "-",
        task_name=trace.metadata.task_name, success=oracle_success,
        checks=[
            OracleCheck(name=c.get("name", "?"), passed=bool(c.get("passed")), detail=str(c.get("detail", "")))
            for o in oracle_rows for c in o.checks
        ],
    )
    result = RunResult(
        run_id=run_id, task_id=trace.metadata.task_id or "-",
        task_name=trace.metadata.task_name or "-", trace=trace, oracle=outcome,
        faults_injected=len(trace.fault_injections),
    )
    report = MetricsReport()
    from failsafe.evaluation.metrics import compute_run_metrics

    report = compute_run_metrics([result])
    st.markdown("**Recovery & reliability metrics**")
    st.dataframe(
        [
            {
                "metric": v.name,
                "value": v.value,
                "fraction": f"{v.numerator}/{v.denominator}" if v.numerator is not None else "—",
                "denominator": v.denominator_definition,
            }
            for v in report.values
        ],
        use_container_width=True,
        hide_index=True,
    )


def _compare(run_ids: list[str]) -> None:
    if len(run_ids) < 2:
        st.info("Select at least two runs to compare.")
        return
    traces = [(rid, _load_trace(rid)) for rid in run_ids]
    rows = []
    for rid, t in traces:
        if t is None:
            continue
        findings = get_repo().list_findings(run_id=rid)
        oracles = get_repo().list_oracles(run_id=rid)
        rows.append({
            "run_id": rid,
            "agent": t.metadata.agent_name,
            "version": t.metadata.agent_version or "—",
            "status": t.metadata.status.value,
            "steps": len(t.steps),
            "findings": len(findings),
            "oracle": "PASS" if oracles and all(o.success for o in oracles) else ("FAIL" if oracles else "—"),
            "faults": len(t.fault_injections),
            "final_response": _short(t.metadata.final_response or "", 60),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.markdown("**Tool sequences**")
    for rid, t in traces:
        if t is None:
            continue
        seq = " → ".join(tc.name for tc in t.tool_calls()) or "(no tool calls)"
        st.markdown(f"- `{rid}`: {seq}")


def _investigation(run_id: str) -> None:
    """Render a stored investigation report when one exists."""
    repo = get_repo()
    evals = repo.list_evaluations(kind="investigation")
    matched = [e for e in evals if e.data.get("run_id") == run_id]
    if not matched:
        st.info(
            "No stored investigation report for this run. Generate one with: "
            "`failsafe evaluate` (advisory) or the InvestigationAgent API."
        )
        return
    for e in matched:
        st.markdown(f"### Investigation — {e.name}")
        st.json(e.data)


def main() -> None:
    st.title("🛟 Failsafe Trace Viewer")
    st.caption("Evidence before conclusions — verified findings, rule triggers, and suspicions are labelled distinctly.")
    repo = get_repo()
    runs = repo.list_runs(limit=200)
    if not runs:
        st.warning("No runs recorded in the database yet.")
        st.markdown("Try: `failsafe replay good task-db-lookup` or `failsafe evaluate --agent good --save`")
        return

    with st.sidebar:
        st.header("Filters")
        statuses = sorted({r.status for r in runs})
        status = st.multiselect("Run status", statuses, default=statuses)
        categories = sorted({f.category for f in repo.list_findings()})
        category = st.selectbox("Failure category", ["(all)"] + categories)
        agents = sorted({r.agent_name for r in runs})
        agent = st.selectbox("Agent", ["(all)"] + agents)
        run_ids = [r.id for r in runs if r.status in status and (agent == "(all)" or r.agent_name == agent)]
        if category != "(all)":
            filtered = set(f.run_id for f in repo.list_findings(category=category))
            run_ids = [r for r in run_ids if r in filtered]

    st.markdown(f"**{len(run_ids)} run(s)**")
    selected = st.selectbox("Run", run_ids)
    compare = st.multiselect("Compare runs", [r for r in run_ids if r != selected])

    if selected:
        tab_list, tab_timeline, tab_detail, tab_findings, tab_faults, tab_metrics, tab_invest, tab_compare = st.tabs([
            "Runs", "Timeline", "Step detail", "Findings", "Faults & oracles", "Metrics", "Investigation", "Compare",
        ])
        trace = _load_trace(selected)
        with tab_list:
            st.dataframe(
                [
                    {
                        "run_id": r.id,
                        "agent": r.agent_name,
                        "status": r.status,
                        "task": r.task_name or "",
                        "started": str(r.started_at),
                        "findings": len(repo.list_findings(run_id=r.id)),
                    }
                    for r in repo.list_runs(limit=200) if r.id in run_ids
                ],
                use_container_width=True,
                hide_index=True,
            )
        if trace is not None:
            with tab_timeline:
                st.markdown(f"**Final response** — {_short(trace.metadata.final_response or '(none)', 200)}")
                _timeline(trace)
            with tab_detail:
                _step_detail(trace)
            with tab_findings:
                _findings(selected)
            with tab_faults:
                _faults(repo.list_oracles(run_id=selected), trace)
            with tab_metrics:
                _metrics(selected)
            with tab_invest:
                _investigation(selected)
            with tab_compare:
                _compare([selected] + compare)

            st.download_button(
                "Download run JSON",
                data=trace.model_dump_json(indent=2),
                file_name=f"{selected}.json",
                mime="application/json",
            )


if __name__ == "__main__":
    main()
