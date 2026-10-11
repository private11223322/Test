"""Repository layer: persistence operations, kept separate from detector logic."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from failsafe.schemas import (
    DetectorFinding,
    EvaluationRecord,
    OracleOutcome,
    RegressionTest,
    Trace,
)
from failsafe.storage.models import (
    EvaluationRecord as EvaluationRecordORM,
)
from failsafe.storage.models import (
    FaultRecord,
    FindingRecord,
    OracleRecord,
    RegressionTestRecord,
    ReportRecord,
    RunRecord,
    StepRecord,
    ToolCallRecord,
    make_session_factory,
)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class Repository:
    """Persistence facade over SQLAlchemy.  Detectors never import this."""

    def __init__(self, db_url: str = "sqlite:///failsafe.db", session_factory: sessionmaker | None = None):
        self.db_url = db_url
        self._session_factory = session_factory or make_session_factory(db_url)

    def _session(self) -> Session:
        return self._session_factory()

    # ---- runs / traces ----------------------------------------------
    def save_trace(self, trace: Trace) -> str:
        with self._session() as s:
            meta = trace.metadata
            run = RunRecord(
                id=meta.run_id,
                agent_name=meta.agent_name,
                agent_version=meta.agent_version,
                task_id=meta.task_id,
                task_name=meta.task_name,
                status=meta.status.value,
                started_at=meta.started_at,
                ended_at=meta.ended_at,
                seed=meta.seed,
                final_response=meta.final_response,
                error=meta.error,
                labels=dict(meta.labels),
                llm_usage=dict(meta.llm_usage),
                source_format=trace.source_format,
                raw_trace=trace.model_dump(mode="json"),
            )
            s.merge(run)
            for step in trace.steps:
                s.merge(StepRecord(
                    id=step.id,
                    run_id=meta.run_id,
                    kind=step.kind.value,
                    name=step.name,
                    status=step.status.value,
                    parent_id=step.parent_id,
                    timestamp=step.timestamp,
                    end_timestamp=step.end_timestamp,
                    input=step.input,
                    output=step.output,
                    error_type=step.error_type,
                    error_message=step.error_message,
                    metadata_json=step.metadata,
                ))
                if step.tool_call is not None:
                    result = step.tool_result
                    s.merge(ToolCallRecord(
                        id=step.tool_call.id,
                        run_id=meta.run_id,
                        step_id=step.id,
                        name=step.tool_call.name,
                        arguments=step.tool_call.arguments,
                        success=None if result is None else result.success,
                        output=None if result is None else result.output,
                        error_type=None if result is None else result.error_type,
                        error_message=None if result is None else result.error_message,
                    ))
            for fi in trace.fault_injections:
                s.merge(FaultRecord(
                    id=fi.id,
                    run_id=meta.run_id,
                    step_id=fi.step_id,
                    tool_call_id=fi.tool_call_id,
                    tool_name=fi.tool_name,
                    kind=fi.kind.value,
                    scheduled_index=fi.scheduled_index,
                    seed=fi.seed,
                    triggered=fi.triggered,
                    injected_at=fi.injected_at,
                    details=fi.details,
                ))
            s.commit()
            return meta.run_id

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._session() as s:
            return s.get(RunRecord, run_id)

    def get_trace(self, run_id: str) -> Trace | None:
        with self._session() as s:
            run = s.get(RunRecord, run_id)
            if run is None or run.raw_trace is None:
                return None
            return Trace.model_validate(run.raw_trace)

    def list_runs(
        self,
        status: str | None = None,
        agent_name: str | None = None,
        category: str | None = None,
        limit: int = 100,
    ) -> list[RunRecord]:
        with self._session() as s:
            stmt = select(RunRecord).order_by(RunRecord.started_at.desc()).limit(limit)
            if status:
                stmt = stmt.where(RunRecord.status == status)
            if agent_name:
                stmt = stmt.where(RunRecord.agent_name == agent_name)
            runs = list(s.scalars(stmt))
            if category:
                filtered = []
                for r in runs:
                    fstmt = select(FindingRecord).where(FindingRecord.run_id == r.id, FindingRecord.category == category)
                    if s.scalars(fstmt).first():
                        filtered.append(r)
                runs = filtered
            return runs

    def delete_run(self, run_id: str) -> None:
        with self._session() as s:
            for model in (StepRecord, ToolCallRecord, FaultRecord, FindingRecord, OracleRecord):
                s.execute(delete(model).where(model.run_id == run_id))
            s.execute(delete(RunRecord).where(RunRecord.id == run_id))
            s.commit()

    # ---- findings ----------------------------------------------------
    def save_findings(self, findings: Sequence[DetectorFinding]) -> None:
        with self._session() as s:
            for f in findings:
                s.merge(FindingRecord(
                    id=f.id,
                    run_id=f.run_id,
                    detector_id=f.detector_id,
                    category=f.category,
                    severity=f.severity.value,
                    status=f.status.value,
                    summary=f.summary,
                    description=f.description,
                    evidence=[e.model_dump(mode="json") for e in f.evidence],
                    metrics=f.metrics,
                    created_at=f.created_at,
                ))
            s.commit()

    def list_findings(self, run_id: str | None = None, category: str | None = None) -> list[FindingRecord]:
        with self._session() as s:
            stmt = select(FindingRecord)
            if run_id:
                stmt = stmt.where(FindingRecord.run_id == run_id)
            if category:
                stmt = stmt.where(FindingRecord.category == category)
            return list(s.scalars(stmt.order_by(FindingRecord.created_at)))

    def update_finding_status(self, finding_id: str, status: str) -> None:
        with self._session() as s:
            rec = s.get(FindingRecord, finding_id)
            if rec is not None:
                rec.status = status
                s.commit()

    # ---- oracle results ----------------------------------------------
    def save_oracle(self, outcome: OracleOutcome) -> None:
        with self._session() as s:
            s.merge(OracleRecord(
                id=outcome.id,
                run_id=outcome.run_id,
                task_id=outcome.task_id,
                task_name=outcome.task_name,
                success=outcome.success,
                checks=[c.model_dump(mode="json") for c in outcome.checks],
                detail=outcome.detail,
                created_at=outcome.created_at,
            ))
            s.commit()

    def list_oracles(self, run_id: str | None = None) -> list[OracleRecord]:
        with self._session() as s:
            stmt = select(OracleRecord)
            if run_id:
                stmt = stmt.where(OracleRecord.run_id == run_id)
            return list(s.scalars(stmt))

    # ---- regression tests --------------------------------------------
    def save_regression_test(self, test: RegressionTest) -> None:
        with self._session() as s:
            s.merge(RegressionTestRecord(
                id=test.id,
                state=test.state.value,
                scenario=test.scenario,
                payload=test.model_dump(mode="json"),
                source=test.source,
                original_run_id=test.original_run_id,
                created_at=test.created_at,
                last_result=test.last_result,
            ))
            s.commit()

    def list_regression_tests(self, state: str | None = None) -> list[RegressionTestRecord]:
        with self._session() as s:
            stmt = select(RegressionTestRecord)
            if state:
                stmt = stmt.where(RegressionTestRecord.state == state)
            return list(s.scalars(stmt))

    def get_regression_test(self, test_id: str) -> RegressionTest | None:
        with self._session() as s:
            rec = s.get(RegressionTestRecord, test_id)
            return None if rec is None else RegressionTest.model_validate(rec.payload)

    def update_regression_test_state(self, test_id: str, state: str, last_result: dict[str, Any] | None = None) -> None:
        with self._session() as s:
            rec = s.get(RegressionTestRecord, test_id)
            if rec is not None:
                rec.state = state
                if last_result is not None:
                    rec.last_result = last_result
                    payload = dict(rec.payload)
                    payload["state"] = state
                    payload["last_result"] = last_result
                    rec.payload = payload
                s.commit()

    # ---- evaluations / reports ---------------------------------------
    def save_evaluation(self, record: EvaluationRecord) -> None:
        with self._session() as s:
            s.merge(EvaluationRecordORM(
                id=record.id,
                name=record.name,
                kind=record.kind,
                run_id=record.run_id,
                data=record.data,
                created_at=record.created_at,
            ))
            s.commit()

    def list_evaluations(self, kind: str | None = None) -> list[EvaluationRecordORM]:
        with self._session() as s:
            stmt = select(EvaluationRecordORM)
            if kind:
                stmt = stmt.where(EvaluationRecordORM.kind == kind)
            return list(s.scalars(stmt))

    def save_report(self, run_id: str | None, fmt: str, path: str) -> str:
        rid = _new_id("report")
        with self._session() as s:
            s.add(ReportRecord(id=rid, run_id=run_id, format=fmt, path=path))
            s.commit()
        return rid

    def list_reports(self, run_id: str | None = None) -> list[ReportRecord]:
        with self._session() as s:
            stmt = select(ReportRecord)
            if run_id:
                stmt = stmt.where(ReportRecord.run_id == run_id)
            return list(s.scalars(stmt))
