"""Failsafe command-line interface.

Commands:

- ``failsafe init``              create config, .env.example, and data dirs
- ``failsafe validate-trace``    validate a trace file against the schema
- ``failsafe analyze``           run deterministic detectors on a trace
- ``failsafe replay``            record/replay a sample agent deterministically
- ``failsafe evaluate``          run an agent against an oracle task suite
- ``failsafe report``            generate HTML + Markdown reports
- ``failsafe test``              run regression tests (generated or stored)
- ``failsafe inspect``           inspect recorded runs, findings, and metrics

Exit codes:

- 0  success (gate passed / validation OK)
- 1  failure (gate failed / oracle failed / validation found errors)
- 2  usage or input error (bad path, bad JSON, unknown id)
- 3  internal error
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click

from failsafe import __version__

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_INTERNAL = 3


def _load_config(config_path: str | None, **overrides: Any):
    from failsafe.config import load_config

    clean = {k: v for k, v in overrides.items() if v is not None}
    return load_config(config_path, overrides=clean or None)


def _repo(config: Any):
    from failsafe.storage import Repository

    return Repository(config.db_url)


def _echo_findings(findings: list[Any], verbose: bool = False) -> None:
    if not findings:
        click.echo("  No findings.")
        return
    for f in findings:
        steps = ",".join(f.step_ids) or "n/a"
        click.echo(f"  [{f.severity.value:8s}] {f.detector_id}: {f.summary}")
        click.echo(f"             status={f.status.value} evidence_steps={steps}")
        if verbose:
            for e in f.evidence:
                click.echo(f"             evidence[{e.kind}]: {json.dumps(e.detail, default=str)[:200]}")


@click.group()
@click.version_option(__version__, prog_name="failsafe")
def main() -> None:
    """Failsafe — agentic AI reliability testing framework.

    Detect failures, inject controlled tool faults, verify task outcomes with
    oracles, and keep known reliability failures out of your releases.
    """


@main.command()
@click.option("--config", "config_path", default=None, help="Path to write the config to.")
@click.option("--force", is_flag=True, help="Overwrite existing files.")
def init(config_path: str | None, force: bool) -> None:
    """Create a starter configuration, .env.example, and data directories."""
    cfg_path = Path(config_path or "failsafe.json")
    env_path = Path(".env.example")
    if cfg_path.exists() and not force:
        click.echo(f"Config already exists: {cfg_path} (use --force to overwrite).")
    else:
        default_cfg = {
            "db_url": "sqlite:///failsafe.db",
            "data_dir": ".failsafe",
            "reports_dir": "reports",
            "redact_secrets": True,
            "detectors": {
                "repeat_min_count": 3,
                "retry_budget": 3,
                "require_failure_report": True,
                "ignored_error_allow_actions": ["retry", "fallback", "escalate", "report_failure"],
            },
            "injection": {"seed": 1337, "max_injections": 5, "allow_real_systems": False},
            "budgets": {"max_iterations": 12, "max_tool_calls": 30, "max_llm_calls": 20, "max_seconds": 120.0},
            "ci": {"mode": "strict", "fail_on_min_severity": "high"},
            "llm_enabled": False,
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
        }
        cfg_path.write_text(json.dumps(default_cfg, indent=2), encoding="utf-8")
        click.echo(f"Wrote {cfg_path}")
    if env_path.exists() and not force:
        click.echo(f"{env_path} already exists (use --force to overwrite).")
    else:
        env_path.write_text(
            "# Failsafe environment variables (never commit real secrets)\n"
            "# Copy to .env and fill in values as needed.\n\n"
            "# LLM features are optional. The deterministic core needs none of these.\n"
            "FAILSAFE_LLM_ENABLED=false\n"
            "FAILSAFE_LLM_PROVIDER=openai\n"
            "FAILSAFE_LLM_MODEL=gpt-4o-mini\n"
            "OPENAI_API_KEY=\n\n"
            "# Optional PostgreSQL instead of SQLite\n"
            "# FAILSAFE_DB_URL=postgresql+psycopg2://user:pass@localhost:5432/failsafe\n\n"
            "# Optional overrides\n"
            "# FAILSAFE_DETECTORS__RETRY_BUDGET=3\n"
            "# FAILSAFE_CI__MODE=strict\n",
            encoding="utf-8",
        )
        click.echo(f"Wrote {env_path}")
    Path(".failsafe").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    click.echo("Initialized data dir .failsafe/ and reports/.")
    click.echo("Next: failsafe validate-trace examples/traces/sample_clean.json")


@main.command("validate-trace")
@click.argument("path", type=click.Path(exists=True))
@click.option("--config", "config_path", default=None)
def validate_trace(path: str, config_path: str | None) -> None:
    """Validate a trace file (Failsafe or OpenInference format)."""
    from failsafe.ingestion.loader import TraceValidationError, load_trace

    cfg = _load_config(config_path)
    try:
        trace = load_trace(path, redact=cfg.redact_secrets, max_bytes=cfg.trace_max_bytes, max_steps=cfg.trace_max_steps)
    except TraceValidationError as exc:
        click.echo(f"INVALID: {exc}", err=True)
        for err in exc.errors:
            click.echo(f"  - {err}", err=True)
        sys.exit(EXIT_USAGE)
    click.echo(f"VALID: {path}")
    click.echo(f"  run_id={trace.metadata.run_id} steps={len(trace.steps)} format={trace.source_format}")
    sys.exit(EXIT_OK)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--config", "config_path", default=None)
@click.option("--output", "-o", "output_json", default=None, help="Write findings JSON to this path.")
@click.option("--verbose", "-v", is_flag=True)
def analyze(path: str, config_path: str | None, output_json: str | None, verbose: bool) -> None:
    """Run deterministic failure detectors on a trace and persist the run."""
    from failsafe.ci import evaluate_gate
    from failsafe.detectors import run_detectors
    from failsafe.evaluation.metrics import RunResult
    from failsafe.ingestion.loader import TraceValidationError, load_trace

    cfg = _load_config(config_path)
    try:
        trace = load_trace(path, redact=cfg.redact_secrets, max_bytes=cfg.trace_max_bytes, max_steps=cfg.trace_max_steps)
    except TraceValidationError as exc:
        click.echo(f"INVALID TRACE: {exc}", err=True)
        sys.exit(EXIT_USAGE)

    findings = run_detectors(trace, cfg.detectors)
    click.echo(f"Run {trace.metadata.run_id}: {len(trace.steps)} steps, {len(findings)} finding(s)")
    _echo_findings(findings, verbose)

    repo = _repo(cfg)
    run_id = repo.save_trace(trace)
    repo.save_findings(findings)

    # Derive a trace-scoped oracle from required outcomes (world checks need a
    # live world and are used by `failsafe evaluate`, not here).
    from failsafe.evaluation.oracles import run_checks
    from failsafe.evaluation.world import make_demo_world

    run_results = []
    if trace.task is not None and trace.task.required_outcomes:
        derived_checks = []
        for outcome in trace.task.required_outcomes:
            kind = outcome.get("type")
            if kind == "tool_called":
                derived_checks.append({"check": "trace_tool_called", "tool": outcome.get("tool")})
            elif kind == "tool_success":
                derived_checks.append({"check": "trace_tool_succeeded", "tool": outcome.get("tool")})
            elif kind == "response_contains":
                derived_checks.append({"check": "response_contains", "text": outcome.get("text")})
        if derived_checks:
            oracle = run_checks(make_demo_world(), trace, derived_checks,
                                task_id=trace.task.id, task_name=trace.task.name)
            repo.save_oracle(oracle)
            run_results.append(
                RunResult(run_id=run_id, task_id=trace.task.id or "", task_name=trace.task.name or "",
                          trace=trace, oracle=oracle, findings=findings)
            )

    decision = evaluate_gate(findings, cfg.ci, run_results)
    click.echo(f"Gate ({decision.mode}): {'PASS' if decision.passed else 'FAIL'}")
    for r in decision.reasons:
        click.echo(f"  - {r}")
    click.echo(f"Persisted run {run_id} to {cfg.db_url}")

    if output_json:
        Path(output_json).write_text(
            json.dumps([json.loads(f.model_dump_json()) for f in findings], indent=2),
            encoding="utf-8",
        )
        click.echo(f"Wrote findings to {output_json}")
    sys.exit(EXIT_OK if decision.passed else EXIT_FAIL)


@main.command()
@click.argument("agent_name", type=click.Choice(["good", "buggy", "looping", "flaky"]))
@click.argument("task_name", type=str)
@click.option("--config", "config_path", default=None)
@click.option("--seed", type=int, default=None, help="Seed for fault schedules (default from config).")
@click.option("--replay-count", type=int, default=1, help="Repeat the run N times for replay stability.")
@click.option("--no-faults", is_flag=True, help="Disable fault injection.")
def replay(agent_name: str, task_name: str, config_path: str | None, seed: int | None,
           replay_count: int, no_faults: bool) -> None:
    """Execute a built-in sample agent on a built-in task and record the trace.

    AGENT is one of good|buggy|looping|flaky; TASK_NAME is a built-in task id
    (e.g. task-db-lookup) or its name. Run `failsafe inspect --tasks` to list them.
    """
    from failsafe.agents.sample import get_agent
    from failsafe.evaluation import RunnerConfig, execute_task, get_builtin_tasks
    from failsafe.evaluation.metrics import replay_stability

    cfg = _load_config(config_path)
    tasks = get_builtin_tasks()
    task = next((t for t in tasks if t.spec.id == task_name or t.spec.name == task_name), None)
    if task is None:
        click.echo(f"Unknown task '{task_name}'. Use: failsafe inspect --tasks", err=True)
        sys.exit(EXIT_USAGE)

    agent = get_agent(agent_name)
    run_cfg = RunnerConfig(seed=seed if seed is not None else cfg.injection.seed, inject_faults=not no_faults,
                           detector_config=cfg.detectors)
    results = []
    for i in range(max(1, replay_count)):
        res = execute_task(agent, task, run_cfg, run_id=None if i == 0 else None)
        results.append(res)
        click.echo(f"Run {res.run_id}: oracle={'PASS' if res.oracle.success else 'FAIL'} "
                   f"findings={len(res.findings)} faults_injected={res.faults_injected} "
                   f"recovered={res.recovered} runtime={res.runtime_ms:.1f}ms")
    if len(results) > 1:
        stab = replay_stability(results, results)
        click.echo(f"Replay stability: {stab.value} ({stab.numerator}/{stab.denominator})")

    repo = _repo(cfg)
    for res in results:
        repo.save_trace(res.trace)
        repo.save_findings(res.findings)
        repo.save_oracle(res.oracle)
    click.echo(f"Persisted {len(results)} run(s) to {cfg.db_url}")
    sys.exit(EXIT_OK if all(r.oracle.success for r in results) else EXIT_FAIL)


@main.command()
@click.option("--suite", "suite_path", default=None, help="Directory of JSON task files (default: built-in suite).")
@click.option("--agent", "agent_name", default="good", help="Sample agent: good|buggy|looping|flaky.")
@click.option("--config", "config_path", default=None)
@click.option("--seed", type=int, default=1337)
@click.option("--output", "-o", "output_json", default=None, help="Write metrics JSON to this path.")
@click.option("--save", is_flag=True, help="Persist runs and metrics to the database.")
@click.option("--faults/--no-faults", default=True, help="Enable fault injection schedules.")
def evaluate(suite_path: str | None, agent_name: str, config_path: str | None, seed: int,
             output_json: str | None, save: bool, faults: bool) -> None:
    """Evaluate an agent against oracle tasks and compute reliability metrics."""
    from failsafe.agents.sample import get_agent
    from failsafe.evaluation import (
        RunnerConfig,
        compute_run_metrics,
        execute_suite,
        get_builtin_tasks,
        load_task_suite,
    )

    cfg = _load_config(config_path)
    if suite_path:
        try:
            tasks = load_task_suite(suite_path)
        except Exception as exc:  # noqa: BLE001
            click.echo(f"Failed to load suite: {exc}", err=True)
            sys.exit(EXIT_USAGE)
    else:
        tasks = get_builtin_tasks()

    agent = get_agent(agent_name)
    run_cfg = RunnerConfig(seed=seed, inject_faults=faults, detector_config=cfg.detectors)
    results = execute_suite(agent, tasks, run_cfg)
    metrics = compute_run_metrics(results)

    click.echo(f"Evaluated {len(results)} task(s) with agent '{agent_name}' (seed={seed})")
    for r in results:
        click.echo(f"  {r.task_id:28s} oracle={'PASS' if r.oracle.success else 'FAIL':4s} "
                   f"recovered={str(r.recovered):5s} silent_failure={str(r.silent_failure):5s} "
                   f"findings={len(r.findings)}")
    click.echo("Metrics:")
    for v in metrics.values:
        frac = f" ({v.numerator}/{v.denominator})" if v.numerator is not None else ""
        click.echo(f"  {v.name:32s} = {v.value}{frac}")

    if save:
        repo = _repo(cfg)
        for r in results:
            repo.save_trace(r.trace)
            repo.save_findings(r.findings)
            repo.save_oracle(r.oracle)
        from failsafe.schemas import EvaluationRecord

        repo.save_evaluation(EvaluationRecord(
            name=f"evaluate-{agent_name}", kind="suite",
            data={"metrics": metrics.to_dict(), "task_ids": [r.task_id for r in results], "seed": seed},
        ))
        click.echo(f"Persisted to {cfg.db_url}")

    if output_json:
        Path(output_json).write_text(json.dumps(metrics.to_dict(), indent=2, default=str), encoding="utf-8")
        click.echo(f"Wrote metrics to {output_json}")

    all_ok = all(r.oracle.success for r in results)
    sys.exit(EXIT_OK if all_ok else EXIT_FAIL)


@main.command()
@click.option("--run-id", default=None, help="Generate a report for one stored run.")
@click.option("--all", "all_runs", is_flag=True, help="Generate reports for all stored runs.")
@click.option("--config", "config_path", default=None)
@click.option("--out-dir", default=None, help="Report output directory.")
def report(run_id: str | None, all_runs: bool, config_path: str | None, out_dir: str | None) -> None:
    """Generate HTML and Markdown reports for stored runs."""
    from failsafe.reporting import build_report_payload, write_reports

    cfg = _load_config(config_path)
    repo = _repo(cfg)
    out = Path(out_dir or cfg.reports_dir)

    if run_id:
        run_ids = [run_id]
    elif all_runs:
        run_ids = [r.id for r in repo.list_runs(limit=1000)]
    else:
        click.echo("Provide --run-id or --all.", err=True)
        sys.exit(EXIT_USAGE)

    if not run_ids:
        click.echo("No runs found in the database. Run `failsafe analyze` or `failsafe replay` first.", err=True)
        sys.exit(EXIT_USAGE)

    for rid in run_ids:
        trace = repo.get_trace(rid)
        if trace is None:
            click.echo(f"Run {rid} not found.", err=True)
            sys.exit(EXIT_USAGE)
        from failsafe.schemas import DetectorFinding

        findings = [DetectorFinding.model_validate({
            "id": f.id, "run_id": f.run_id, "detector_id": f.detector_id, "category": f.category,
            "severity": f.severity, "status": f.status, "summary": f.summary,
            "description": f.description, "evidence": f.evidence, "metrics": f.metrics,
            "created_at": f.created_at.isoformat() if f.created_at else None,
        }) for f in repo.list_findings(run_id=rid)]

        payload = build_report_payload(title=f"Failsafe run report — {rid}", trace=trace, findings=findings)
        paths = write_reports(payload, out, basename=rid)
        repo.save_report(rid, "html", str(paths["html"]))
        repo.save_report(rid, "markdown", str(paths["markdown"]))
        click.echo(f"Report for {rid}: {paths['html']} , {paths['markdown']}")
    sys.exit(EXIT_OK)


@main.command()
@click.option("--config", "config_path", default=None)
@click.option("--generate/--no-generate", default=True, help="Generate candidate regression tests from stored findings first.")
@click.option("--agent", "agent_name", default="buggy", help="Agent used to execute regression tests.")
@click.option("--list-only", is_flag=True, help="List regression tests without executing.")
def test(config_path: str | None, generate: bool, agent_name: str, list_only: bool) -> None:
    """Run stored/generated regression tests through the controlled runner."""
    from failsafe.agents.regression import generate_from_findings, task_from_regression_test
    from failsafe.agents.sample import get_agent
    from failsafe.evaluation import RunnerConfig, execute_task
    from failsafe.schemas import DetectorFinding, RegressionTest, RegressionTestState

    cfg = _load_config(config_path)
    repo = _repo(cfg)

    if generate:
        by_run: dict[str, list[DetectorFinding]] = {}
        for f in repo.list_findings():
            by_run.setdefault(f.run_id, []).append(DetectorFinding.model_validate({
                "id": f.id, "run_id": f.run_id, "detector_id": f.detector_id, "category": f.category,
                "severity": f.severity, "status": f.status, "summary": f.summary,
                "description": f.description, "evidence": f.evidence, "metrics": f.metrics,
                "created_at": f.created_at.isoformat() if f.created_at else None,
            }))
        generated = 0
        for rid, findings in by_run.items():
            trace = repo.get_trace(rid)
            if trace is None:
                continue
            existing = {t.scenario for t in repo.list_regression_tests()}
            for t in generate_from_findings(trace, findings):
                if t.scenario in existing:
                    continue
                repo.save_regression_test(t)
                generated += 1
        if generated:
            click.echo(f"Generated {generated} candidate regression test(s).")

    records = repo.list_regression_tests()
    if not records:
        click.echo("No regression tests. Run `failsafe analyze` on a failing trace first.")
        sys.exit(EXIT_OK)

    if list_only:
        for rec in records:
            click.echo(f"  [{rec.state:10s}] {rec.id}  {rec.scenario[:80]}")
        sys.exit(EXIT_OK)

    agent = get_agent(agent_name)
    run_cfg = RunnerConfig(seed=cfg.injection.seed, inject_faults=True, detector_config=cfg.detectors)
    any_failing = False
    for rec in records:
        test_model = RegressionTest.model_validate(rec.payload)
        if test_model.state == RegressionTestState.REJECTED:
            click.echo(f"  [rejected  ] {rec.id}  (validation errors: {rec.last_result.get('validation_errors')})")
            continue
        task = task_from_regression_test(test_model)
        res = execute_task(agent, task, run_cfg)
        # regression test "reproduces the issue" if oracle fails on the buggy agent
        reproduced = not res.oracle.success
        new_state = RegressionTestState.FAILING.value if reproduced else RegressionTestState.PASSING.value
        repo.save_trace(res.trace)
        repo.save_oracle(res.oracle)
        repo.update_regression_test_state(
            rec.id, new_state,
            {"run_id": res.run_id, "oracle_success": res.oracle.success, "reproduced": reproduced},
        )
        click.echo(f"  [{new_state:10s}] {rec.id}  reproduced={reproduced} oracle={'PASS' if res.oracle.success else 'FAIL'}")
        if reproduced:
            any_failing = True

    click.echo("Regression suite: " + ("FAILING tests reproduce known defects (expected for a buggy agent)." if any_failing else "no defects reproduced."))
    sys.exit(EXIT_OK)


@main.command()
@click.option("--config", "config_path", default=None)
@click.option("--run-id", default=None, help="Inspect one run in detail.")
@click.option("--tasks", is_flag=True, help="List built-in oracle tasks.")
@click.option("--findings", is_flag=True, help="List stored findings.")
@click.option("--limits", is_flag=True, help="Show detector known limitations.")
def inspect(config_path: str | None, run_id: str | None, tasks: bool, findings: bool, limits: bool) -> None:
    """Inspect runs, tasks, findings, and detector limitations."""
    cfg = _load_config(config_path)

    if tasks:
        from failsafe.evaluation import get_builtin_tasks

        for t in get_builtin_tasks():
            click.echo(f"  {t.spec.id:30s} {t.spec.name}  [{t.category}]")
            if t.fault_schedule:
                click.echo(f"      faults: {t.fault_schedule}")
        sys.exit(EXIT_OK)

    if limits:
        from failsafe.detectors import registry

        for did in registry.ids():
            det = registry.create(did)
            click.echo(f"  {did}: {det.description[:100]}")
            for lim in det.known_limitations:
                click.echo(f"      limitation: {lim}")
        sys.exit(EXIT_OK)

    repo = _repo(cfg)

    if run_id:
        trace = repo.get_trace(run_id)
        if trace is None:
            click.echo(f"Run {run_id} not found.", err=True)
            sys.exit(EXIT_USAGE)
        meta = trace.metadata
        click.echo(f"Run {meta.run_id}: agent={meta.agent_name} status={meta.status.value} task={meta.task_name}")
        click.echo(f"  steps={len(trace.steps)} faults={len(trace.fault_injections)} final={str(meta.final_response)[:120]!r}")
        click.echo("  Timeline:")
        for s in trace.ordered_steps():
            extra = ""
            if s.tool_call:
                extra = f" {s.tool_call.name}({json.dumps(s.tool_call.arguments, default=str)[:80]})"
            if s.tool_result and not s.tool_result.success:
                extra += f" ERROR={s.tool_result.error_message}"
            click.echo(f"    {s.id:40s} {s.kind.value:12s} {s.status.value:7s}{extra}")
        fnds = repo.list_findings(run_id=run_id)
        click.echo(f"  Findings ({len(fnds)}):")
        for f in fnds:
            click.echo(f"    [{f.severity}] {f.detector_id}: {f.summary[:100]}")
        oracles = repo.list_oracles(run_id=run_id)
        for o in oracles:
            click.echo(f"  Oracle {o.task_name}: {'PASS' if o.success else 'FAIL'} ({o.detail})")
        sys.exit(EXIT_OK)

    if findings:
        for f in repo.list_findings():
            click.echo(f"  {f.run_id}  [{f.severity}] {f.category}: {f.summary[:90]}")
        sys.exit(EXIT_OK)

    runs = repo.list_runs(limit=50)
    if not runs:
        click.echo("No runs recorded. Try: failsafe replay good task-db-lookup")
        sys.exit(EXIT_OK)
    click.echo(f"{'RUN ID':40s} {'STATUS':10s} {'AGENT':16s} {'TASK'}")
    for r in runs:
        click.echo(f"{r.id:40s} {r.status:10s} {r.agent_name:16s} {r.task_name or ''}")
    sys.exit(EXIT_OK)


if __name__ == "__main__":  # pragma: no cover
    main()
