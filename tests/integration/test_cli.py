"""Integration tests for the CLI and end-to-end workflows."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from failsafe.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def workdir(tmp_path):
    return tmp_path


def test_cli_help(runner):
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for cmd in ("init", "validate-trace", "analyze", "replay", "evaluate", "report", "test", "inspect"):
        assert cmd in result.output


def test_cli_init(runner, workdir):
    with runner.isolated_filesystem(temp_dir=workdir):
        result = runner.invoke(main, ["init"])
        assert result.exit_code == 0
        assert (workdir / ".." or True)
        import os

        assert os.path.exists("failsafe.json")
        assert os.path.exists(".env.example")
        cfg = json.load(open("failsafe.json"))
        assert cfg["llm_enabled"] is False
        # idempotent
        result2 = runner.invoke(main, ["init"])
        assert result2.exit_code == 0


def test_cli_validate_trace_good(runner, workdir, tmp_path):
    trace = {
        "metadata": {"run_id": "r-1", "agent_name": "a", "status": "success"},
        "steps": [{"id": "s1", "kind": "response", "output": "hi"}],
    }
    p = tmp_path / "t.json"
    p.write_text(json.dumps(trace), encoding="utf-8")
    result = runner.invoke(main, ["validate-trace", str(p)])
    assert result.exit_code == 0
    assert "VALID" in result.output


def test_cli_validate_trace_bad(runner, tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"metadata": {"run_id": 1}, "steps": [{"kind": "bogus"}]}), encoding="utf-8")
    result = runner.invoke(main, ["validate-trace", str(p)])
    assert result.exit_code == 2
    assert "INVALID" in result.output


def test_cli_analyze_and_report(runner, workdir):
    with runner.isolated_filesystem(temp_dir=workdir):
        runner.invoke(main, ["init"])
        trace = {
            "metadata": {"run_id": "r-loop", "agent_name": "a", "status": "success"},
            "task": {"id": "t1", "name": "t", "required_outcomes": [{"type": "tool_called", "tool": "db_get"}]},
            "steps": [
                {"id": f"s{i}", "kind": "tool_call", "name": "db_get",
                 "tool_call": {"id": f"tc{i}", "name": "db_get", "arguments": {"record_id": "u-1"}}}
                for i in range(5)
            ],
        }
        p = "t.json"
        open(p, "w").write(json.dumps(trace))
        result = runner.invoke(main, ["analyze", p, "--output", "findings.json"])
        assert result.exit_code == 1  # gate fails: repeated actions + unmet outcome
        assert "repeated_actions" in result.output or "premature" in result.output
        assert "findings.json" in __import__("os").listdir(".")
        rep = runner.invoke(main, ["report", "--run-id", "r-loop"])
        assert rep.exit_code == 0
        assert __import__("os").path.exists("reports/r-loop.html")
        assert __import__("os").path.exists("reports/r-loop.md")


def test_cli_analyze_clean_control(runner, workdir):
    with runner.isolated_filesystem(temp_dir=workdir):
        runner.invoke(main, ["init"])
        trace = {
            "metadata": {"run_id": "r-clean", "agent_name": "a", "status": "success"},
            "steps": [
                {"id": "s1", "kind": "tool_call", "name": "db_get",
                 "tool_call": {"id": "tc1", "name": "db_get", "arguments": {"record_id": "u-1"}}},
                {"id": "s2", "kind": "tool_result", "name": "db_get",
                 "tool_result": {"tool_call_id": "tc1", "name": "db_get", "success": True, "output": {}}},
                {"id": "s3", "kind": "response", "output": "The plan is free."},
            ],
        }
        open("t.json", "w").write(json.dumps(trace))
        result = runner.invoke(main, ["analyze", "t.json"])
        assert result.exit_code == 0
        assert "PASS" in result.output


def test_cli_replay_good_and_buggy(runner, workdir):
    with runner.isolated_filesystem(temp_dir=workdir):
        runner.invoke(main, ["init"])
        good = runner.invoke(main, ["replay", "good", "task-db-lookup"])
        assert good.exit_code == 0
        buggy = runner.invoke(main, ["replay", "buggy", "task-db-lookup"])
        assert buggy.exit_code == 1  # oracle fails
        insp = runner.invoke(main, ["inspect"])
        assert insp.exit_code == 0
        assert "good-agent" in insp.output and "buggy-agent" in insp.output


def test_cli_replay_unknown_task(runner, workdir):
    with runner.isolated_filesystem(temp_dir=workdir):
        runner.invoke(main, ["init"])
        result = runner.invoke(main, ["replay", "good", "no-such-task"])
        assert result.exit_code == 2


def test_cli_replay_stability(runner, workdir):
    with runner.isolated_filesystem(temp_dir=workdir):
        runner.invoke(main, ["init"])
        result = runner.invoke(main, ["replay", "good", "task-db-lookup", "--replay-count", "2"])
        assert result.exit_code == 0
        assert "Replay stability" in result.output


def test_cli_evaluate_suite(runner, workdir):
    with runner.isolated_filesystem(temp_dir=workdir):
        runner.invoke(main, ["init"])
        result = runner.invoke(main, ["evaluate", "--agent", "good", "--output", "metrics.json", "--save"])
        assert result.exit_code == 0
        metrics = json.load(open("metrics.json"))
        names = {v["name"] for v in metrics["values"]}
        assert "task_success_rate" in names
        assert "fault_recovery_rate" in names
        assert "detector_f1" not in names  # detector metrics need expected labels


def test_cli_test_generates_and_runs(runner, workdir):
    with runner.isolated_filesystem(temp_dir=workdir):
        runner.invoke(main, ["init"])
        runner.invoke(main, ["replay", "looping", "task-db-lookup"])
        result = runner.invoke(main, ["test"])
        assert result.exit_code == 0
        assert "regression" in result.output.lower() or "reproduced" in result.output
        listed = runner.invoke(main, ["test", "--list-only"])
        assert listed.exit_code == 0


def test_cli_inspect_tasks_and_limits(runner):
    result = runner.invoke(main, ["inspect", "--tasks"])
    assert result.exit_code == 0
    assert "task-db-lookup" in result.output
    limits = runner.invoke(main, ["inspect", "--limits"])
    assert limits.exit_code == 0
    assert "limitation" in limits.output


def test_cli_inspect_run_detail(runner, workdir):
    with runner.isolated_filesystem(temp_dir=workdir):
        runner.invoke(main, ["init"])
        runner.invoke(main, ["replay", "buggy", "task-db-lookup"])
        listing = runner.invoke(main, ["inspect"])
        run_id = listing.output.splitlines()[1].split()[0]
        detail = runner.invoke(main, ["inspect", "--run-id", run_id])
        assert detail.exit_code == 0
        assert "Timeline" in detail.output
        assert "Oracle" in detail.output


def test_end_to_end_demo_flow(runner, workdir):
    """The full documented demo: init -> analyze -> evaluate -> report -> test."""
    with runner.isolated_filesystem(temp_dir=workdir):
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["replay", "good", "task-db-lookup"]).exit_code == 0
        assert runner.invoke(main, ["replay", "buggy", "task-db-lookup"]).exit_code == 1
        ev = runner.invoke(main, ["evaluate", "--agent", "good"])
        assert ev.exit_code == 0
        rep = runner.invoke(main, ["report", "--all"])
        assert rep.exit_code == 0
        import os

        assert any(f.endswith(".html") for f in os.listdir("reports"))
        t = runner.invoke(main, ["test"])
        assert t.exit_code == 0
