"""Oracle-based task verification, reliability metrics, and evaluation."""

from __future__ import annotations

from failsafe.evaluation.campaigns import (
    AdaptiveCampaign,
    CampaignBudget,
    CampaignPhase,
    CampaignReport,
    FixedCampaign,
    Scenario,
)
from failsafe.evaluation.metrics import (
    MetricsReport,
    MetricValue,
    RunResult,
    compute_run_metrics,
    confusion_matrix,
    detector_metrics,
    live_run_variability,
    precision_recall_f1,
    replay_stability,
)
from failsafe.evaluation.oracles import checker_names, register_checker, run_checks
from failsafe.evaluation.runner import Agent, RunnerConfig, execute_suite, execute_task
from failsafe.evaluation.tasks import OracleTask, get_builtin_tasks, load_task_suite
from failsafe.evaluation.world import SyntheticWorld, make_demo_world

__all__ = [
    "SyntheticWorld",
    "make_demo_world",
    "OracleTask",
    "get_builtin_tasks",
    "load_task_suite",
    "run_checks",
    "register_checker",
    "checker_names",
    "RunResult",
    "MetricsReport",
    "MetricValue",
    "compute_run_metrics",
    "confusion_matrix",
    "detector_metrics",
    "precision_recall_f1",
    "replay_stability",
    "live_run_variability",
    "Agent",
    "RunnerConfig",
    "execute_task",
    "execute_suite",
    "AdaptiveCampaign",
    "FixedCampaign",
    "CampaignBudget",
    "CampaignPhase",
    "CampaignReport",
    "Scenario",
]
