"""Configuration for Failsafe.

Everything is driven by a dataclass-like Pydantic settings object that can be
loaded from environment variables, a JSON/YAML-free config file (JSON or TOML
subset via simple parsing), or constructed programmatically.  No secrets are
stored here; LLM API keys are read from the environment only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_CONFIG_NAMES = ("failsafe.json", ".failsafe.json", "failsafe.config.json")


class DetectorConfig(BaseModel):
    """Thresholds for the deterministic detectors."""

    model_config = ConfigDict(extra="forbid")

    # Repeated actions
    repeat_min_count: int = 3
    repeat_normalize_args: bool = True
    repeat_window: int | None = None  # None = whole run

    # Excessive retries
    retry_budget: int = 3

    # Ignored tool errors
    ignored_error_allow_actions: list[str] = Field(
        default_factory=lambda: ["retry", "fallback", "escalate", "report_failure"]
    )
    require_failure_report: bool = True

    # Premature termination
    premature_require_task_spec: bool = True

    # Unsupported success claims
    success_claim_max_unverified: int = 0

    # Lost / repeated context
    context_repeat_min_count: int = 2


class InjectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 1337
    max_injections: int = 5
    allow_real_systems: bool = False
    fault_weights: dict[str, int] = Field(
        default_factory=lambda: {
            "timeout": 1,
            "error": 1,
            "empty_response": 1,
            "malformed_response": 1,
            "stale_response": 1,
            "rate_limit": 1,
            "partial_result": 1,
            "schema_drift": 1,
        }
    )


class BudgetConfig(BaseModel):
    """Hard budgets for agentic / evaluation components."""

    model_config = ConfigDict(extra="forbid")

    max_iterations: int = 12
    max_tool_calls: int = 30
    max_llm_calls: int = 20
    max_seconds: float = 120.0
    max_estimated_cost_usd: float = 1.0
    max_tokens: int = 100_000


class CIGateConfig(BaseModel):
    """Deterministic CI gate criteria.  An LLM judge can never set these."""

    model_config = ConfigDict(extra="forbid")

    mode: str = "strict"  # "strict" | "report-only"
    fail_on_categories: list[str] = Field(
        default_factory=lambda: ["repeated_actions", "excessive_retries", "ignored_tool_errors"]
    )
    fail_on_min_severity: str = "high"
    fail_on_oracle_failure: bool = True
    fail_on_silent_failure: bool = True


class FailsafeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    db_url: str = "sqlite:///failsafe.db"
    data_dir: str = ".failsafe"
    trace_max_bytes: int = 5_000_000
    trace_max_steps: int = 5_000
    redact_secrets: bool = True
    detectors: DetectorConfig = Field(default_factory=DetectorConfig)
    injection: InjectionConfig = Field(default_factory=InjectionConfig)
    budgets: BudgetConfig = Field(default_factory=BudgetConfig)
    ci: CIGateConfig = Field(default_factory=CIGateConfig)
    # LLM provider settings (optional; no keys stored in config)
    llm_provider: str = "openai"  # "openai" | "echo" | "disabled"
    llm_model: str = "gpt-4o-mini"
    llm_enabled: bool = False
    reports_dir: str = "reports"

    def ensure_dirs(self) -> None:
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.reports_dir).mkdir(parents=True, exist_ok=True)


def _load_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".json",):
        return json.loads(text)
    if path.suffix in (".toml",):
        try:
            import tomllib

            return tomllib.loads(text)
        except Exception:
            return {}
    # try JSON anyway
    try:
        return json.loads(text)
    except Exception:
        return {}


def find_config_file(start: str | Path | None = None) -> Path | None:
    start_path = Path(start) if start else Path.cwd()
    for name in DEFAULT_CONFIG_NAMES:
        candidate = start_path / name
        if candidate.is_file():
            return candidate
    return None


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> FailsafeConfig:
    """Load configuration from file, then environment, then explicit overrides.

    Environment variables use the prefix ``FAILSAFE_`` and ``__`` for nesting,
    e.g. ``FAILSAFE_DETECTORS__RETRY_BUDGET=5``.
    """
    data: dict[str, Any] = {}
    cfg_path = Path(path) if path else find_config_file()
    if cfg_path and cfg_path.is_file():
        data = _load_file(cfg_path)

    # Environment overrides
    env_data = _env_overrides(os.environ)
    _deep_merge(data, env_data)

    if overrides:
        _deep_merge(data, overrides)

    return FailsafeConfig.model_validate(data)


def _env_overrides(env: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in env.items():
        if not key.startswith("FAILSAFE_"):
            continue
        rest = key[len("FAILSAFE_") :]
        if rest in ("DB_URL",):
            out["db_url"] = value
            continue
        parts = rest.lower().split("__")
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        leaf = parts[-1]
        node[leaf] = _coerce(value)
    # Fix single-level known keys that used no nesting
    for k in list(out.keys()):
        if k in ("db_url", "data_dir", "llm_provider", "llm_model", "reports_dir"):
            continue
        if isinstance(out[k], str) and k in ("llm_enabled", "redact_secrets"):
            out[k] = out[k].lower() in ("1", "true", "yes")
    return out


def _coerce(value: str) -> Any:
    lowered = value.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _deep_merge(base: dict[str, Any], other: dict[str, Any]) -> None:
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
