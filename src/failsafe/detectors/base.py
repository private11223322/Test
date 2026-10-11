"""Detector interface and registry.

Every detector implements :class:`Detector` and is registered in the
:class:`DetectorRegistry` so new detectors can be added without modifying
unrelated components.  Detectors must be deterministic: the same trace and
configuration produce the same findings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from failsafe.config import DetectorConfig
from failsafe.schemas import DetectorFinding, Evidence, FindingStatus, Severity, Trace


class Detector(ABC):
    """Documented detector interface.

    Subclasses must set ``id``, ``category``, and ``description`` and
    implement :meth:`detect`.  ``known_limitations`` documents what the
    detector cannot see.
    """

    id: str = "base"
    category: str = "base"
    description: str = ""
    known_limitations: list[str] = []

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()

    @abstractmethod
    def detect(self, trace: Trace) -> list[DetectorFinding]:
        """Return findings for the given trace.  Must be deterministic."""

    def make_finding(
        self,
        trace: Trace,
        summary: str,
        *,
        evidence: list[Evidence] | None = None,
        severity: Severity = Severity.MEDIUM,
        status: FindingStatus = FindingStatus.RULE_TRIGGERED,
        description: str = "",
        metrics: dict[str, Any] | None = None,
        category: str | None = None,
    ) -> DetectorFinding:
        return DetectorFinding(
            run_id=trace.metadata.run_id,
            detector_id=self.id,
            category=category or self.category,
            severity=severity,
            status=status,
            summary=summary,
            description=description or self.description,
            evidence=evidence or [],
            metrics=metrics or {},
        )


class DetectorRegistry:
    """Registry mapping detector ids to detector classes."""

    def __init__(self) -> None:
        self._registry: dict[str, type[Detector]] = {}

    def register(self, cls: type[Detector]) -> type[Detector]:
        if cls.id in self._registry:
            raise ValueError(f"Detector id already registered: {cls.id}")
        self._registry[cls.id] = cls
        return cls

    def create(self, detector_id: str, config: DetectorConfig | None = None) -> Detector:
        if detector_id not in self._registry:
            raise KeyError(f"Unknown detector: {detector_id}. Known: {sorted(self._registry)}")
        return self._registry[detector_id](config)

    def create_all(self, config: DetectorConfig | None = None) -> list[Detector]:
        return [cls(config) for cls in self._registry.values()]

    def ids(self) -> list[str]:
        return sorted(self._registry)

    def __contains__(self, detector_id: str) -> bool:
        return detector_id in self._registry


# Global registry populated by failsafe.detectors.__init__
registry = DetectorRegistry()


def run_detectors(
    trace: Trace,
    config: DetectorConfig | None = None,
    detector_ids: Iterable[str] | None = None,
) -> list[DetectorFinding]:
    """Run all (or selected) registered detectors over a trace."""
    if detector_ids is None:
        detectors = registry.create_all(config)
    else:
        detectors = [registry.create(did, config) for did in detector_ids]
    findings: list[DetectorFinding] = []
    for det in detectors:
        findings.extend(det.detect(trace))
    # deterministic ordering: by detector id, then summary, then id
    findings.sort(key=lambda f: (f.detector_id, f.created_at.isoformat(), f.id))
    return findings
