"""Deterministic failure detectors for Failsafe.

Importing this package registers the built-in detectors in the global
:class:`~failsafe.detectors.base.DetectorRegistry`.
"""

from __future__ import annotations

from failsafe.detectors.base import Detector, DetectorRegistry, registry, run_detectors
from failsafe.detectors.excessive_retries import ExcessiveRetriesDetector
from failsafe.detectors.ignored_tool_errors import IgnoredToolErrorsDetector
from failsafe.detectors.lost_context import LostContextDetector
from failsafe.detectors.premature_termination import PrematureTerminationDetector
from failsafe.detectors.repeated_actions import RepeatedActionsDetector
from failsafe.detectors.unsupported_success import UnsupportedSuccessDetector

BUILTIN_DETECTOR_IDS = [
    RepeatedActionsDetector.id,
    ExcessiveRetriesDetector.id,
    IgnoredToolErrorsDetector.id,
    PrematureTerminationDetector.id,
    UnsupportedSuccessDetector.id,
    LostContextDetector.id,
]

__all__ = [
    "Detector",
    "DetectorRegistry",
    "registry",
    "run_detectors",
    "BUILTIN_DETECTOR_IDS",
    "RepeatedActionsDetector",
    "ExcessiveRetriesDetector",
    "IgnoredToolErrorsDetector",
    "PrematureTerminationDetector",
    "UnsupportedSuccessDetector",
    "LostContextDetector",
]
