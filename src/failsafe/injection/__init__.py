"""Controlled fault injection for Failsafe."""

from __future__ import annotations

from failsafe.injection.faults import FaultSchedule, FaultSpec, InjectedFault
from failsafe.injection.injector import FaultyResult, InjectionLog, ToolFaultInjector, wrap_soft
from failsafe.injection.replay import Cassette, Recorder, Replayer

__all__ = [
    "FaultSchedule",
    "FaultSpec",
    "InjectedFault",
    "ToolFaultInjector",
    "InjectionLog",
    "FaultyResult",
    "wrap_soft",
    "Cassette",
    "Recorder",
    "Replayer",
]
