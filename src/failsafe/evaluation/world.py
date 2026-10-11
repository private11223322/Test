"""Synthetic world and tools for oracle-based task verification.

A :class:`SyntheticWorld` is a fully local, deterministic environment: an
in-memory record store ("database"), a controlled search corpus, canned HTTP
responses, and an audit log of every mutation.  Agents interact with it only
through the provided tool functions, so oracle checks can verify outcomes
independently of what the agent claims.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from failsafe.injection.faults import InjectedFault


@dataclass
class AuditEntry:
    op: str
    record_id: str
    fields: dict[str, Any]
    at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "record_id": self.record_id,
            "fields": self.fields,
            "at": self.at.isoformat(),
        }


class SyntheticWorld:
    """Deterministic local environment with a record store and canned services."""

    def __init__(
        self,
        records: dict[str, dict[str, Any]] | None = None,
        corpus: list[dict[str, Any]] | None = None,
        http_responses: dict[str, Any] | None = None,
        clock: datetime | None = None,
    ) -> None:
        self.db: dict[str, dict[str, Any]] = copy.deepcopy(records or {})
        self.corpus: list[dict[str, Any]] = copy.deepcopy(corpus or [])
        self.http_responses: dict[str, Any] = copy.deepcopy(http_responses or {})
        self.audit_log: list[AuditEntry] = []
        self.clock = clock or datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        self.emails_sent: list[dict[str, Any]] = []
        self.call_log: list[dict[str, Any]] = []

    # ---- mutation helpers -------------------------------------------
    def _log(self, op: str, record_id: str, fields: dict[str, Any]) -> None:
        self.audit_log.append(AuditEntry(op=op, record_id=record_id, fields=dict(fields), at=self.clock))

    def mutation_count(self, record_id: str | None = None, op: str | None = None) -> int:
        n = 0
        for e in self.audit_log:
            if record_id is not None and e.record_id != record_id:
                continue
            if op is not None and e.op != op:
                continue
            n += 1
        return n

    def snapshot(self) -> dict[str, Any]:
        return {
            "db": copy.deepcopy(self.db),
            "audit_log": [e.to_dict() for e in self.audit_log],
            "emails_sent": copy.deepcopy(self.emails_sent),
        }

    # ---- tool implementations ---------------------------------------
    def db_get(self, record_id: str) -> dict[str, Any]:
        self.call_log.append({"tool": "db_get", "args": {"record_id": record_id}})
        if record_id not in self.db:
            raise KeyError(f"record not found: {record_id}")
        return copy.deepcopy(self.db[record_id])

    def db_update(self, record_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        self.call_log.append({"tool": "db_update", "args": {"record_id": record_id, "fields": fields}})
        if record_id not in self.db:
            raise KeyError(f"record not found: {record_id}")
        self.db[record_id].update(copy.deepcopy(fields))
        self._log("update", record_id, fields)
        return copy.deepcopy(self.db[record_id])

    def db_insert(self, record_id: str, record: dict[str, Any]) -> dict[str, Any]:
        self.call_log.append({"tool": "db_insert", "args": {"record_id": record_id, "record": record}})
        if record_id in self.db:
            raise ValueError(f"record already exists: {record_id}")
        self.db[record_id] = copy.deepcopy(record)
        self._log("insert", record_id, dict(record))
        return copy.deepcopy(self.db[record_id])

    def db_delete(self, record_id: str) -> dict[str, Any]:
        self.call_log.append({"tool": "db_delete", "args": {"record_id": record_id}})
        if record_id not in self.db:
            raise KeyError(f"record not found: {record_id}")
        del self.db[record_id]
        self._log("delete", record_id, {})
        return {"deleted": record_id}

    def search(self, query: str) -> list[dict[str, Any]]:
        self.call_log.append({"tool": "search", "args": {"query": query}})
        q = query.lower().split()
        results = []
        for doc in self.corpus:
            text = f"{doc.get('title', '')} {doc.get('body', '')}".lower()
            if all(term in text for term in q):
                results.append(copy.deepcopy(doc))
        return results

    def fetch_url(self, url: str) -> dict[str, Any]:
        self.call_log.append({"tool": "fetch_url", "args": {"url": url}})
        if url not in self.http_responses:
            raise KeyError(f"no canned response for url: {url}")
        return copy.deepcopy(self.http_responses[url])

    def send_email(self, to: str, body: str) -> dict[str, Any]:
        self.call_log.append({"tool": "send_email", "args": {"to": to, "body": body}})
        entry = {"to": to, "body": body, "at": self.clock.isoformat()}
        self.emails_sent.append(entry)
        return {"status": "sent", "to": to}

    def advance_clock(self, seconds: float) -> datetime:
        self.clock = self.clock + timedelta(seconds=seconds)
        return self.clock

    def tool_map(self, injector: Any = None) -> dict[str, Callable[..., Any]]:
        """Return the tool callables, optionally wrapped by a fault injector."""
        tools: dict[str, Callable[..., Any]] = {
            "db_get": self.db_get,
            "db_update": self.db_update,
            "db_insert": self.db_insert,
            "db_delete": self.db_delete,
            "search": self.search,
            "fetch_url": self.fetch_url,
            "send_email": self.send_email,
        }
        if injector is not None:
            tools = {name: injector.wrap(fn, name) for name, fn in tools.items()}
        return tools


def stale_http_response(url: str, age_seconds: int = 3600) -> dict[str, Any]:
    """Build a canned response whose content is explicitly stale."""
    return {
        "url": url,
        "body": {"value": "stale", "stale": True},
        "fetched_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
        "age_seconds": age_seconds,
    }


def make_demo_world() -> SyntheticWorld:
    """A small world used by the sample tasks and the e2e demo."""
    records = {
        "u-1001": {"id": "u-1001", "name": "Ada Lovelace", "email": "ada@example.com", "plan": "free", "active": True},
        "u-1002": {"id": "u-1002", "name": "Grace Hopper", "email": "grace@example.com", "plan": "pro", "active": True},
        "u-1003": {"id": "u-1003", "name": "Alan Turing", "email": "alan@example.com", "plan": "free", "active": False},
        "inv-9": {"id": "inv-9", "sku": "WIDGET-9", "qty": 3, "warehouse": "east"},
    }
    corpus = [
        {"id": "doc-1", "title": "Refund policy", "body": "Refunds are processed within 14 days of purchase."},
        {"id": "doc-2", "title": "Shipping policy", "body": "Standard shipping takes 5 to 7 business days."},
        {"id": "doc-3", "title": "Password reset", "body": "Reset your password using the emailed secure link."},
    ]
    http = {
        "https://api.example.com/exchange": {"rates": {"USD": 1.0, "EUR": 0.92}, "date": "2026-01-01"},
    }
    return SyntheticWorld(records=records, corpus=corpus, http_responses=http)


def simulate_fault_in_tool(tool_fn: Callable[..., Any], fault: InjectedFault) -> Callable[..., Any]:
    """Wrap a tool so the first call raises the given injected fault."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        wrapper.called = getattr(wrapper, "called", 0) + 1
        if wrapper.called == 1:
            raise fault
        return tool_fn(*args, **kwargs)

    return wrapper
