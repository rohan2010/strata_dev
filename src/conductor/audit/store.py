"""Append-only audit log — the spine every other component writes through.

Two invariants, both load-bearing for FR-A1 ("trace every run"):

1. **Append-only.** There is no update or delete path. Corrections are new events.
2. **Write-ahead.** Actions are logged *before* they execute, then a result event is
   appended after. A crash between the two leaves a `tool_call` with no `tool_result`,
   which is the correct and readable signal that something died mid-flight. Logging
   after the fact would silently lose exactly the events you most want.

Storage is SQLite for the Phase 0 spike. The schema is deliberately Postgres-portable
(no SQLite-only types, JSON held as TEXT, timestamps as ISO-8601 UTC strings) so the
Phase 1 move to Postgres is a driver swap rather than a migration rewrite.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path(".conductor/conductor.db")


class EventType(StrEnum):
    """Every distinct thing that can happen in a run.

    Kept as a closed set so the run view can render each type deliberately rather
    than falling back to a generic JSON dump.
    """

    RUN_STARTED = "run_started"
    PLAN_CREATED = "plan_created"
    DELEGATION = "delegation"
    SUBAGENT_RESULT = "subagent_result"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_DECIDED = "approval_decided"
    MODEL_USAGE = "model_usage"
    ERROR = "error"
    RUN_FINISHED = "run_finished"


class RunStatus(StrEnum):
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    goal         TEXT NOT NULL,
    status       TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    finished_at  TEXT,
    summary      TEXT
);

CREATE TABLE IF NOT EXISTS run_events (
    event_id        TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    seq             INTEGER NOT NULL,
    ts              TEXT NOT NULL,
    actor           TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    payload         TEXT NOT NULL,
    parent_event_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_run_events_run ON run_events(run_id, seq);
CREATE INDEX IF NOT EXISTS idx_run_events_type ON run_events(run_id, event_type);
"""


@dataclass(frozen=True)
class Event:
    event_id: str
    run_id: str
    seq: int
    ts: str
    actor: str
    event_type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    parent_event_id: str | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class AuditStore:
    """Owns the connection and the append-only guarantee.

    Thread-safe. LangGraph runs an independent dependency layer's tasks on worker threads
    (FR-S3), and every one of them writes here — so a single connection guarded by a lock,
    rather than SQLite's default same-thread-only connection.

    A lock rather than a connection pool because the sequence number is assigned by reading
    `MAX(seq)` and inserting: two concurrent writers would otherwise race and produce
    duplicate `seq` values within a run, which silently corrupts event ordering in the
    trace. Serializing writes is the cheap correct answer at this volume; the Postgres move
    replaces it with a sequence and can drop the lock.
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self.db_path, isolation_level=None, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- runs -------------------------------------------------------------

    def start_run(self, goal: str) -> str:
        run_id = _new_id("run")
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs (run_id, goal, status, created_at) VALUES (?, ?, ?, ?)",
                (run_id, goal, RunStatus.RUNNING, _now()),
            )
        self.append(run_id, actor="user", event_type=EventType.RUN_STARTED, payload={"goal": goal})
        return run_id

    def finish_run(self, run_id: str, status: RunStatus, summary: str | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET status = ?, finished_at = ?, summary = ? WHERE run_id = ?",
                (status, _now(), summary, run_id),
            )
        self.append(
            run_id,
            actor="orchestrator",
            event_type=EventType.RUN_FINISHED,
            payload={"status": str(status), "summary": summary},
        )

    def set_status(self, run_id: str, status: RunStatus) -> None:
        with self._lock:
            self._conn.execute("UPDATE runs SET status = ? WHERE run_id = ?", (status, run_id))

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- events -----------------------------------------------------------

    def append(
        self,
        run_id: str,
        *,
        actor: str,
        event_type: EventType,
        payload: dict[str, Any] | None = None,
        parent_event_id: str | None = None,
    ) -> str:
        """Append one event. Returns the event id so callers can parent a result to it."""
        event_id = _new_id("evt")
        # Read-then-insert must be atomic: concurrent subagent threads would otherwise
        # both read the same MAX(seq) and write duplicate sequence numbers.
        with self._lock:
            seq_row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 AS next FROM run_events WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            self._conn.execute(
                "INSERT INTO run_events "
                "(event_id, run_id, seq, ts, actor, event_type, payload, parent_event_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event_id,
                    run_id,
                    seq_row["next"],
                    _now(),
                    actor,
                    str(event_type),
                    json.dumps(payload or {}, default=str),
                    parent_event_id,
                ),
            )
        return event_id

    def events(self, run_id: str) -> list[Event]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM run_events WHERE run_id = ? ORDER BY seq", (run_id,)
            ).fetchall()
        return [
            Event(
                event_id=r["event_id"],
                run_id=r["run_id"],
                seq=r["seq"],
                ts=r["ts"],
                actor=r["actor"],
                event_type=EventType(r["event_type"]),
                payload=json.loads(r["payload"]),
                parent_event_id=r["parent_event_id"],
            )
            for r in rows
        ]

    # ---- write-ahead helper ----------------------------------------------

    @contextmanager
    def tool_call(
        self, run_id: str, *, actor: str, tool: str, arguments: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        """Log a tool call before it runs, and its outcome after.

        Usage::

            with store.tool_call(run_id, actor="implementer", tool="gh.create_pr",
                                 arguments=args) as outcome:
                outcome["result"] = do_the_thing()

        On an exception the error is recorded and re-raised — the caller still sees
        the failure, and the trace still shows what was attempted.
        """
        call_id = self.append(
            run_id,
            actor=actor,
            event_type=EventType.TOOL_CALL,
            payload={"tool": tool, "arguments": arguments},
        )
        outcome: dict[str, Any] = {"result": None}
        try:
            yield outcome
        except Exception as exc:
            self.append(
                run_id,
                actor=actor,
                event_type=EventType.ERROR,
                payload={"tool": tool, "error": repr(exc)},
                parent_event_id=call_id,
            )
            raise
        else:
            self.append(
                run_id,
                actor=actor,
                event_type=EventType.TOOL_RESULT,
                payload={"tool": tool, "result": outcome["result"]},
                parent_event_id=call_id,
            )

    # ---- cost accounting (FR-A3) -----------------------------------------

    def record_usage(self, run_id: str, *, actor: str, model: str, usage: dict[str, Any]) -> None:
        self.append(
            run_id,
            actor=actor,
            event_type=EventType.MODEL_USAGE,
            payload={"model": model, **usage},
        )

    #: Usage counters summed per actor. These are LangChain's normalized names, which is
    #: what makes one breakdown span Anthropic, OpenAI, Gemini, and Ollama runs.
    USAGE_COUNTERS = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_read",
        "cache_creation",
        "reasoning_tokens",
    )

    def cost_breakdown(self, run_id: str) -> dict[str, dict[str, Any]]:
        """Token and dollar totals per actor, across every provider used in the run.

        `cost_usd` is summed only from calls whose model had a price in the registry.
        `models` lists what actually served each actor — with fallback chains, that is not
        knowable from config alone. `unpriced_calls` counts the calls contributing tokens
        but no dollars, so a cost dashboard can say "$X across N priced calls, M unpriced"
        rather than implying the total is complete.
        """
        totals: dict[str, dict[str, Any]] = {}
        for ev in self.events(run_id):
            if ev.event_type is not EventType.MODEL_USAGE:
                continue
            bucket = totals.setdefault(
                ev.actor,
                {k: 0 for k in self.USAGE_COUNTERS}
                | {"cost_usd": 0.0, "unpriced_calls": 0, "models": []},
            )
            for key in self.USAGE_COUNTERS:
                bucket[key] += int(ev.payload.get(key) or 0)

            cost = ev.payload.get("cost_usd")
            if cost is None:
                bucket["unpriced_calls"] += 1
            else:
                bucket["cost_usd"] += float(cost)

            model = ev.payload.get("model_alias") or ev.payload.get("model")
            if model and model not in bucket["models"]:
                bucket["models"].append(model)
        return totals

    def run_cost_usd(self, run_id: str) -> tuple[float, int]:
        """(total dollars, number of unpriced calls) for the whole run."""
        breakdown = self.cost_breakdown(run_id)
        return (
            sum(b["cost_usd"] for b in breakdown.values()),
            sum(b["unpriced_calls"] for b in breakdown.values()),
        )
