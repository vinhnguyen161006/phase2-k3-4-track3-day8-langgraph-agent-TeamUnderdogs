"""Checkpointer adapter.

A checkpointer is what turns the graph from a one-shot function into a
resumable workflow: after every super-step LangGraph writes the state to the
configured store, keyed by the ``thread_id`` passed in the run config. That is
what makes HITL interrupts, time-travel replay and crash-resume possible.

Backends
--------
``memory``   — in-process, per-run. Fast, used by tests and CI.
``sqlite``   — file-backed, survives process death. Used for the recovery demo.
``postgres`` — server-backed, for multi-process/production deployments.
``none``     — no persistence; ``thread_id`` is then ignored.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

#: Default on-disk location for the SQLite checkpoint database.
DEFAULT_SQLITE_PATH = "checkpoints.db"


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer for the requested backend.

    Args:
        kind: one of ``none``, ``memory``, ``sqlite``, ``postgres``.
        database_url: SQLite file path, or a Postgres connection URL.

    Raises:
        ValueError: on an unknown backend name.
        RuntimeError: when the backend's optional dependency is not installed.
    """
    if kind == "none":
        return None

    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()

    if kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "SQLite checkpointer requires: pip install langgraph-checkpoint-sqlite"
            ) from exc

        path = Path(database_url or DEFAULT_SQLITE_PATH)
        if path.parent != Path(""):
            path.parent.mkdir(parents=True, exist_ok=True)

        # check_same_thread=False: LangGraph may touch the connection from a
        # worker thread. WAL mode lets a reader inspect history while a run writes.
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return SqliteSaver(conn=conn)

    if kind == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "Postgres checkpointer requires: pip install langgraph-checkpoint-postgres"
            ) from exc

        if not database_url:
            raise ValueError("Postgres checkpointer requires a database_url")

        # from_conn_string() is a context manager that closes the connection on
        # exit, which would kill a long-lived saver. Own the connection instead.
        from psycopg import Connection
        from psycopg.rows import dict_row

        conn = Connection.connect(database_url, autocommit=True, row_factory=dict_row)
        saver = PostgresSaver(conn=conn)
        saver.setup()  # idempotent: creates checkpoint tables on first use
        return saver

    raise ValueError(f"Unknown checkpointer kind: {kind}")
