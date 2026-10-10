"""SQLite persistence layer (stdlib sqlite3 — zero extra dependencies).

Three concerns live here:

- ``triage_results``: the idempotency ledger. One row per ticket_id holding the
  exact result we returned. A second request for the same ticket_id reads this
  row and returns it verbatim — the model is never called again and no second
  refund is created.
- ``refund_ledger``: an append-only record of refunds actually executed, keyed
  by an idempotency key, so "it runs twice safely" is enforced at the money
  layer too, not just the response layer.
- ``audit_log``: PII-free, correlation-id-tagged events for 3am debugging.

For production this maps cleanly onto Postgres (see README); the schema is the
same, only the driver changes.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path


def _path_from_url(database_url: str) -> str:
    if database_url.startswith("sqlite:///"):
        return database_url[len("sqlite:///"):]
    return database_url


class Storage:
    def __init__(self, database_url: str):
        self._path = _path_from_url(database_url)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS triage_results (
                    ticket_id     TEXT PRIMARY KEY,
                    result_json   TEXT NOT NULL,
                    created_at    TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS refund_ledger (
                    idempotency_key TEXT PRIMARY KEY,
                    ticket_id       TEXT NOT NULL,
                    amount          REAL NOT NULL,
                    currency        TEXT NOT NULL,
                    rule_id         TEXT NOT NULL,
                    created_at      TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_log (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts             TEXT NOT NULL,
                    correlation_id TEXT,
                    ticket_id      TEXT,
                    event          TEXT NOT NULL,
                    detail         TEXT
                );
                """
            )
            self._conn.commit()

    # --- idempotency: triage results --------------------------------------
    def get_result(self, ticket_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT result_json FROM triage_results WHERE ticket_id = ?",
                (ticket_id,),
            ).fetchone()
        return json.loads(row["result_json"]) if row else None

    def save_result(self, ticket_id: str, result: dict, created_at: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO triage_results(ticket_id, result_json, created_at)"
                " VALUES(?,?,?)",
                (ticket_id, json.dumps(result), created_at),
            )
            self._conn.commit()

    # --- idempotency: refunds ---------------------------------------------
    def refund_exists(self, idempotency_key: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM refund_ledger WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        return row is not None

    def record_refund(
        self, idempotency_key: str, ticket_id: str, amount: float,
        currency: str, rule_id: str, created_at: str,
    ) -> bool:
        """Return True if this call recorded the refund, False if it already
        existed (i.e. a replay — do not pay again)."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO refund_ledger"
                "(idempotency_key, ticket_id, amount, currency, rule_id, created_at)"
                " VALUES(?,?,?,?,?,?)",
                (idempotency_key, ticket_id, amount, currency, rule_id, created_at),
            )
            self._conn.commit()
            return cur.rowcount == 1

    # --- audit -------------------------------------------------------------
    def append_audit(
        self, ts: str, correlation_id: str | None, ticket_id: str | None,
        event: str, detail: dict | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO audit_log(ts, correlation_id, ticket_id, event, detail)"
                " VALUES(?,?,?,?,?)",
                (ts, correlation_id, ticket_id, event,
                 json.dumps(detail) if detail else None),
            )
            self._conn.commit()
