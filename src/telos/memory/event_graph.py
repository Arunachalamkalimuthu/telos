"""Causal event graph backed by SQLite.

Tracks sessions, decisions, code changes, and outcomes as a traversable
graph so you can answer questions like "why did we make that decision?"
and "what happened last time we changed this?"
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime


_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    session_id  TEXT,
    summary     TEXT NOT NULL,
    data        TEXT DEFAULT '{}',
    file_path   TEXT DEFAULT '',
    node_id     TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS event_links (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    source  TEXT REFERENCES events(id),
    target  TEXT REFERENCES events(id),
    kind    TEXT NOT NULL,
    weight  REAL DEFAULT 1.0
);

CREATE INDEX IF NOT EXISTS idx_events_session_id ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_kind       ON events(kind);
CREATE INDEX IF NOT EXISTS idx_events_node_id    ON events(node_id);
CREATE INDEX IF NOT EXISTS idx_links_source      ON event_links(source);
CREATE INDEX IF NOT EXISTS idx_links_target      ON event_links(target);
"""


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    if "data" in d:
        try:
            d["data"] = json.loads(d["data"])
        except (json.JSONDecodeError, TypeError):
            d["data"] = {}
    return d


class EventGraph:
    """A causal event graph persisted in SQLite."""

    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Event CRUD
    # ------------------------------------------------------------------

    def add_event(
        self,
        kind: str,
        summary: str,
        session_id: str = "",
        data: dict | None = None,
        file_path: str = "",
        node_id: str = "",
    ) -> str:
        """Insert a new event and return its id."""
        event_id = uuid.uuid4().hex[:12]
        timestamp = datetime.now().isoformat()
        self._conn.execute(
            "INSERT INTO events (id, kind, timestamp, session_id, summary, data, file_path, node_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                kind,
                timestamp,
                session_id,
                summary,
                json.dumps(data or {}),
                file_path,
                node_id,
            ),
        )
        self._conn.commit()
        return event_id

    def get_event(self, event_id: str) -> dict | None:
        """Retrieve a single event by id, or None if not found."""
        row = self._conn.execute(
            "SELECT * FROM events WHERE id = ?", (event_id,)
        ).fetchone()
        return _row_to_dict(row) if row else None

    def link_events(
        self,
        source_id: str,
        target_id: str,
        kind: str = "caused",
        weight: float = 1.0,
    ) -> None:
        """Create a directed causal link from source to target."""
        self._conn.execute(
            "INSERT INTO event_links (source, target, kind, weight) VALUES (?, ?, ?, ?)",
            (source_id, target_id, kind, weight),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_session_events(self, session_id: str) -> list[dict]:
        """All events in a session, ordered by timestamp."""
        rows = self._conn.execute(
            "SELECT * FROM events WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_events_for_node(self, node_id: str) -> list[dict]:
        """All events related to a code graph node."""
        rows = self._conn.execute(
            "SELECT * FROM events WHERE node_id = ? ORDER BY timestamp",
            (node_id,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_events_for_file(self, file_path: str) -> list[dict]:
        """All events related to a file."""
        rows = self._conn.execute(
            "SELECT * FROM events WHERE file_path = ? ORDER BY timestamp",
            (file_path,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_causal_chain(self, event_id: str) -> list[dict]:
        """Walk event_links backwards to find the root cause chain.

        Follows target -> source links recursively (max depth 20).
        Returns the chain from root cause to the given event.
        """
        chain: list[dict] = []
        visited: set[str] = set()
        current = event_id

        for _ in range(20):
            if current in visited:
                break
            visited.add(current)

            row = self._conn.execute(
                "SELECT source FROM event_links WHERE target = ?", (current,)
            ).fetchone()
            if not row:
                break

            source_id = row["source"]
            event = self.get_event(source_id)
            if event:
                chain.append(event)
            current = source_id

        chain.reverse()
        return chain

    def get_consequences(self, event_id: str) -> list[dict]:
        """Walk event_links forward from event to find what it caused.

        Follows source -> target links recursively (max depth 20).
        """
        results: list[dict] = []
        visited: set[str] = set()
        queue = [event_id]

        for _ in range(20):
            if not queue:
                break
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            rows = self._conn.execute(
                "SELECT target FROM event_links WHERE source = ?", (current,)
            ).fetchall()
            for row in rows:
                target_id = row["target"]
                if target_id not in visited:
                    event = self.get_event(target_id)
                    if event:
                        results.append(event)
                    queue.append(target_id)

        return results

    def get_recent_events(self, limit: int = 20) -> list[dict]:
        """Most recent events across all sessions."""
        rows = self._conn.execute(
            "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def search_events(self, query: str) -> list[dict]:
        """Simple LIKE search on the summary field."""
        rows = self._conn.execute(
            "SELECT * FROM events WHERE summary LIKE ? ORDER BY timestamp",
            (f"%{query}%",),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return event_count, link_count, and session_count."""
        event_count = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        link_count = self._conn.execute("SELECT COUNT(*) FROM event_links").fetchone()[0]
        session_count = self._conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM events WHERE session_id != ''"
        ).fetchone()[0]
        return {
            "event_count": event_count,
            "link_count": link_count,
            "session_count": session_count,
        }

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
