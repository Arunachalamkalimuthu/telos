"""Project Memory — session management and causal event recording.

Wraps EventGraph with session lifecycle, structured recording helpers,
and a query API that answers questions like "why did we make that
decision?" and "what happened last time we changed this file?"
"""

from __future__ import annotations

import uuid

from .event_graph import EventGraph


class ProjectMemory:
    """High-level memory layer over an EventGraph."""

    def __init__(self, db_path: str) -> None:
        self._graph = EventGraph(db_path)
        self._current_session: str | None = None
        self._last_event_id: str | None = None
        self._last_change_id: str | None = None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def start_session(self, description: str = "") -> str:
        """Create a session_start event and return the session id."""
        session_id = f"sess_{uuid.uuid4().hex[:8]}"
        event_id = self._graph.add_event(
            kind="session_start",
            summary=description or "Session started",
            session_id=session_id,
        )
        self._current_session = session_id
        self._last_event_id = event_id
        self._last_change_id = None
        return session_id

    def end_session(self, summary: str = "") -> None:
        """Create a session_end event linked to the session_start."""
        if self._current_session is None:
            return
        event_id = self._graph.add_event(
            kind="session_end",
            summary=summary or "Session ended",
            session_id=self._current_session,
        )
        if self._last_event_id:
            self._graph.link_events(self._last_event_id, event_id, kind="led_to")
        self._current_session = None
        self._last_event_id = None
        self._last_change_id = None

    @property
    def current_session(self) -> str | None:
        """Return the active session id, or None."""
        return self._current_session

    # ------------------------------------------------------------------
    # Recording helpers (auto-link to previous event in session)
    # ------------------------------------------------------------------

    def _auto_link(self, new_event_id: str) -> None:
        """Link the new event to the previous event in the session."""
        if self._last_event_id:
            self._graph.link_events(
                self._last_event_id, new_event_id, kind="led_to"
            )
        self._last_event_id = new_event_id

    def record_decision(
        self,
        summary: str,
        reasoning: str = "",
        file_path: str = "",
        node_id: str = "",
    ) -> str:
        """Record a decision event."""
        event_id = self._graph.add_event(
            kind="decision",
            summary=summary,
            session_id=self._current_session or "",
            data={"reasoning": reasoning},
            file_path=file_path,
            node_id=node_id,
        )
        self._auto_link(event_id)
        return event_id

    def record_change(
        self,
        summary: str,
        file_path: str,
        node_id: str = "",
        diff: str = "",
    ) -> str:
        """Record a code change event."""
        event_id = self._graph.add_event(
            kind="change",
            summary=summary,
            session_id=self._current_session or "",
            data={"diff": diff},
            file_path=file_path,
            node_id=node_id,
        )
        self._auto_link(event_id)
        self._last_change_id = event_id
        return event_id

    def record_outcome(
        self,
        summary: str,
        success: bool,
        file_path: str = "",
        node_id: str = "",
    ) -> str:
        """Record an outcome event.

        On failure, an additional "caused" link is created from the most
        recent change to this outcome.
        """
        event_id = self._graph.add_event(
            kind="outcome",
            summary=summary,
            session_id=self._current_session or "",
            data={"success": success},
            file_path=file_path,
            node_id=node_id,
        )
        self._auto_link(event_id)
        if not success and self._last_change_id:
            self._graph.link_events(
                self._last_change_id, event_id, kind="caused"
            )
        return event_id

    def record_query(self, question: str, answer: str = "") -> str:
        """Record a query event (question asked of the system)."""
        event_id = self._graph.add_event(
            kind="query",
            summary=question,
            session_id=self._current_session or "",
            data={"question": question, "answer": answer},
        )
        # Queries are recorded but don't break the causal chain,
        # so we do NOT update _last_event_id.
        return event_id

    # ------------------------------------------------------------------
    # Querying memory
    # ------------------------------------------------------------------

    def why(self, event_id: str) -> list[dict]:
        """Return the causal chain leading to *event_id* (root cause first)."""
        return self._graph.get_causal_chain(event_id)

    def what_happened(
        self, file_path: str = "", node_id: str = ""
    ) -> list[dict]:
        """Return events related to a file or node, most recent first."""
        if file_path:
            events = self._graph.get_events_for_file(file_path)
        elif node_id:
            events = self._graph.get_events_for_node(node_id)
        else:
            events = self._graph.get_recent_events()
        # EventGraph returns oldest-first; reverse for most-recent-first.
        events.reverse()
        return events

    def last_time(
        self, file_path: str = "", node_id: str = ""
    ) -> dict | None:
        """Return the most recent event for a file or node."""
        events = self.what_happened(file_path=file_path, node_id=node_id)
        return events[0] if events else None

    def search(self, query: str) -> list[dict]:
        """Keyword search across event summaries."""
        return self._graph.search_events(query)

    def recent(self, limit: int = 20) -> list[dict]:
        """Return the most recent events across all sessions."""
        return self._graph.get_recent_events(limit=limit)

    def get_session_history(self, session_id: str = "") -> list[dict]:
        """Full event chain for a session (defaults to current session)."""
        sid = session_id or self._current_session or ""
        if not sid:
            return []
        return self._graph.get_session_events(sid)

    # ------------------------------------------------------------------
    # Stats & lifecycle
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return aggregate stats from the event graph."""
        return self._graph.get_stats()

    def close(self) -> None:
        """Close the underlying database connection."""
        self._graph.close()
