"""Cross-session learner — pattern detection across event history.

Analyses the EventGraph to surface recurring patterns: which files change
most, which changes tend to break things, which files are changed together,
and what decisions were made and whether they worked.
"""

from __future__ import annotations

import json
from collections import Counter
from itertools import combinations

from .event_graph import EventGraph


class CrossSessionLearner:
    """Discovers patterns by mining the causal event graph."""

    def __init__(self, event_graph: EventGraph) -> None:
        self._graph = event_graph

    # ------------------------------------------------------------------
    # Hot-spot analysis
    # ------------------------------------------------------------------

    def most_changed_files(self, top_n: int = 10) -> list[dict]:
        """Files ranked by number of ``change`` events.

        Returns:
            ``[{"file_path": str, "change_count": int}]`` sorted descending.
        """
        changes = self._all_changes()
        counter: Counter[str] = Counter()
        for ev in changes:
            fp = ev["file_path"]
            if fp:
                counter[fp] += 1
        return [
            {"file_path": fp, "change_count": n}
            for fp, n in counter.most_common(top_n)
        ]

    def most_changed_nodes(self, top_n: int = 10) -> list[dict]:
        """Nodes ranked by number of ``change`` events.

        Returns:
            ``[{"node_id": str, "change_count": int}]`` sorted descending.
        """
        changes = self._all_changes()
        counter: Counter[str] = Counter()
        for ev in changes:
            nid = ev["node_id"]
            if nid:
                counter[nid] += 1
        return [
            {"node_id": nid, "change_count": n}
            for nid, n in counter.most_common(top_n)
        ]

    # ------------------------------------------------------------------
    # Failure analysis
    # ------------------------------------------------------------------

    def failure_prone_files(self, top_n: int = 10) -> list[dict]:
        """Files where changes are frequently followed by failed outcomes.

        For each ``change`` event the method walks its consequences
        (via ``get_consequences``) and looks for ``outcome`` events
        where ``data["success"]`` is ``False``.

        Returns:
            ``[{"file_path": str, "failure_count": int,
                 "change_count": int, "failure_rate": float}]``
            sorted by failure_count descending.
        """
        changes = self._all_changes()

        change_counts: Counter[str] = Counter()
        failure_counts: Counter[str] = Counter()

        for ev in changes:
            fp = ev["file_path"]
            if not fp:
                continue
            change_counts[fp] += 1

            consequences = self._graph.get_consequences(ev["id"])
            for cons in consequences:
                if cons["kind"] == "outcome":
                    data = cons["data"]
                    if isinstance(data, str):
                        try:
                            data = json.loads(data)
                        except (json.JSONDecodeError, TypeError):
                            data = {}
                    if data.get("success") is False:
                        failure_counts[fp] += 1
                        break  # one failure per change is enough

        # Build result for files that have at least one failure.
        results = []
        for fp in failure_counts:
            cc = change_counts[fp]
            fc = failure_counts[fp]
            results.append(
                {
                    "file_path": fp,
                    "failure_count": fc,
                    "change_count": cc,
                    "failure_rate": round(fc / cc, 4) if cc else 0.0,
                }
            )

        results.sort(key=lambda r: r["failure_count"], reverse=True)
        return results[:top_n]

    # ------------------------------------------------------------------
    # Co-change analysis
    # ------------------------------------------------------------------

    def change_pairs(self, top_n: int = 10) -> list[dict]:
        """File pairs frequently changed together in the same session.

        For each session, the unique file paths from ``change`` events are
        collected and every unordered pair is counted.

        Returns:
            ``[{"file_a": str, "file_b": str, "co_change_count": int}]``
            sorted descending.
        """
        changes = self._all_changes()

        # Group file paths by session.
        session_files: dict[str, set[str]] = {}
        for ev in changes:
            sid = ev.get("session_id", "")
            fp = ev["file_path"]
            if sid and fp:
                session_files.setdefault(sid, set()).add(fp)

        pair_counter: Counter[tuple[str, str]] = Counter()
        for files in session_files.values():
            for a, b in combinations(sorted(files), 2):
                pair_counter[(a, b)] += 1

        return [
            {"file_a": a, "file_b": b, "co_change_count": n}
            for (a, b), n in pair_counter.most_common(top_n)
        ]

    # ------------------------------------------------------------------
    # Decision history
    # ------------------------------------------------------------------

    def decision_history(
        self, file_path: str = "", node_id: str = ""
    ) -> list[dict]:
        """All ``decision`` events related to a file or node, with outcomes.

        For each decision its consequences are walked to find linked
        ``outcome`` events.

        Returns:
            ``[{"decision": str, "reasoning": str, "outcome": str,
                 "success": bool | None, "timestamp": str}]``
        """
        if file_path:
            events = self._graph.get_events_for_file(file_path)
        elif node_id:
            events = self._graph.get_events_for_node(node_id)
        else:
            events = self._graph.get_recent_events(limit=1000)

        results = []
        for ev in events:
            if ev["kind"] != "decision":
                continue

            data = ev["data"]
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except (json.JSONDecodeError, TypeError):
                    data = {}

            reasoning = data.get("reasoning", "")

            # Find outcome among consequences.
            outcome_summary = ""
            outcome_success: bool | None = None
            consequences = self._graph.get_consequences(ev["id"])
            for cons in consequences:
                if cons["kind"] == "outcome":
                    outcome_summary = cons["summary"]
                    cdata = cons["data"]
                    if isinstance(cdata, str):
                        try:
                            cdata = json.loads(cdata)
                        except (json.JSONDecodeError, TypeError):
                            cdata = {}
                    outcome_success = cdata.get("success")
                    break

            results.append(
                {
                    "decision": ev["summary"],
                    "reasoning": reasoning,
                    "outcome": outcome_summary,
                    "success": outcome_success,
                    "timestamp": ev["timestamp"],
                }
            )

        return results

    # ------------------------------------------------------------------
    # Session summary
    # ------------------------------------------------------------------

    def session_summary(self, session_id: str) -> dict:
        """Summarise a single session.

        Returns:
            ``{"session_id": str, "decisions": int, "changes": int,
               "outcomes": int, "successes": int, "failures": int,
               "files_touched": list[str]}``
        """
        events = self._graph.get_session_events(session_id)

        decisions = 0
        changes = 0
        outcomes = 0
        successes = 0
        failures = 0
        files: set[str] = set()

        for ev in events:
            kind = ev["kind"]
            if kind == "decision":
                decisions += 1
            elif kind == "change":
                changes += 1
                if ev["file_path"]:
                    files.add(ev["file_path"])
            elif kind == "outcome":
                outcomes += 1
                data = ev["data"]
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except (json.JSONDecodeError, TypeError):
                        data = {}
                if data.get("success") is True:
                    successes += 1
                elif data.get("success") is False:
                    failures += 1

        return {
            "session_id": session_id,
            "decisions": decisions,
            "changes": changes,
            "outcomes": outcomes,
            "successes": successes,
            "failures": failures,
            "files_touched": sorted(files),
        }

    # ------------------------------------------------------------------
    # High-level patterns
    # ------------------------------------------------------------------

    def patterns(self) -> dict:
        """High-level summary combining multiple analyses.

        Returns:
            ``{"most_changed": [...], "failure_prone": [...],
               "co_changes": [...], "total_sessions": int,
               "total_events": int}``
        """
        stats = self._graph.get_stats()
        return {
            "most_changed": self.most_changed_files(top_n=5),
            "failure_prone": self.failure_prone_files(top_n=5),
            "co_changes": self.change_pairs(top_n=5),
            "total_sessions": stats["session_count"],
            "total_events": stats["event_count"],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _all_changes(self) -> list[dict]:
        """Return all ``change`` events from the graph."""
        rows = self._graph._conn.execute(
            "SELECT * FROM events WHERE kind = 'change' ORDER BY timestamp"
        ).fetchall()
        from .event_graph import _row_to_dict

        return [_row_to_dict(r) for r in rows]
