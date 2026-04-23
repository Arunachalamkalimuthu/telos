"""Counterfactual engine: 'what if we intervene at node X?'."""

from __future__ import annotations

from collections import deque

from telos.code_parser.store import GraphStore
from telos.impact.analyzer import ImpactAnalyzer


class CounterfactualAnalyzer:
    """Compares normal blast radius with a hypothetical intervention."""

    def __init__(self, store: GraphStore, analyzer: ImpactAnalyzer) -> None:
        self._store = store
        self._analyzer = analyzer

    def analyze(self, target: str, intervention_at: str) -> dict:
        """Evaluate 'what if we add a fallback at *intervention_at*?'

        1. Run normal impact analysis from *target* (without fix).
        2. Determine which affected nodes are reachable **only** through
           *intervention_at* — those are "saved" by the fix.
        3. Return comparative report.
        """
        without_fix = self._analyzer.analyze(target)

        # Nodes reachable from target while skipping outgoing edges of
        # intervention_at.
        still_reachable = self._reachable_without(target, intervention_at)

        saved_ids = set()
        with_fix_affected: list[dict] = []
        for entry in without_fix["affected"]:
            if entry["node_id"] in still_reachable:
                with_fix_affected.append(entry)
            else:
                saved_ids.add(entry["node_id"])

        with_fix = dict(without_fix)
        with_fix["affected"] = with_fix_affected
        with_fix["files_affected"] = sorted(
            {e["file_path"] for e in with_fix_affected if e["file_path"]}
        )

        return {
            "target": target,
            "intervention_at": intervention_at,
            "without_fix": without_fix,
            "with_fix": with_fix,
            "reduction": len(saved_ids),
            "without_count": len(without_fix["affected"]),
            "with_count": len(with_fix_affected),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reachable_without(
        self, target: str, skip_node: str
    ) -> set[str]:
        """BFS from *target*, but do NOT follow outgoing edges of *skip_node*.

        Returns the set of node ids that are still reachable.
        """
        visited: set[str] = set()
        queue: deque[str] = deque([target])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            # Do not expand edges from the intervention point
            if current == skip_node:
                continue

            for edge in self._store.get_edges_from(current):
                child = edge["target"]
                if child not in visited:
                    queue.append(child)

        # Remove the seed itself — we only care about affected (non-seed)
        visited.discard(target)
        return visited
