"""Impact analysis via BFS traversal of the dependency graph."""

from __future__ import annotations

from collections import deque

from telos.code_parser.store import GraphStore


class ImpactAnalyzer:
    """Walks outgoing edges from a target node to find transitive impact."""

    def __init__(self, store: GraphStore) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        target: str,
        max_depth: int | None = None,
        min_risk: float = 0.0,
    ) -> dict:
        """BFS from *target* through outgoing edges.

        Returns
        -------
        dict with keys:
            target         – the seed node id
            affected       – list[dict] with node_id, risk, depth, edge_kind,
                             file_path, via
            hottest_path   – list[str] path from target to the deepest
                             high-risk node
            files_affected – deduplicated list of file paths
        """
        if self._store.get_node(target) is None:
            return {
                "target": target,
                "affected": [],
                "hottest_path": [],
                "files_affected": [],
            }

        # BFS state: node_id -> {risk, depth, edge_kind, via, file_path}
        best: dict[str, dict] = {}
        # For hottest-path reconstruction: node -> predecessor
        parent: dict[str, str | None] = {}
        queue: deque[tuple[str, float, int, str | None]] = deque()

        # Seed: enqueue all direct neighbours of target
        for edge in self._store.get_edges_from(target):
            neighbour = edge["target"]
            risk = edge["weight"]
            queue.append((neighbour, risk, 1, target))

        parent[target] = None

        while queue:
            node_id, risk, depth, via = queue.popleft()

            if max_depth is not None and depth > max_depth:
                continue

            # Deduplicate: keep highest risk per node
            if node_id in best and best[node_id]["risk"] >= risk:
                continue

            node_info = self._store.get_node(node_id)
            edge_kind = ""
            file_path = ""
            if via is not None:
                # Find the edge that brought us here
                for e in self._store.get_edges_from(via):
                    if e["target"] == node_id:
                        edge_kind = e["kind"]
                        file_path = e.get("file_path", "")
                        break
            if node_info:
                file_path = file_path or node_info.get("file_path", "")

            best[node_id] = {
                "node_id": node_id,
                "risk": risk,
                "depth": depth,
                "edge_kind": edge_kind,
                "file_path": node_info.get("file_path", "") if node_info else "",
                "via": via,
            }
            parent[node_id] = via

            # Expand neighbours
            for edge in self._store.get_edges_from(node_id):
                child = edge["target"]
                child_risk = risk * edge["weight"]
                queue.append((child, child_risk, depth + 1, node_id))

        # Apply min_risk filter
        affected = [v for v in best.values() if v["risk"] >= min_risk]

        # Sort: risk desc, then depth asc
        affected.sort(key=lambda x: (-x["risk"], x["depth"]))

        # Hottest path: find the node with highest depth among highest-risk
        # nodes, then walk parent pointers back to target.
        hottest_path: list[str] = []
        if affected:
            # Pick the node that is deepest among max-risk entries,
            # then among ties pick the one with highest risk.
            endpoint = max(affected, key=lambda x: (x["depth"], x["risk"]))
            node = endpoint["node_id"]
            while node is not None:
                hottest_path.append(node)
                node = parent.get(node)
            hottest_path.reverse()

        # Collect unique file paths
        files_affected = sorted(
            {
                entry["file_path"]
                for entry in affected
                if entry["file_path"]
            }
        )

        return {
            "target": target,
            "affected": affected,
            "hottest_path": hottest_path,
            "files_affected": files_affected,
        }

    def hotspots(self, top_n: int = 10) -> list[dict]:
        """Return the *top_n* nodes with the most incoming dependents."""
        counts: dict[str, int] = {}
        for edge in self._store.get_all_edges():
            tgt = edge["target"]
            counts[tgt] = counts.get(tgt, 0) + 1

        ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:top_n]
        result: list[dict] = []
        for node_id, dep_count in ranked:
            node = self._store.get_node(node_id)
            result.append(
                {
                    "node_id": node_id,
                    "name": node["name"] if node else node_id,
                    "file_path": node["file_path"] if node else "",
                    "dependent_count": dep_count,
                }
            )
        return result
