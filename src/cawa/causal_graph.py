"""Causal DAG with do-calculus. Hand-built per scene, not learned."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


Mechanism = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True)
class CausalEdge:
    parents: tuple[str, ...]
    effect: str
    mechanism: Mechanism
    label: str = ""


class CausalGraph:
    """Directed acyclic graph of causal mechanisms over named variables."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}
        self._edges: list[CausalEdge] = []

    def add_variable(self, name: str, initial: Any = None) -> None:
        if name in self._values:
            raise ValueError(f"variable {name!r} already declared")
        self._values[name] = initial

    def add_mechanism(
        self,
        effect: str,
        parents: list[str],
        mechanism: Mechanism,
        label: str = "",
    ) -> None:
        if effect not in self._values:
            raise ValueError(f"effect variable {effect!r} not declared")
        for p in parents:
            if p not in self._values:
                raise ValueError(f"parent variable {p!r} not declared")
        self._edges.append(CausalEdge(tuple(parents), effect, mechanism, label))

    def get(self, name: str) -> Any:
        return self._values[name]

    def variables(self) -> list[str]:
        return list(self._values.keys())

    def edges_into(self, effect: str) -> list[CausalEdge]:
        return [e for e in self._edges if e.effect == effect]

    def all_edges(self) -> list[CausalEdge]:
        return list(self._edges)

    def _topological_order(self) -> list[str]:
        # Kahn's algorithm.
        indegree: dict[str, int] = {v: 0 for v in self._values}
        children: dict[str, list[str]] = {v: [] for v in self._values}
        for edge in self._edges:
            indegree[edge.effect] += len(edge.parents)
            for p in edge.parents:
                children[p].append(edge.effect)

        queue = [v for v, d in indegree.items() if d == 0]
        order: list[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in children[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(order) != len(self._values):
            raise ValueError("causal graph contains a cycle")
        return order

    def propagate(self) -> dict[str, Any]:
        order = self._topological_order()
        state = dict(self._values)
        for var in order:
            incoming = self.edges_into(var)
            if not incoming:
                continue
            # Variables with multiple incoming edges: compose by the last edge
            # defined (caller is expected to declare a single mechanism per var;
            # additional edges represent joint causes inside one mechanism).
            edge = incoming[-1]
            parent_values = {p: state[p] for p in edge.parents}
            state[var] = edge.mechanism(parent_values)
        return state
