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
