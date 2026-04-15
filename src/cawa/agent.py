"""CAWAAgent: perceive → build causal graph → plan → explain."""

from __future__ import annotations

from typing import Any, Mapping

from .active_inference import Action, Plan, select_action
from .causal_graph import CausalGraph
from .physics import apply_all
from .world import WorldState


class CAWAAgent:
    def __init__(self) -> None:
        self._world: WorldState | None = None

    def perceive(self, world: WorldState) -> None:
        self._world = world

    def build_causal_graph(self) -> CausalGraph:
        if self._world is None:
            raise RuntimeError("agent has not perceived a world yet")
        graph = CausalGraph()
        edges = apply_all(self._world)
        # Register all variables first (parents and effects).
        declared: set[str] = set()
        for edge in edges:
            for var in (*edge.parents, edge.effect):
                if var not in declared:
                    graph.add_variable(var, initial=False)
                    declared.add(var)
        for edge in edges:
            graph.add_mechanism(
                edge.effect,
                list(edge.parents),
                edge.mechanism,
                label=edge.label,
            )
        return graph

    def plan(
        self,
        graph: CausalGraph,
        goal: Mapping[str, Any],
        actions: list[Action],
    ) -> Plan:
        return select_action(graph, actions, goal)

    def explain(self, plan: Plan) -> str:
        lines = [
            f"Chosen action: {plan.action.name}",
            f"  description: {plan.action.description}" if plan.action.description else "",
            f"  EFE = {plan.efe:.2f}  (pragmatic={plan.pragmatic:.2f}, epistemic={plan.epistemic:.2f})",
            "",
            "Causal chain:",
        ]
        for edge in plan.causal_chain:
            parents = ", ".join(edge.parents) if edge.parents else "(root)"
            lines.append(f"  {parents} → {edge.effect}   [{edge.label}]")
        lines.append("")
        lines.append("Counterfactual predictions:")
        for name, state in plan.counterfactuals.items():
            goal_vars = ", ".join(f"{k}={state.get(k)}" for k in state)
            lines.append(f"  {name}: {goal_vars}")
        return "\n".join(line for line in lines if line != "")
