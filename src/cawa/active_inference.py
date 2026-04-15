"""Active Inference: choose actions that minimise expected free energy.

EFE(a) = -(pragmatic(a) + epistemic(a))

Higher pragmatic + epistemic values mean lower EFE, i.e. more preferred.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .causal_graph import CausalEdge, CausalGraph


@dataclass(frozen=True)
class Action:
    name: str
    effects: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""


def pragmatic_value(state: Mapping[str, Any], goal: Mapping[str, Any]) -> float:
    """Negative count of goal variables whose values diverge from the goal."""
    mismatches = 0
    for var, wanted in goal.items():
        if state.get(var) != wanted:
            mismatches += 1
    return -float(mismatches)


def epistemic_value(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    unknown_marker: Any = "UNKNOWN",
) -> float:
    """Count of variables that become known (were UNKNOWN, now aren't)."""
    resolved = 0
    for var, v_before in before.items():
        if v_before == unknown_marker and after.get(var, unknown_marker) != unknown_marker:
            resolved += 1
    return float(resolved)


@dataclass
class Plan:
    action: Action
    efe: float
    pragmatic: float
    epistemic: float
    causal_chain: list[CausalEdge]
    counterfactuals: dict[str, dict[str, Any]]
    chosen_state: dict[str, Any]


def select_action(
    graph: CausalGraph,
    actions: list[Action],
    goal: Mapping[str, Any],
) -> Plan:
    """Return the action minimising expected free energy."""
    if not actions:
        raise ValueError("no actions to choose from")

    baseline_state = graph.propagate()
    scored: list[tuple[float, float, float, Action, dict[str, Any]]] = []
    counterfactuals: dict[str, dict[str, Any]] = {}

    for action in actions:
        predicted = graph.counterfactual(dict(action.effects)) if action.effects else baseline_state
        counterfactuals[action.name] = predicted
        prag = pragmatic_value(predicted, goal)
        epi = epistemic_value(baseline_state, predicted)
        efe = -(prag + epi)
        scored.append((efe, prag, epi, action, predicted))

    # Stable ordering: lowest EFE, ties broken by original action order.
    best_index = min(range(len(scored)), key=lambda i: (scored[i][0], i))
    efe, prag, epi, action, predicted = scored[best_index]

    # Build the causal chain for the primary goal variable (first key in goal).
    target_var = next(iter(goal), None)
    if target_var is not None and target_var in graph.variables():
        chain = graph.explain_path(target_var)
    else:
        chain = graph.all_edges()

    return Plan(
        action=action,
        efe=efe,
        pragmatic=prag,
        epistemic=epi,
        causal_chain=chain,
        counterfactuals=counterfactuals,
        chosen_state=predicted,
    )
