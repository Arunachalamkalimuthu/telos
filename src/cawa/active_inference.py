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
