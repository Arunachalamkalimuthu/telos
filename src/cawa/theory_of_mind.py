"""Theory of Mind: other agents have their own beliefs and perceptual capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .world import WorldState


@dataclass(frozen=True)
class AgentMind:
    id: str
    beliefs: WorldState
    goals: tuple[Mapping[str, Any], ...] = ()
    capabilities: frozenset[str] = frozenset()
    actions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "goals", tuple(self.goals))
        object.__setattr__(self, "actions", tuple(self.actions))


def predict_action(mind: AgentMind, _ground_truth: WorldState) -> str:
    """Predict the action the agent will take.

    The second argument is ground truth — but we deliberately ignore it
    and reason from `mind.beliefs` only. This is what distinguishes ToM
    from behaviourist prediction: the agent acts on their own model of
    the world, not the true world.
    """
    if not mind.actions:
        return ""
    if not mind.goals:
        return mind.actions[0]
    # Pick the action whose name most closely matches the primary goal's intent.
    goal = mind.goals[0]
    goal_intent = f"{goal.get('type', '')}_{goal.get('target', '')}".strip("_")
    for action in mind.actions:
        if goal_intent and goal_intent in action:
            return action
    return mind.actions[0]


@dataclass(frozen=True)
class Intervention:
    kind: str  # "verbal", "visual", "physical", "tactile"
    content: str = ""


# Mapping from intervention kind to the capability it requires on the target.
_CHANNEL_REQUIREMENT: dict[str, str] = {
    "verbal": "auditory",
    "visual": "visual",
    "tactile": "tactile",
}


def intervention_effect(intervention: Intervention, target: AgentMind) -> bool:
    """Return True iff `intervention` can reach `target` given their capabilities.

    Physical interventions bypass sensory channels (you can intercept a deaf
    child by running in front of them). Sensory channels require the matching
    capability on the target.
    """
    if intervention.kind == "physical":
        return True
    required = _CHANNEL_REQUIREMENT.get(intervention.kind)
    if required is None:
        return False
    return required in target.capabilities
