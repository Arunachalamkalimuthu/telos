"""CAWA — Causal Active World Architecture (reference implementation)."""

from .active_inference import Action, Plan, select_action
from .agent import CAWAAgent
from .causal_graph import CausalEdge, CausalGraph
from .physics import ALL_PRIMITIVES, apply_all, containment, gravity, impact, liquid_damage
from .theory_of_mind import AgentMind, Intervention, intervention_effect, predict_action
from .world import UNKNOWN, Entity, Relation, WorldState

__all__ = [
    "UNKNOWN",
    "Entity",
    "Relation",
    "WorldState",
    "CausalEdge",
    "CausalGraph",
    "ALL_PRIMITIVES",
    "apply_all",
    "gravity",
    "containment",
    "impact",
    "liquid_damage",
    "AgentMind",
    "Intervention",
    "predict_action",
    "intervention_effect",
    "Action",
    "Plan",
    "select_action",
    "CAWAAgent",
]
