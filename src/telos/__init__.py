"""CAWA — Causal Active World Architecture (reference implementation)."""

from .active_inference import Action, Plan, select_action
from .agent import CAWAAgent
from .causal_graph import CausalEdge, CausalGraph
from .physics import ALL_PRIMITIVES, apply_all, containment, gravity, impact, liquid_damage
from .theory_of_mind import AgentMind, Intervention, intervention_effect, predict_action
from .world import UNKNOWN, Entity, Relation, WorldState
from .structure_learner import compare_graphs, generate_samples, has_latent_edges, learn_graph
from .nlu import load_model, parse_query, parse_scene
from .perception import (
    build_world, detect_objects, estimate_depth, extract_relations,
    get_physics_properties, process_video, track_objects, PROPERTY_KB,
)

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
    "generate_samples",
    "learn_graph",
    "compare_graphs",
    "has_latent_edges",
    "detect_objects",
    "extract_relations",
    "estimate_depth",
    "get_physics_properties",
    "track_objects",
    "process_video",
    "PROPERTY_KB",
    "build_world",
    "parse_scene",
    "parse_query",
    "load_model",
]
