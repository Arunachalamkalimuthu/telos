"""Physics primitives as axiomatic pure functions over WorldState.

Each primitive returns a list of CausalEdges that should be added to the
causal graph for the scene. Primitives are composable: apply_all takes the
union of edges from every applicable primitive.
"""

from __future__ import annotations

from typing import Callable

from .causal_graph import CausalEdge
from .world import Entity, Relation, WorldState


def _is_supported(entity: Entity, world: WorldState) -> bool:
    """An entity is supported iff it has an ON or HELD_BY relation as src."""
    for r in world.relations_for(entity.id):
        if r.src == entity.id and r.name in ("ON", "HELD_BY", "ATTACHED_TO"):
            return True
    return False


def gravity(world: WorldState) -> list[CausalEdge]:
    """Unsupported massed entities fall."""
    edges: list[CausalEdge] = []
    for entity in world.entities.values():
        mass = entity.get("mass")
        if mass is None or mass is False:
            continue
        if not isinstance(mass, (int, float)):
            continue
        supported = _is_supported(entity, world)
        if supported:
            continue
        fall_var = f"{entity.id}.falls"
        edges.append(
            CausalEdge(
                parents=(),
                effect=fall_var,
                mechanism=lambda _: True,
                label=f"gravity({entity.id})",
            )
        )
    return edges
