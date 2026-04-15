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


def containment(world: WorldState) -> list[CausalEdge]:
    """Contents of an inverted, unsealed container escape."""
    edges: list[CausalEdge] = []
    for entity in world.entities.values():
        if not entity.has("contains"):
            continue
        orientation = entity.get("orientation")
        sealed = entity.get("sealed")
        if orientation != "inverted":
            continue
        if sealed is True:
            continue
        # Material that absorbs its own contents breaks the chain too.
        if entity.get("material") == "absorbent":
            continue
        escape_var = f"{entity.id}.contents_escape"
        edges.append(
            CausalEdge(
                parents=(),
                effect=escape_var,
                mechanism=lambda _: True,
                label=f"containment({entity.id})",
            )
        )
    return edges


def impact(world: WorldState) -> list[CausalEdge]:
    """A fragile object impacting above its threshold breaks."""
    edges: list[CausalEdge] = []
    for r in world.relations_of("WILL_HIT"):
        obj = world.entities.get(r.src)
        if obj is None:
            continue
        if not obj.get("fragile"):
            continue
        threshold = obj.get("impact_threshold")
        velocity = r.attributes.get("velocity")
        if not isinstance(threshold, (int, float)) or not isinstance(velocity, (int, float)):
            continue
        if velocity >= threshold:
            break_var = f"{obj.id}.breaks"
            edges.append(
                CausalEdge(
                    parents=(),
                    effect=break_var,
                    mechanism=lambda _: True,
                    label=f"impact({obj.id})",
                )
            )
    return edges
