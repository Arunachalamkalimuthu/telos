"""World state primitives: entities, relations, and immutable state snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


class _Unknown:
    """Sentinel for properties that are not known. Never compare with ==; use `is UNKNOWN`."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNKNOWN"

    def __bool__(self) -> bool:
        return False


UNKNOWN = _Unknown()


@dataclass(frozen=True)
class Entity:
    id: str
    type: str
    properties: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Freeze properties dict so the entity is truly immutable.
        object.__setattr__(self, "properties", MappingProxyType(dict(self.properties)))

    def __hash__(self) -> int:
        return hash((self.id, self.type))

    def get(self, key: str, default: Any = UNKNOWN) -> Any:
        return self.properties.get(key, default)

    def has(self, key: str) -> bool:
        return key in self.properties


@dataclass(frozen=True)
class Relation:
    name: str
    src: str
    dst: str
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))


@dataclass(frozen=True)
class WorldState:
    entities: Mapping[str, Entity] = field(default_factory=dict)
    relations: tuple[Relation, ...] = ()
    time: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "entities", MappingProxyType(dict(self.entities)))
        object.__setattr__(self, "relations", tuple(self.relations))

    def get_entity(self, entity_id: str) -> Entity:
        return self.entities[entity_id]

    def with_entity(self, entity: Entity) -> "WorldState":
        new_entities = dict(self.entities)
        new_entities[entity.id] = entity
        return WorldState(entities=new_entities, relations=self.relations, time=self.time)

    def with_relation(self, relation: Relation) -> "WorldState":
        return WorldState(
            entities=dict(self.entities),
            relations=self.relations + (relation,),
            time=self.time,
        )

    def relations_of(self, name: str) -> list[Relation]:
        return [r for r in self.relations if r.name == name]

    def relations_for(self, entity_id: str) -> list[Relation]:
        return [r for r in self.relations if r.src == entity_id or r.dst == entity_id]
