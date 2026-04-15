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
