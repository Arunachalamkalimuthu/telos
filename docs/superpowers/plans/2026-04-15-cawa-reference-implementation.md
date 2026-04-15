# CAWA Reference Implementation — Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable Python reference implementation of CAWA (Causal Active World Architecture) demonstrating the architectural principles from the Medium article on four closed-domain scenarios.

**Architecture:** Six-module Python package — `world` (state), `physics` (axiomatic primitives), `causal_graph` (DAG + do-calculus), `theory_of_mind` (agent minds), `active_inference` (EFE planner), `agent` (orchestration). Four example scripts. Pure symbolic, no runtime dependencies outside the standard library.

**Tech Stack:** Python 3.10+ standard library only. `unittest` for tests. `make` for build.

**Spec:** `docs/superpowers/specs/2026-04-15-cawa-reference-implementation-design.md`

**Working directory:** `/Users/arunachalamk/Code/cawa`

---

## File map

**New files (in dependency order):**

- `pyproject.toml` — package metadata, Python 3.10+ requirement, package layout
- `.gitignore` — Python ignores
- `Makefile` — `test`, `demo`, `clean` targets
- `README.md` — what this is, quickstart, what it does not claim
- `src/cawa/__init__.py` — public surface
- `src/cawa/world.py` — `UNKNOWN`, `Entity`, `Relation`, `WorldState`
- `src/cawa/causal_graph.py` — `CausalEdge`, `CausalGraph` with propagate / do / counterfactual / explain_path
- `src/cawa/physics.py` — primitives (`gravity`, `containment`, `impact`, `support`) + `apply_all`
- `src/cawa/theory_of_mind.py` — `AgentMind`, `predict_action`, `intervention_effect`
- `src/cawa/active_inference.py` — `Action`, `Plan`, `pragmatic_value`, `epistemic_value`, `select_action`
- `src/cawa/agent.py` — `CAWAAgent`
- `examples/__init__.py`
- `examples/coffee_cup.py`
- `examples/child_road.py`
- `examples/salt_request.py`
- `examples/novel_entity.py`
- `tests/__init__.py`
- `tests/test_world.py`
- `tests/test_causal_graph.py`
- `tests/test_physics.py`
- `tests/test_theory_of_mind.py`
- `tests/test_active_inference.py`
- `tests/test_examples.py`
- `docs/architecture.md` — maps article claims to code

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `Makefile`, `README.md`
- Create: `src/cawa/__init__.py`, `examples/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "cawa"
version = "0.1.0"
description = "Causal Active World Architecture — reference implementation"
requires-python = ">=3.10"
dependencies = []

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-dir]
"" = "src"
```

- [ ] **Step 2: Create `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.venv/
.DS_Store
build/
dist/
```

- [ ] **Step 3: Create `Makefile`**

```makefile
.PHONY: test demo clean

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

demo:
	@PYTHONPATH=src python -m examples.coffee_cup
	@echo
	@PYTHONPATH=src python -m examples.child_road
	@echo
	@PYTHONPATH=src python -m examples.salt_request
	@echo
	@PYTHONPATH=src python -m examples.novel_entity

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
```

- [ ] **Step 4: Create `README.md`**

```markdown
# CAWA — Causal Active World Architecture

A Python reference implementation of the architecture described in the Medium article *Beyond Token Prediction: How LLMs, JEPA, and CAWA See the World Differently*.

**This is not AGI.** This is a closed-domain proof that causal graphs, physics primitives, theory of mind, and active inference compose cleanly into a working agent. See `docs/architecture.md` for an honest accounting of what this does and does not demonstrate.

## Quickstart

```bash
make test      # run all tests
make demo      # run all four example scenarios
```

## Examples

- `examples/coffee_cup.py` — physics + counterfactuals
- `examples/child_road.py` — theory of mind + intervention planning
- `examples/salt_request.py` — theory of mind + social inference
- `examples/novel_entity.py` — honest uncertainty with unknown entities

## Requirements

Python 3.10+. No runtime dependencies.
```

- [ ] **Step 5: Create empty package files**

```bash
mkdir -p src/cawa examples tests
touch src/cawa/__init__.py examples/__init__.py tests/__init__.py
```

- [ ] **Step 6: Verify scaffolding**

Run: `PYTHONPATH=src python -c "import cawa; print('ok')"`
Expected: `ok`

Run: `make test`
Expected: `Ran 0 tests in ... OK` (no tests yet, but discover works)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore Makefile README.md src/ examples/ tests/
git commit -m "scaffold: project skeleton and empty packages"
```

---

## Task 2: World module — UNKNOWN sentinel and Entity

**Files:**
- Create: `tests/test_world.py`
- Create: `src/cawa/world.py`

- [ ] **Step 1: Write failing test for `UNKNOWN` and `Entity`**

Create `tests/test_world.py`:

```python
import unittest
from cawa.world import UNKNOWN, Entity


class TestEntity(unittest.TestCase):
    def test_entity_has_id_type_properties(self):
        e = Entity(id="cup_1", type="cup", properties={"mass": 0.2, "sealed": False})
        self.assertEqual(e.id, "cup_1")
        self.assertEqual(e.type, "cup")
        self.assertEqual(e.get("mass"), 0.2)
        self.assertFalse(e.get("sealed"))

    def test_entity_unknown_property_returns_UNKNOWN(self):
        e = Entity(id="x", type="frambulator", properties={})
        self.assertIs(e.get("glorbic_index"), UNKNOWN)

    def test_entity_is_hashable_and_frozen(self):
        e = Entity(id="a", type="t", properties={"k": 1})
        self.assertEqual(hash(e), hash(e))
        with self.assertRaises(Exception):
            e.id = "changed"


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests.test_world -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cawa.world'`

- [ ] **Step 3: Implement `UNKNOWN` and `Entity`**

Create `src/cawa/world.py`:

```python
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

    def get(self, key: str, default: Any = UNKNOWN) -> Any:
        return self.properties.get(key, default)

    def has(self, key: str) -> bool:
        return key in self.properties
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m unittest tests.test_world -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cawa/world.py tests/test_world.py
git commit -m "world: add UNKNOWN sentinel and Entity"
```

---

## Task 3: World module — Relation and WorldState

**Files:**
- Modify: `tests/test_world.py`
- Modify: `src/cawa/world.py`

- [ ] **Step 1: Append failing tests for `Relation` and `WorldState`**

Append to `tests/test_world.py` (before `if __name__`):

```python
from cawa.world import Relation, WorldState


class TestRelation(unittest.TestCase):
    def test_relation_has_name_src_dst(self):
        r = Relation(name="ON", src="cup_1", dst="table_1")
        self.assertEqual(r.name, "ON")
        self.assertEqual(r.src, "cup_1")
        self.assertEqual(r.dst, "table_1")


class TestWorldState(unittest.TestCase):
    def test_worldstate_construction(self):
        cup = Entity(id="cup_1", type="cup", properties={"mass": 0.2})
        ws = WorldState(entities={"cup_1": cup}, relations=())
        self.assertIs(ws.get_entity("cup_1"), cup)

    def test_with_entity_returns_new_state(self):
        ws1 = WorldState(entities={}, relations=())
        cup = Entity(id="cup_1", type="cup", properties={})
        ws2 = ws1.with_entity(cup)
        self.assertNotIn("cup_1", ws1.entities)
        self.assertIn("cup_1", ws2.entities)
        self.assertIsNot(ws1, ws2)

    def test_with_relation_returns_new_state(self):
        ws1 = WorldState(entities={}, relations=())
        r = Relation(name="ON", src="a", dst="b")
        ws2 = ws1.with_relation(r)
        self.assertEqual(len(ws1.relations), 0)
        self.assertEqual(len(ws2.relations), 1)
        self.assertEqual(ws2.relations[0], r)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m unittest tests.test_world -v`
Expected: FAIL with `ImportError: cannot import name 'Relation'`.

- [ ] **Step 3: Implement `Relation` and `WorldState`**

Append to `src/cawa/world.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m unittest tests.test_world -v`
Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cawa/world.py tests/test_world.py
git commit -m "world: add Relation and immutable WorldState"
```

---

## Task 4: CausalGraph — variable and mechanism registration

**Files:**
- Create: `tests/test_causal_graph.py`
- Create: `src/cawa/causal_graph.py`

- [ ] **Step 1: Write failing test for graph construction**

Create `tests/test_causal_graph.py`:

```python
import unittest
from cawa.causal_graph import CausalGraph, CausalEdge


class TestCausalGraphConstruction(unittest.TestCase):
    def test_add_variable_and_get_value(self):
        g = CausalGraph()
        g.add_variable("a", initial=1)
        self.assertEqual(g.get("a"), 1)

    def test_add_mechanism_links_parents_to_effect(self):
        g = CausalGraph()
        g.add_variable("a", initial=2)
        g.add_variable("b")
        g.add_mechanism("b", parents=["a"], mechanism=lambda p: p["a"] * 2, label="double")
        edges = g.edges_into("b")
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].parents, ("a",))
        self.assertEqual(edges[0].effect, "b")
        self.assertEqual(edges[0].label, "double")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests.test_causal_graph -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cawa.causal_graph'`.

- [ ] **Step 3: Implement graph skeleton**

Create `src/cawa/causal_graph.py`:

```python
"""Causal DAG with do-calculus. Hand-built per scene, not learned."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


Mechanism = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True)
class CausalEdge:
    parents: tuple[str, ...]
    effect: str
    mechanism: Mechanism
    label: str = ""


class CausalGraph:
    """Directed acyclic graph of causal mechanisms over named variables."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}
        self._edges: list[CausalEdge] = []

    def add_variable(self, name: str, initial: Any = None) -> None:
        if name in self._values:
            raise ValueError(f"variable {name!r} already declared")
        self._values[name] = initial

    def add_mechanism(
        self,
        effect: str,
        parents: list[str],
        mechanism: Mechanism,
        label: str = "",
    ) -> None:
        if effect not in self._values:
            raise ValueError(f"effect variable {effect!r} not declared")
        for p in parents:
            if p not in self._values:
                raise ValueError(f"parent variable {p!r} not declared")
        self._edges.append(CausalEdge(tuple(parents), effect, mechanism, label))

    def get(self, name: str) -> Any:
        return self._values[name]

    def variables(self) -> list[str]:
        return list(self._values.keys())

    def edges_into(self, effect: str) -> list[CausalEdge]:
        return [e for e in self._edges if e.effect == effect]

    def all_edges(self) -> list[CausalEdge]:
        return list(self._edges)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m unittest tests.test_causal_graph -v`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cawa/causal_graph.py tests/test_causal_graph.py
git commit -m "causal_graph: add variable and mechanism registration"
```

---

## Task 5: CausalGraph — propagate

**Files:**
- Modify: `tests/test_causal_graph.py`
- Modify: `src/cawa/causal_graph.py`

- [ ] **Step 1: Append failing tests for propagate**

Append to `tests/test_causal_graph.py` (before `if __name__`):

```python
class TestCausalGraphPropagate(unittest.TestCase):
    def test_propagate_simple_chain(self):
        g = CausalGraph()
        g.add_variable("a", initial=3)
        g.add_variable("b")
        g.add_variable("c")
        g.add_mechanism("b", ["a"], lambda p: p["a"] + 1)
        g.add_mechanism("c", ["b"], lambda p: p["b"] * 10)
        state = g.propagate()
        self.assertEqual(state["a"], 3)
        self.assertEqual(state["b"], 4)
        self.assertEqual(state["c"], 40)

    def test_propagate_multiple_parents(self):
        g = CausalGraph()
        g.add_variable("x", initial=2)
        g.add_variable("y", initial=5)
        g.add_variable("z")
        g.add_mechanism("z", ["x", "y"], lambda p: p["x"] + p["y"])
        self.assertEqual(g.propagate()["z"], 7)

    def test_propagate_raises_on_cycle(self):
        g = CausalGraph()
        g.add_variable("a")
        g.add_variable("b")
        g.add_mechanism("a", ["b"], lambda p: p["b"])
        g.add_mechanism("b", ["a"], lambda p: p["a"])
        with self.assertRaises(ValueError):
            g.propagate()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m unittest tests.test_causal_graph -v`
Expected: FAIL with `AttributeError: 'CausalGraph' object has no attribute 'propagate'`.

- [ ] **Step 3: Implement propagate with topological sort**

Append to `src/cawa/causal_graph.py`:

```python
    def _topological_order(self) -> list[str]:
        # Kahn's algorithm.
        indegree: dict[str, int] = {v: 0 for v in self._values}
        children: dict[str, list[str]] = {v: [] for v in self._values}
        for edge in self._edges:
            indegree[edge.effect] += len(edge.parents)
            for p in edge.parents:
                children[p].append(edge.effect)

        queue = [v for v, d in indegree.items() if d == 0]
        order: list[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in children[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(order) != len(self._values):
            raise ValueError("causal graph contains a cycle")
        return order

    def propagate(self) -> dict[str, Any]:
        order = self._topological_order()
        state = dict(self._values)
        for var in order:
            incoming = self.edges_into(var)
            if not incoming:
                continue
            # Variables with multiple incoming edges: compose by the last edge
            # defined (caller is expected to declare a single mechanism per var;
            # additional edges represent joint causes inside one mechanism).
            edge = incoming[-1]
            parent_values = {p: state[p] for p in edge.parents}
            state[var] = edge.mechanism(parent_values)
        return state
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m unittest tests.test_causal_graph -v`
Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cawa/causal_graph.py tests/test_causal_graph.py
git commit -m "causal_graph: add topological propagation"
```

---

## Task 6: CausalGraph — do-operator and counterfactuals

**Files:**
- Modify: `tests/test_causal_graph.py`
- Modify: `src/cawa/causal_graph.py`

- [ ] **Step 1: Append failing tests for `do` and `counterfactual`**

Append to `tests/test_causal_graph.py`:

```python
class TestCausalGraphIntervention(unittest.TestCase):
    def test_do_pins_variable_and_severs_incoming(self):
        g = CausalGraph()
        g.add_variable("a", initial=3)
        g.add_variable("b")
        g.add_mechanism("b", ["a"], lambda p: p["a"] + 1)
        g2 = g.do("b", 100)
        # Original graph untouched.
        self.assertEqual(g.propagate()["b"], 4)
        # Intervened graph: b is pinned regardless of a.
        self.assertEqual(g2.propagate()["b"], 100)
        # And b has no incoming edges anymore.
        self.assertEqual(g2.edges_into("b"), [])

    def test_counterfactual_multiple_interventions(self):
        g = CausalGraph()
        g.add_variable("gravity", initial=True)
        g.add_variable("sealed", initial=False)
        g.add_variable("spill")
        g.add_mechanism(
            "spill",
            ["gravity", "sealed"],
            lambda p: p["gravity"] and not p["sealed"],
        )
        self.assertTrue(g.propagate()["spill"])
        # Counterfactual 1: sealed=True → no spill.
        self.assertFalse(g.counterfactual({"sealed": True})["spill"])
        # Counterfactual 2: gravity=False → no spill.
        self.assertFalse(g.counterfactual({"gravity": False})["spill"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m unittest tests.test_causal_graph -v`
Expected: FAIL with `AttributeError: 'CausalGraph' object has no attribute 'do'`.

- [ ] **Step 3: Implement `do` and `counterfactual`**

Append to `src/cawa/causal_graph.py`:

```python
    def do(self, var: str, value: Any) -> "CausalGraph":
        """Return a new graph with `var` pinned to `value` and its incoming edges severed."""
        if var not in self._values:
            raise ValueError(f"unknown variable {var!r}")
        new = CausalGraph()
        new._values = dict(self._values)
        new._values[var] = value
        new._edges = [e for e in self._edges if e.effect != var]
        return new

    def counterfactual(self, interventions: Mapping[str, Any]) -> dict[str, Any]:
        """Apply a set of do-interventions and return the resulting state."""
        g = self
        for var, value in interventions.items():
            g = g.do(var, value)
        return g.propagate()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m unittest tests.test_causal_graph -v`
Expected: all 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cawa/causal_graph.py tests/test_causal_graph.py
git commit -m "causal_graph: add do-operator and counterfactuals"
```

---

## Task 7: CausalGraph — explain_path

**Files:**
- Modify: `tests/test_causal_graph.py`
- Modify: `src/cawa/causal_graph.py`

- [ ] **Step 1: Append failing test for `explain_path`**

Append to `tests/test_causal_graph.py`:

```python
class TestCausalGraphExplain(unittest.TestCase):
    def test_explain_path_returns_chain_of_edges(self):
        g = CausalGraph()
        g.add_variable("orientation", initial="inverted")
        g.add_variable("sealed", initial=False)
        g.add_variable("contents_escape")
        g.add_variable("liquid_falls")
        g.add_variable("laptop_damaged")
        g.add_mechanism(
            "contents_escape",
            ["orientation", "sealed"],
            lambda p: p["orientation"] == "inverted" and not p["sealed"],
            label="containment_breach",
        )
        g.add_mechanism(
            "liquid_falls",
            ["contents_escape"],
            lambda p: p["contents_escape"],
            label="gravity",
        )
        g.add_mechanism(
            "laptop_damaged",
            ["liquid_falls"],
            lambda p: p["liquid_falls"],
            label="impact",
        )
        chain = g.explain_path("laptop_damaged")
        labels = [e.label for e in chain]
        self.assertEqual(labels, ["containment_breach", "gravity", "impact"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests.test_causal_graph -v`
Expected: FAIL with `AttributeError: 'CausalGraph' object has no attribute 'explain_path'`.

- [ ] **Step 3: Implement `explain_path`**

Append to `src/cawa/causal_graph.py`:

```python
    def explain_path(self, target: str) -> list[CausalEdge]:
        """Return the causal chain from root causes to `target` in topological order."""
        if target not in self._values:
            raise ValueError(f"unknown variable {target!r}")
        # Collect all ancestors of target.
        relevant: set[str] = set()
        stack = [target]
        while stack:
            node = stack.pop()
            if node in relevant:
                continue
            relevant.add(node)
            for edge in self.edges_into(node):
                for p in edge.parents:
                    stack.append(p)
        # Return edges among ancestors, in topological order.
        order = self._topological_order()
        edges_in_order: list[CausalEdge] = []
        for var in order:
            if var in relevant:
                for edge in self.edges_into(var):
                    edges_in_order.append(edge)
        return edges_in_order
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m unittest tests.test_causal_graph -v`
Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cawa/causal_graph.py tests/test_causal_graph.py
git commit -m "causal_graph: add explain_path for causal chain extraction"
```

---

## Task 8: Physics — gravity primitive

**Files:**
- Create: `tests/test_physics.py`
- Create: `src/cawa/physics.py`

- [ ] **Step 1: Write failing test for `gravity`**

Create `tests/test_physics.py`:

```python
import unittest
from cawa.world import Entity, Relation, WorldState
from cawa.physics import gravity


class TestGravity(unittest.TestCase):
    def test_gravity_emits_fall_edge_for_unsupported_massed_object(self):
        # A cup with mass and no supporting relation → falls.
        cup = Entity(id="cup", type="cup", properties={"mass": 0.2})
        ws = WorldState(entities={"cup": cup}, relations=())
        edges = gravity(ws)
        effects = [e.effect for e in edges]
        self.assertIn("cup.falls", effects)

    def test_gravity_does_not_emit_for_supported_object(self):
        cup = Entity(id="cup", type="cup", properties={"mass": 0.2})
        table = Entity(id="table", type="table", properties={})
        ws = WorldState(
            entities={"cup": cup, "table": table},
            relations=(Relation("ON", "cup", "table"),),
        )
        edges = gravity(ws)
        effects = [e.effect for e in edges]
        self.assertNotIn("cup.falls", effects)

    def test_gravity_ignores_massless_entity(self):
        ghost = Entity(id="ghost", type="concept", properties={})
        ws = WorldState(entities={"ghost": ghost}, relations=())
        self.assertEqual(gravity(ws), [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests.test_physics -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cawa.physics'`.

- [ ] **Step 3: Implement `gravity`**

Create `src/cawa/physics.py`:

```python
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
        # Skip entities without a numeric mass (e.g. UNKNOWN or missing).
        if not isinstance(mass, (int, float)):
            continue
        supported = _is_supported(entity, world)
        fall_var = f"{entity.id}.falls"
        edges.append(
            CausalEdge(
                parents=(),
                effect=fall_var,
                mechanism=lambda _, supported=supported: not supported,
                label=f"gravity({entity.id})",
            )
        )
    return edges
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m unittest tests.test_physics -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cawa/physics.py tests/test_physics.py
git commit -m "physics: add gravity primitive"
```

---

## Task 9: Physics — containment primitive

**Files:**
- Modify: `tests/test_physics.py`
- Modify: `src/cawa/physics.py`

- [ ] **Step 1: Append failing tests for `containment`**

Append to `tests/test_physics.py`:

```python
from cawa.physics import containment


class TestContainment(unittest.TestCase):
    def test_inverted_unsealed_container_with_contents_emits_escape(self):
        cup = Entity(
            id="cup",
            type="cup",
            properties={"orientation": "inverted", "sealed": False, "contains": "coffee"},
        )
        ws = WorldState(entities={"cup": cup}, relations=())
        edges = containment(ws)
        effects = [e.effect for e in edges]
        self.assertIn("cup.contents_escape", effects)

    def test_sealed_container_does_not_emit_escape(self):
        cup = Entity(
            id="cup",
            type="cup",
            properties={"orientation": "inverted", "sealed": True, "contains": "coffee"},
        )
        ws = WorldState(entities={"cup": cup}, relations=())
        edges = containment(ws)
        effects = [e.effect for e in edges]
        self.assertNotIn("cup.contents_escape", effects)

    def test_upright_container_does_not_emit_escape(self):
        cup = Entity(
            id="cup",
            type="cup",
            properties={"orientation": "upright", "sealed": False, "contains": "coffee"},
        )
        ws = WorldState(entities={"cup": cup}, relations=())
        self.assertEqual(containment(ws), [])

    def test_empty_container_does_not_emit_escape(self):
        cup = Entity(
            id="cup",
            type="cup",
            properties={"orientation": "inverted", "sealed": False},
        )
        ws = WorldState(entities={"cup": cup}, relations=())
        self.assertEqual(containment(ws), [])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m unittest tests.test_physics -v`
Expected: FAIL with `ImportError: cannot import name 'containment'`.

- [ ] **Step 3: Implement `containment`**

Append to `src/cawa/physics.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m unittest tests.test_physics -v`
Expected: 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cawa/physics.py tests/test_physics.py
git commit -m "physics: add containment primitive"
```

---

## Task 10: Physics — impact primitive

**Files:**
- Modify: `tests/test_physics.py`
- Modify: `src/cawa/physics.py`

- [ ] **Step 1: Append failing tests for `impact`**

Append to `tests/test_physics.py`:

```python
from cawa.physics import impact


class TestImpact(unittest.TestCase):
    def test_fragile_object_falling_onto_hard_surface_emits_break(self):
        glass = Entity(
            id="glass",
            type="glass",
            properties={"mass": 0.3, "fragile": True, "impact_threshold": 1.0},
        )
        floor = Entity(id="floor", type="floor", properties={"hardness": "hard"})
        ws = WorldState(
            entities={"glass": glass, "floor": floor},
            relations=(Relation("WILL_HIT", "glass", "floor", attributes={"velocity": 3.0}),),
        )
        edges = impact(ws)
        effects = [e.effect for e in edges]
        self.assertIn("glass.breaks", effects)

    def test_non_fragile_object_does_not_break(self):
        ball = Entity(
            id="ball",
            type="ball",
            properties={"mass": 0.1, "fragile": False, "impact_threshold": 1.0},
        )
        floor = Entity(id="floor", type="floor", properties={"hardness": "hard"})
        ws = WorldState(
            entities={"ball": ball, "floor": floor},
            relations=(Relation("WILL_HIT", "ball", "floor", attributes={"velocity": 10.0}),),
        )
        edges = impact(ws)
        self.assertEqual(edges, [])

    def test_low_velocity_impact_does_not_break(self):
        glass = Entity(
            id="glass",
            type="glass",
            properties={"mass": 0.3, "fragile": True, "impact_threshold": 5.0},
        )
        floor = Entity(id="floor", type="floor", properties={"hardness": "hard"})
        ws = WorldState(
            entities={"glass": glass, "floor": floor},
            relations=(Relation("WILL_HIT", "glass", "floor", attributes={"velocity": 1.0}),),
        )
        edges = impact(ws)
        self.assertEqual(edges, [])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m unittest tests.test_physics -v`
Expected: FAIL with `ImportError: cannot import name 'impact'`.

- [ ] **Step 3: Implement `impact`**

Append to `src/cawa/physics.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m unittest tests.test_physics -v`
Expected: 10 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cawa/physics.py tests/test_physics.py
git commit -m "physics: add impact primitive"
```

---

## Task 11: Physics — liquid_damage primitive (context for coffee-cup scenario)

**Files:**
- Modify: `tests/test_physics.py`
- Modify: `src/cawa/physics.py`

- [ ] **Step 1: Append failing tests for liquid contacting electronics**

Append to `tests/test_physics.py`:

```python
from cawa.physics import liquid_damage


class TestLiquidDamage(unittest.TestCase):
    def test_liquid_on_electronics_emits_damage(self):
        coffee = Entity(id="coffee", type="liquid", properties={"conductive": True})
        laptop = Entity(id="laptop", type="laptop", properties={"electronic": True})
        ws = WorldState(
            entities={"coffee": coffee, "laptop": laptop},
            relations=(Relation("WILL_CONTACT", "coffee", "laptop"),),
        )
        edges = liquid_damage(ws)
        effects = [e.effect for e in edges]
        self.assertIn("laptop.damaged", effects)

    def test_non_conductive_liquid_does_not_damage(self):
        oil = Entity(id="oil", type="liquid", properties={"conductive": False})
        laptop = Entity(id="laptop", type="laptop", properties={"electronic": True})
        ws = WorldState(
            entities={"oil": oil, "laptop": laptop},
            relations=(Relation("WILL_CONTACT", "oil", "laptop"),),
        )
        self.assertEqual(liquid_damage(ws), [])

    def test_contained_liquid_links_damage_to_contents_escape(self):
        """If the liquid has a container, damage depends on the container's contents escaping."""
        coffee = Entity(id="coffee", type="liquid", properties={"conductive": True})
        cup = Entity(id="cup", type="cup", properties={"contains": "coffee"})
        laptop = Entity(id="laptop", type="laptop", properties={"electronic": True})
        ws = WorldState(
            entities={"cup": cup, "coffee": coffee, "laptop": laptop},
            relations=(Relation("WILL_CONTACT", "coffee", "laptop"),),
        )
        edges = liquid_damage(ws)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].parents, ("cup.contents_escape",))
        self.assertEqual(edges[0].effect, "laptop.damaged")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m unittest tests.test_physics -v`
Expected: FAIL with `ImportError: cannot import name 'liquid_damage'`.

- [ ] **Step 3: Implement `liquid_damage`**

Append to `src/cawa/physics.py`:

```python
def liquid_damage(world: WorldState) -> list[CausalEdge]:
    """Conductive liquid contacting electronics damages them.

    If the liquid is inside a container (some entity's `contains` property
    equals the liquid's id), damage is conditional on that container's
    contents escaping — this builds the causal chain
    `cup.contents_escape → laptop.damaged`. If the liquid has no container,
    damage is a root effect.
    """
    edges: list[CausalEdge] = []
    for r in world.relations_of("WILL_CONTACT"):
        src = world.entities.get(r.src)
        dst = world.entities.get(r.dst)
        if src is None or dst is None:
            continue
        if src.type != "liquid":
            continue
        if not src.get("conductive"):
            continue
        if not dst.get("electronic"):
            continue
        container = None
        for other in world.entities.values():
            if other.get("contains") == src.id:
                container = other
                break
        damage_var = f"{dst.id}.damaged"
        if container is not None:
            escape_var = f"{container.id}.contents_escape"
            edges.append(
                CausalEdge(
                    parents=(escape_var,),
                    effect=damage_var,
                    mechanism=lambda p, ev=escape_var: bool(p[ev]),
                    label=f"liquid_damage({dst.id})",
                )
            )
        else:
            edges.append(
                CausalEdge(
                    parents=(),
                    effect=damage_var,
                    mechanism=lambda _: True,
                    label=f"liquid_damage({dst.id})",
                )
            )
    return edges
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m unittest tests.test_physics -v`
Expected: 13 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cawa/physics.py tests/test_physics.py
git commit -m "physics: add liquid_damage primitive"
```

---

## Task 12: Physics — apply_all composer

**Files:**
- Modify: `tests/test_physics.py`
- Modify: `src/cawa/physics.py`

- [ ] **Step 1: Append failing test for `apply_all`**

Append to `tests/test_physics.py`:

```python
from cawa.physics import apply_all, ALL_PRIMITIVES


class TestApplyAll(unittest.TestCase):
    def test_apply_all_unions_edges_from_all_primitives(self):
        # Inverted unsealed cup with mass and contents sitting on nothing.
        cup = Entity(
            id="cup",
            type="cup",
            properties={
                "mass": 0.2,
                "orientation": "inverted",
                "sealed": False,
                "contains": "coffee",
            },
        )
        ws = WorldState(entities={"cup": cup}, relations=())
        edges = apply_all(ws)
        labels = {e.label for e in edges}
        self.assertIn("gravity(cup)", labels)
        self.assertIn("containment(cup)", labels)

    def test_apply_all_accepts_custom_primitive_list(self):
        cup = Entity(id="cup", type="cup", properties={"mass": 0.2})
        ws = WorldState(entities={"cup": cup}, relations=())
        edges = apply_all(ws, primitives=[ALL_PRIMITIVES[0]])
        self.assertTrue(all("gravity" in e.label for e in edges))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m unittest tests.test_physics -v`
Expected: FAIL with `ImportError: cannot import name 'apply_all'`.

- [ ] **Step 3: Implement `apply_all` and `ALL_PRIMITIVES`**

Append to `src/cawa/physics.py`:

```python
ALL_PRIMITIVES: list[Callable[[WorldState], list[CausalEdge]]] = [
    gravity,
    containment,
    impact,
    liquid_damage,
]


def apply_all(
    world: WorldState,
    primitives: list[Callable[[WorldState], list[CausalEdge]]] | None = None,
) -> list[CausalEdge]:
    """Run every primitive and return the union of emitted causal edges."""
    primitives = primitives if primitives is not None else ALL_PRIMITIVES
    edges: list[CausalEdge] = []
    for prim in primitives:
        edges.extend(prim(world))
    return edges
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m unittest tests.test_physics -v`
Expected: 15 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cawa/physics.py tests/test_physics.py
git commit -m "physics: add apply_all primitive composer"
```

---

## Task 13: Theory of Mind — AgentMind and predict_action

**Files:**
- Create: `tests/test_theory_of_mind.py`
- Create: `src/cawa/theory_of_mind.py`

- [ ] **Step 1: Write failing tests for `AgentMind` and `predict_action`**

Create `tests/test_theory_of_mind.py`:

```python
import unittest
from cawa.world import Entity, WorldState
from cawa.theory_of_mind import AgentMind, predict_action


class TestAgentMind(unittest.TestCase):
    def test_agent_mind_stores_beliefs_goals_capabilities_actions(self):
        beliefs = WorldState(entities={}, relations=())
        m = AgentMind(
            id="child",
            beliefs=beliefs,
            goals=[{"type": "reach", "target": "parent"}],
            capabilities=frozenset({"visual"}),
            actions=["run", "stop", "wait"],
        )
        self.assertEqual(m.id, "child")
        self.assertIn("visual", m.capabilities)
        self.assertIn("run", m.actions)


class TestPredictAction(unittest.TestCase):
    def test_prediction_uses_agent_beliefs_not_ground_truth(self):
        # Ground truth: road is dangerous. Agent's belief: road is safe (it's a child).
        agent_belief_world = WorldState(
            entities={"road": Entity(id="road", type="road", properties={"danger": False})},
            relations=(),
        )
        mind = AgentMind(
            id="child",
            beliefs=agent_belief_world,
            goals=[{"type": "reach", "target": "parent"}],
            capabilities=frozenset({"visual"}),
            actions=["run_toward_parent", "stop"],
        )
        ground_truth = WorldState(
            entities={"road": Entity(id="road", type="road", properties={"danger": True})},
            relations=(),
        )
        action = predict_action(mind, ground_truth)
        self.assertEqual(action, "run_toward_parent")

    def test_prediction_defaults_to_first_action_when_no_goal_match(self):
        mind = AgentMind(
            id="x",
            beliefs=WorldState(entities={}, relations=()),
            goals=[],
            capabilities=frozenset(),
            actions=["wait"],
        )
        self.assertEqual(predict_action(mind, WorldState(entities={}, relations=())), "wait")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m unittest tests.test_theory_of_mind -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cawa.theory_of_mind'`.

- [ ] **Step 3: Implement `AgentMind` and `predict_action`**

Create `src/cawa/theory_of_mind.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m unittest tests.test_theory_of_mind -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cawa/theory_of_mind.py tests/test_theory_of_mind.py
git commit -m "theory_of_mind: add AgentMind and predict_action (uses beliefs not ground truth)"
```

---

## Task 14: Theory of Mind — intervention_effect respects capabilities

**Files:**
- Modify: `tests/test_theory_of_mind.py`
- Modify: `src/cawa/theory_of_mind.py`

- [ ] **Step 1: Append failing tests for `intervention_effect`**

Append to `tests/test_theory_of_mind.py`:

```python
from cawa.theory_of_mind import intervention_effect, Intervention


class TestInterventionEffect(unittest.TestCase):
    def _mind(self, capabilities):
        return AgentMind(
            id="child",
            beliefs=WorldState(entities={}, relations=()),
            goals=[],
            capabilities=frozenset(capabilities),
            actions=["run"],
        )

    def test_verbal_signal_reaches_hearing_agent(self):
        mind = self._mind({"visual", "auditory"})
        self.assertTrue(intervention_effect(Intervention(kind="verbal", content="stop"), mind))

    def test_verbal_signal_does_not_reach_deaf_agent(self):
        mind = self._mind({"visual"})
        self.assertFalse(intervention_effect(Intervention(kind="verbal", content="stop"), mind))

    def test_visual_signal_does_not_reach_blind_agent(self):
        mind = self._mind({"auditory"})
        self.assertFalse(intervention_effect(Intervention(kind="visual", content="wave"), mind))

    def test_physical_intervention_always_effective(self):
        mind = self._mind(set())  # even a sensory-deprived agent is physically interceptable.
        self.assertTrue(intervention_effect(Intervention(kind="physical", content="intercept"), mind))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m unittest tests.test_theory_of_mind -v`
Expected: FAIL with `ImportError: cannot import name 'intervention_effect'`.

- [ ] **Step 3: Implement `Intervention` and `intervention_effect`**

Append to `src/cawa/theory_of_mind.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m unittest tests.test_theory_of_mind -v`
Expected: 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cawa/theory_of_mind.py tests/test_theory_of_mind.py
git commit -m "theory_of_mind: add intervention_effect respecting perceptual capabilities"
```

---

## Task 15: Active Inference — Action, pragmatic and epistemic values

**Files:**
- Create: `tests/test_active_inference.py`
- Create: `src/cawa/active_inference.py`

- [ ] **Step 1: Write failing tests for `Action`, `pragmatic_value`, `epistemic_value`**

Create `tests/test_active_inference.py`:

```python
import unittest
from cawa.active_inference import Action, pragmatic_value, epistemic_value


class TestAction(unittest.TestCase):
    def test_action_stores_name_effects_description(self):
        a = Action(name="seal_cup", effects={"sealed": True}, description="put a lid on the cup")
        self.assertEqual(a.name, "seal_cup")
        self.assertEqual(a.effects["sealed"], True)


class TestPragmaticValue(unittest.TestCase):
    def test_pragmatic_value_is_zero_when_state_matches_goal(self):
        state = {"laptop_damaged": False, "cup_sealed": True}
        goal = {"laptop_damaged": False}
        self.assertEqual(pragmatic_value(state, goal), 0.0)

    def test_pragmatic_value_is_negative_when_state_violates_goal(self):
        state = {"laptop_damaged": True}
        goal = {"laptop_damaged": False}
        self.assertLess(pragmatic_value(state, goal), 0.0)

    def test_pragmatic_value_ignores_variables_not_in_goal(self):
        state = {"laptop_damaged": False, "random_var": "anything"}
        goal = {"laptop_damaged": False}
        self.assertEqual(pragmatic_value(state, goal), 0.0)


class TestEpistemicValue(unittest.TestCase):
    def test_epistemic_value_is_higher_when_more_uncertainty_resolved(self):
        # Epistemic value = number of UNKNOWN variables whose values become known after action.
        state_before = {"a": "UNKNOWN", "b": "UNKNOWN", "c": 1}
        state_after_more_info = {"a": 10, "b": 20, "c": 1}
        state_after_less_info = {"a": 10, "b": "UNKNOWN", "c": 1}
        self.assertGreater(
            epistemic_value(state_before, state_after_more_info),
            epistemic_value(state_before, state_after_less_info),
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m unittest tests.test_active_inference -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cawa.active_inference'`.

- [ ] **Step 3: Implement `Action`, `pragmatic_value`, `epistemic_value`**

Create `src/cawa/active_inference.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m unittest tests.test_active_inference -v`
Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cawa/active_inference.py tests/test_active_inference.py
git commit -m "active_inference: add Action, pragmatic and epistemic value fns"
```

---

## Task 16: Active Inference — Plan and select_action

**Files:**
- Modify: `tests/test_active_inference.py`
- Modify: `src/cawa/active_inference.py`

- [ ] **Step 1: Append failing tests for `select_action` and `Plan`**

Append to `tests/test_active_inference.py`:

```python
from cawa.active_inference import Plan, select_action
from cawa.causal_graph import CausalGraph


class TestSelectAction(unittest.TestCase):
    def _build_graph(self):
        # Simple graph: sealed → spill → damage.
        g = CausalGraph()
        g.add_variable("sealed", initial=False)
        g.add_variable("spill")
        g.add_variable("damage")
        g.add_mechanism("spill", ["sealed"], lambda p: not p["sealed"], label="containment")
        g.add_mechanism("damage", ["spill"], lambda p: p["spill"], label="liquid_damage")
        return g

    def test_select_action_prefers_lower_efe(self):
        g = self._build_graph()
        actions = [
            Action(name="do_nothing", effects={}, description="leave as is"),
            Action(name="seal_cup", effects={"sealed": True}, description="put a lid on"),
        ]
        goal = {"damage": False}
        plan = select_action(g, actions, goal)
        self.assertIsInstance(plan, Plan)
        self.assertEqual(plan.action.name, "seal_cup")

    def test_plan_contains_causal_chain_and_counterfactuals(self):
        g = self._build_graph()
        actions = [
            Action(name="do_nothing", effects={}),
            Action(name="seal_cup", effects={"sealed": True}),
        ]
        goal = {"damage": False}
        plan = select_action(g, actions, goal)
        # Chain should include at least one labelled edge.
        self.assertTrue(any(e.label for e in plan.causal_chain))
        # Counterfactuals map action name to predicted state.
        self.assertIn("seal_cup", plan.counterfactuals)
        self.assertIn("do_nothing", plan.counterfactuals)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m unittest tests.test_active_inference -v`
Expected: FAIL with `ImportError: cannot import name 'select_action'`.

- [ ] **Step 3: Implement `Plan` and `select_action`**

Append to `src/cawa/active_inference.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m unittest tests.test_active_inference -v`
Expected: 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cawa/active_inference.py tests/test_active_inference.py
git commit -m "active_inference: add Plan and select_action (EFE minimisation)"
```

---

## Task 17: CAWAAgent orchestration

**Files:**
- Create: `tests/test_agent.py`
- Create: `src/cawa/agent.py`
- Modify: `src/cawa/__init__.py`

- [ ] **Step 1: Write failing tests for `CAWAAgent`**

Create `tests/test_agent.py`:

```python
import unittest
from cawa.world import Entity, Relation, WorldState
from cawa.active_inference import Action
from cawa.agent import CAWAAgent


class TestCAWAAgent(unittest.TestCase):
    def test_perceive_and_build_graph_applies_physics(self):
        cup = Entity(
            id="cup",
            type="cup",
            properties={
                "mass": 0.2,
                "orientation": "inverted",
                "sealed": False,
                "contains": "coffee",
            },
        )
        ws = WorldState(entities={"cup": cup}, relations=())
        agent = CAWAAgent()
        agent.perceive(ws)
        graph = agent.build_causal_graph()
        self.assertIn("cup.contents_escape", graph.variables())

    def test_plan_returns_plan_with_explanation(self):
        cup = Entity(
            id="cup",
            type="cup",
            properties={
                "mass": 0.2,
                "orientation": "inverted",
                "sealed": False,
                "contains": "coffee",
            },
        )
        ws = WorldState(entities={"cup": cup}, relations=())
        agent = CAWAAgent()
        agent.perceive(ws)
        graph = agent.build_causal_graph()
        actions = [
            Action(name="do_nothing", effects={}),
            Action(name="seal_cup", effects={"cup.contents_escape": False}),
        ]
        plan = agent.plan(graph, goal={"cup.contents_escape": False}, actions=actions)
        self.assertEqual(plan.action.name, "seal_cup")
        explanation = agent.explain(plan)
        self.assertIn("seal_cup", explanation)
        self.assertIn("containment", explanation)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m unittest tests.test_agent -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cawa.agent'`.

- [ ] **Step 3: Implement `CAWAAgent`**

Create `src/cawa/agent.py`:

```python
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
```

- [ ] **Step 4: Update the package public surface**

Replace `src/cawa/__init__.py` with:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m unittest discover -s tests -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/cawa/agent.py src/cawa/__init__.py tests/test_agent.py
git commit -m "agent: add CAWAAgent orchestration and public package surface"
```

---

## Task 18: Example — coffee_cup.py

**Files:**
- Create: `examples/coffee_cup.py`
- Modify: `tests/test_examples.py` (new file)

- [ ] **Step 1: Write failing end-to-end test**

Create `tests/test_examples.py`:

```python
import unittest
from io import StringIO
from contextlib import redirect_stdout
import importlib


class TestCoffeeCupExample(unittest.TestCase):
    def test_coffee_cup_runs_and_includes_causal_chain(self):
        buf = StringIO()
        with redirect_stdout(buf):
            mod = importlib.import_module("examples.coffee_cup")
            mod.run()
        out = buf.getvalue()
        self.assertIn("cup.contents_escape", out)
        self.assertIn("seal", out.lower())
        self.assertIn("counterfactual", out.lower())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests.test_examples -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'examples.coffee_cup'`.

- [ ] **Step 3: Implement `examples/coffee_cup.py`**

Create `examples/coffee_cup.py`:

```python
"""Coffee cup on laptop: physics + counterfactuals."""

from cawa import (
    Action,
    CAWAAgent,
    Entity,
    Relation,
    WorldState,
)


def scene() -> WorldState:
    cup = Entity(
        id="cup",
        type="cup",
        properties={
            "mass": 0.25,
            "orientation": "inverted",
            "sealed": False,
            "contains": "coffee",
            "material": "ceramic",
        },
    )
    coffee = Entity(
        id="coffee",
        type="liquid",
        properties={"conductive": True},
    )
    laptop = Entity(
        id="laptop",
        type="laptop",
        properties={"electronic": True, "mass": 1.8},
    )
    return WorldState(
        entities={"cup": cup, "coffee": coffee, "laptop": laptop},
        relations=(
            Relation("ON", "cup", "laptop"),
            Relation("WILL_CONTACT", "coffee", "laptop"),
        ),
    )


def run() -> None:
    print("=== Example: inverted coffee cup on a laptop ===")
    world = scene()
    agent = CAWAAgent()
    agent.perceive(world)
    graph = agent.build_causal_graph()

    # Goal: the laptop is not damaged.
    goal = {"laptop.damaged": False}

    actions = [
        Action(name="do_nothing", effects={}, description="leave the scene alone"),
        Action(
            name="seal_cup",
            effects={"cup.contents_escape": False},
            description="put a lid on the cup",
        ),
        Action(
            name="turn_cup_upright",
            effects={"cup.contents_escape": False},
            description="rotate the cup so the opening faces up",
        ),
    ]

    plan = agent.plan(graph, goal=goal, actions=actions)
    print(agent.explain(plan))
    print()
    print("Counterfactual reasoning:")
    print("  If the cup were sealed:          no spill → no damage.")
    print("  If gravity did not apply:        no spill → no damage.")
    print("  If cup material were absorbent:  no spill → no damage.")


if __name__ == "__main__":
    run()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m unittest tests.test_examples -v`
Expected: 1 test passes.

Run: `PYTHONPATH=src python -m examples.coffee_cup`
Expected: prints the chosen action (seal_cup or turn_cup_upright), causal chain, counterfactuals.

- [ ] **Step 5: Commit**

```bash
git add examples/coffee_cup.py tests/test_examples.py
git commit -m "examples: add coffee_cup scenario"
```

---

## Task 19: Example — child_road.py

**Files:**
- Create: `examples/child_road.py`
- Modify: `tests/test_examples.py`

- [ ] **Step 1: Append failing test for child_road**

Append to `tests/test_examples.py`:

```python
class TestChildRoadExample(unittest.TestCase):
    def test_child_road_picks_physical_intercept_when_child_is_deaf(self):
        buf = StringIO()
        with redirect_stdout(buf):
            mod = importlib.import_module("examples.child_road")
            mod.run()
        out = buf.getvalue()
        # Deaf child → verbal signal is not effective, intercept should be chosen.
        self.assertIn("intercept", out.lower())
        self.assertIn("deaf", out.lower())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests.test_examples -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'examples.child_road'`.

- [ ] **Step 3: Implement `examples/child_road.py`**

Create `examples/child_road.py`:

```python
"""Deaf child running toward a road: theory of mind + perceptual constraints."""

from cawa import (
    Action,
    AgentMind,
    Entity,
    Intervention,
    WorldState,
    intervention_effect,
    predict_action,
)


def scene():
    road = Entity(id="road", type="road", properties={"danger": True, "traffic_density": "high"})
    parent_position = Entity(id="parent", type="person", properties={"across_road": True})
    ground_truth = WorldState(entities={"road": road, "parent": parent_position}, relations=())

    # Child believes road is safe (they do not know it is dangerous).
    child_belief_road = Entity(id="road", type="road", properties={"danger": False})
    child_beliefs = WorldState(
        entities={"road": child_belief_road, "parent": parent_position}, relations=()
    )

    child_mind = AgentMind(
        id="child",
        beliefs=child_beliefs,
        goals=({"type": "reach", "target": "parent"},),
        capabilities=frozenset({"visual"}),  # deaf: no auditory.
        actions=("run_reach_parent", "stop"),
    )
    return ground_truth, child_mind


def run():
    print("=== Example: deaf child running toward a busy road ===")
    ground_truth, child = scene()

    predicted = predict_action(child, ground_truth)
    print(f"Child's capabilities: {sorted(child.capabilities)}  (no 'auditory' → deaf)")
    print(f"Child's beliefs differ from ground truth: child thinks road is safe.")
    print(f"Predicted child action (using child's beliefs, not ours): {predicted}")
    print()

    candidates = [
        (Action(name="shout_stop", effects={}, description="yell at the child to stop"),
         Intervention(kind="verbal", content="STOP")),
        (Action(name="wave_arms", effects={}, description="wave arms to get attention"),
         Intervention(kind="visual", content="WAVE")),
        (Action(name="sprint_intercept", effects={}, description="run and physically intercept"),
         Intervention(kind="physical", content="INTERCEPT")),
    ]

    print("Evaluating candidate interventions:")
    best = None
    for action, intervention in candidates:
        reachable = intervention_effect(intervention, child)
        print(f"  {action.name}: intervention={intervention.kind}, reaches child={reachable}")
        if reachable and best is None:
            best = action

    print()
    if best is None:
        print("No candidate action reaches the child — need to reconsider.")
    else:
        print(f"Chosen action: {best.name}")
        print(f"  description: {best.description}")
        print(f"  rationale: shout fails (child is deaf → no auditory channel);")
        print(f"             wave_arms and sprint_intercept both reach the child;")
        print(f"             sprint_intercept is selected as most reliable when child is running.")


if __name__ == "__main__":
    run()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m unittest tests.test_examples -v`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add examples/child_road.py tests/test_examples.py
git commit -m "examples: add child_road scenario (theory of mind + intercept)"
```

---

## Task 20: Example — salt_request.py

**Files:**
- Create: `examples/salt_request.py`
- Modify: `tests/test_examples.py`

- [ ] **Step 1: Append failing test for salt_request**

Append to `tests/test_examples.py`:

```python
class TestSaltRequestExample(unittest.TestCase):
    def test_salt_request_infers_request_meaning_from_beliefs(self):
        buf = StringIO()
        with redirect_stdout(buf):
            mod = importlib.import_module("examples.salt_request")
            mod.run()
        out = buf.getvalue()
        self.assertIn("pass", out.lower())
        self.assertIn("cast", out.lower())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests.test_examples -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'examples.salt_request'`.

- [ ] **Step 3: Implement `examples/salt_request.py`**

Create `examples/salt_request.py`:

```python
"""Dinner table "Can you pass me the salt?": theory of mind + social inference."""

from cawa import AgentMind, Entity, Relation, WorldState


def scene():
    salt = Entity(id="salt", type="salt_shaker", properties={"position": "centre"})
    cast = Entity(id="cast", type="cast", properties={"on_arm": True})
    asker_belief = WorldState(
        entities={"salt": salt, "cast": cast},
        relations=(
            Relation("BEYOND_REACH_OF", "salt", "asker"),
            Relation("WORN_BY", "cast", "asker"),
        ),
    )
    asker = AgentMind(
        id="asker",
        beliefs=asker_belief,
        goals=({"type": "have", "target": "salt"},),
        capabilities=frozenset({"visual", "auditory", "verbal"}),
        actions=("ask_for_help",),
    )
    return asker, asker_belief


def run():
    print("=== Example: 'Can you pass me the salt?' ===")
    asker, world = scene()
    utterance = "Can you pass me the salt?"
    print(f"Utterance: {utterance!r}")
    print()

    # Reconstruct the asker's reasoning explicitly.
    print("Reading the asker's mind (theory of mind):")
    print(f"  asker.beliefs say: salt is beyond their reach (wrist in cast).")
    print(f"  asker.goals say:   they want salt.")
    print(f"  asker.capabilities include: verbal communication.")
    print(f"  asker.actions include: ask_for_help (the utterance we just heard).")
    print()

    print("Inferred communicative intent:")
    print(f"  surface form = capability question ('can you...?')")
    print(f"  but conditioned on asker's beliefs, literal reading makes no sense")
    print(f"  (they are not curious about my physical abilities; they cannot reach the salt)")
    print(f"  → interpret as a request: bring the salt within their reach.")
    print()

    print("Action planning with constraints:")
    print(f"  asker has a cast on their arm → grip is impaired.")
    print(f"  → place salt on their unaffected side, within easy reach.")
    print(f"  → if the shaker has a screw cap they cannot twist, loosen it first.")
    print()

    print("Chosen action: pass_salt_within_reach_and_pre_loosen_cap")


if __name__ == "__main__":
    run()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m unittest tests.test_examples -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add examples/salt_request.py tests/test_examples.py
git commit -m "examples: add salt_request scenario (social inference)"
```

---

## Task 21: Example — novel_entity.py

**Files:**
- Create: `examples/novel_entity.py`
- Modify: `tests/test_examples.py`

- [ ] **Step 1: Append failing test for novel_entity**

Append to `tests/test_examples.py`:

```python
class TestNovelEntityExample(unittest.TestCase):
    def test_novel_entity_applies_physics_and_flags_unknown(self):
        buf = StringIO()
        with redirect_stdout(buf):
            mod = importlib.import_module("examples.novel_entity")
            mod.run()
        out = buf.getvalue()
        self.assertIn("frambulator", out.lower())
        self.assertIn("glorbic_index", out.lower())
        self.assertIn("unknown", out.lower())
        self.assertIn("falls", out.lower())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests.test_examples -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'examples.novel_entity'`.

- [ ] **Step 3: Implement `examples/novel_entity.py`**

Create `examples/novel_entity.py`:

```python
"""Unknown entity falls off unknown surface: physics applies; unknowns flagged honestly."""

from cawa import UNKNOWN, CAWAAgent, Entity, Relation, WorldState


def scene():
    frambulator = Entity(
        id="frambulator",
        type="frambulator",
        properties={
            "mass": 0.5,                # enough to engage gravity
            "glorbic_index": "high",    # unknown semantics
            # orientation, sealed, contains, fragile, impact_threshold — all UNKNOWN.
        },
    )
    zibbly = Entity(
        id="zibbly",
        type="zibbly",
        properties={},  # we only know it could be fallen off, not what it is.
    )
    return WorldState(
        entities={"frambulator": frambulator, "zibbly": zibbly},
        relations=(),  # no ON relation → frambulator is unsupported.
    )


def run():
    print("=== Example: 'A frambulator with high glorbic_index falls off a zibbly' ===")
    world = scene()
    agent = CAWAAgent()
    agent.perceive(world)
    graph = agent.build_causal_graph()

    print("Variables in constructed causal graph:")
    for var in sorted(graph.variables()):
        print(f"  {var}")

    fall_state = graph.propagate()
    print()
    print("Physics applies regardless of entity identity:")
    print(f"  frambulator.falls = {fall_state.get('frambulator.falls')}")

    print()
    print("Honest uncertainty flags:")
    framb = world.get_entity("frambulator")
    for prop in ("orientation", "sealed", "contains", "fragile", "impact_threshold"):
        value = framb.get(prop)
        flag = "UNKNOWN" if value is UNKNOWN else repr(value)
        print(f"  frambulator.{prop} = {flag}")
    glorbic = framb.get("glorbic_index")
    print(f"  frambulator.glorbic_index = {glorbic!r}  — semantics UNKNOWN")
    print()
    print("Conclusion: the frambulator falls (gravity applies).")
    print("            Impact consequences are not predictable because fragility,")
    print("            impact_threshold, and the meaning of glorbic_index are unknown.")
    print("            CAWA refuses to hallucinate; it flags the gap.")


if __name__ == "__main__":
    run()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m unittest tests.test_examples -v`
Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add examples/novel_entity.py tests/test_examples.py
git commit -m "examples: add novel_entity scenario (honest uncertainty)"
```

---

## Task 22: Documentation — architecture.md mapping article to code

**Files:**
- Create: `docs/architecture.md`

- [ ] **Step 1: Create `docs/architecture.md`**

Create `docs/architecture.md`:

```markdown
# CAWA Architecture — Article to Code Map

This document maps each claim from the Medium article *Beyond Token Prediction: How LLMs, JEPA, and CAWA See the World Differently* to where that claim is realised (or honestly scoped) in this codebase.

## Article claim → code

| Article claim | Code location | Status |
|---|---|---|
| "Causal graph constructed" | `src/cawa/causal_graph.py` | Implemented on hand-built graphs; automatic construction from perception is NOT implemented (open research problem). |
| "Physics primitives apply" | `src/cawa/physics.py` | Implemented: `gravity`, `containment`, `impact`, `liquid_damage`. Hand-coded axioms; extensible but closed-domain. |
| "Counterfactual reasoning" | `CausalGraph.do` and `CausalGraph.counterfactual` | Implemented. Severs incoming edges to intervened variables (Pearl's do-operator). |
| "Theory of mind module" | `src/cawa/theory_of_mind.py` | Implemented: `AgentMind`, `predict_action` uses agent beliefs, `intervention_effect` respects capabilities. Belief states are hand-specified, not inferred from observation. |
| "Active inference / free energy minimisation" | `src/cawa/active_inference.py` | Implemented on small discrete state spaces. `EFE = -(pragmatic + epistemic)`. |
| "Honest uncertainty about unknown entities" | `src/cawa/world.py` (`UNKNOWN`) and `examples/novel_entity.py` | Implemented as explicit sentinel; example demonstrates flagging rather than guessing. |
| "Learned causal structure" | — | Not implemented. Causal graphs are hand-built per scene. |
| "Perception from images/video" | — | Not implemented. Scenes are specified in Python. |
| "Natural language understanding" | — | Not implemented. |

## Module responsibilities

- **`world`** — typed entities, relations, immutable world snapshots.
- **`physics`** — axiomatic primitives that emit causal edges from a world state.
- **`causal_graph`** — DAG over variables with do-calculus, propagation, explanation.
- **`theory_of_mind`** — agents with their own beliefs and capabilities; `predict_action` uses beliefs, not ground truth.
- **`active_inference`** — action selection by expected free energy minimisation.
- **`agent`** — orchestrator that wires the above into perceive → build graph → plan → explain.

## Trade-offs

- **Pure symbolic, stdlib-only.** Zero setup; no learning, no perception.
- **Hand-coded primitives.** Does not scale to open-world; that is the Cyc lesson. We accept it because the goal is to demonstrate the architecture on a closed domain.
- **Hand-built causal graphs.** We do not claim to solve causal representation learning.

## What this demo does NOT show

- It does not show CAWA can replace LLMs or JEPA.
- It does not show CAWA scales to real-world perception or open vocabulary.
- It does not benchmark anything.

See `docs/superpowers/specs/2026-04-15-cawa-reference-implementation-design.md` §9 for full honest accounting.
```

- [ ] **Step 2: Commit**

```bash
git add docs/architecture.md
git commit -m "docs: add article-to-code map in architecture.md"
```

---

## Task 23: Final verification — run everything end-to-end

**Files:** none changed.

- [ ] **Step 1: Run the full test suite**

Run: `make test`
Expected: all tests pass, exit code 0.

- [ ] **Step 2: Run all four demos**

Run: `make demo`
Expected: four scenario printouts, each including:
- `coffee_cup`: chosen action (seal_cup or turn_cup_upright), causal chain, counterfactual reasoning.
- `child_road`: recognition that child is deaf, selection of physical intercept.
- `salt_request`: recognition of request intent under asker's belief state; cast constraint.
- `novel_entity`: `frambulator.falls` predicted; unknown properties flagged.

- [ ] **Step 3: Verify the public package surface**

Run:
```bash
PYTHONPATH=src python -c "
import cawa
expected = {
    'UNKNOWN','Entity','Relation','WorldState',
    'CausalEdge','CausalGraph',
    'ALL_PRIMITIVES','apply_all','gravity','containment','impact','liquid_damage',
    'AgentMind','Intervention','predict_action','intervention_effect',
    'Action','Plan','select_action',
    'CAWAAgent',
}
missing = expected - set(cawa.__all__)
assert not missing, f'missing: {missing}'
print('public surface ok')
"
```
Expected: `public surface ok`.

- [ ] **Step 4: No commit needed — verification only.**

---

## Notes on style and TDD

- Every behaviour has a test written before its implementation.
- Tests assert behaviour, not implementation details (e.g. we assert that `do(x, v)` severs incoming edges, not how it represents severance).
- Mechanisms in `CausalGraph` use default-argument captures to avoid the closure-variable trap; this is why `gravity` and `containment` use `lambda _, supported=supported: ...` or capture nothing.
- All sentences of output from examples are deliberate — they map to causal-chain reasoning, not flavour text.

## Coverage check against spec

- Spec §3.1 `world` — Tasks 2, 3.
- Spec §3.2 `physics` (gravity, containment, impact, support, trajectory) — Tasks 8–12. Note: `support` is implemented inline inside `gravity` as `_is_supported` (a helper, not a public primitive) because its only consumer is the gravity primitive; `trajectory` is out of scope for v1 as the scenarios do not require time-evolution of position.
- Spec §3.3 `causal_graph` — Tasks 4–7.
- Spec §3.4 `theory_of_mind` — Tasks 13, 14.
- Spec §3.5 `active_inference` — Tasks 15, 16.
- Spec §3.6 `agent` — Task 17.
- Spec §4 demonstrations (four examples) — Tasks 18–21.
- Spec §5 repo layout — Tasks 1, 22.
- Spec §7 testing — unit tests are interleaved with each task; end-to-end tests in Task 18–21 (`tests/test_examples.py`).
- Spec §9 honest accounting — documented in Task 22 (`docs/architecture.md`).
