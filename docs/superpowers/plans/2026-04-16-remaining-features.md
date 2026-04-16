# Remaining Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add prototype implementations for learned causal structure (causal-learn), perception (YOLOv8-nano), and NLU (spaCy) to the telos codebase.

**Architecture:** Three new modules under `src/telos/` — each converts external input (data, images, text) into existing telos types (`CausalGraph`, `WorldState`). Each module has a corresponding example and test. Dependencies added to `pyproject.toml`.

**Tech Stack:** `causal-learn` (PC algorithm), `ultralytics` (YOLOv8-nano), `spacy` + `en_core_web_sm`

---

### Task 1: Add Dependencies and Install Target

**Files:**
- Modify: `pyproject.toml`
- Modify: `Makefile`

- [ ] **Step 1: Update pyproject.toml with new dependencies**

```toml
[project]
name = "telos"
version = "0.1.0"
description = "telos — a reference implementation of the Causal Active World Architecture (CAWA)"
requires-python = ">=3.10"
dependencies = [
    "causal-learn",
    "ultralytics",
    "spacy",
]
```

- [ ] **Step 2: Add install target to Makefile**

Add before the `clean` target:

```makefile
install:
	pip install -e .
	python3 -m spacy download en_core_web_sm
```

- [ ] **Step 3: Run install**

Run: `make install`
Expected: all three libraries install successfully, spaCy model downloads.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml Makefile
git commit -m "build: add causal-learn, ultralytics, spacy dependencies"
```

---

### Task 2: Structure Learner — Test

**Files:**
- Create: `tests/test_structure_learner.py`

- [ ] **Step 1: Write failing tests for structure learner**

```python
import unittest
import numpy as np
from telos.causal_graph import CausalGraph
from telos.structure_learner import generate_samples, learn_graph, compare_graphs


class TestGenerateSamples(unittest.TestCase):
    def test_generate_samples_returns_correct_shape(self):
        """Generating 100 samples from 3 variables produces a (100, 3) array."""
        from telos.world import Entity, WorldState

        cup = Entity(id="cup", type="cup", properties={
            "mass": 0.25, "orientation": "inverted", "sealed": False,
            "contains": "coffee", "material": "ceramic",
        })
        coffee = Entity(id="coffee", type="liquid", properties={"conductive": True})
        laptop = Entity(id="laptop", type="laptop", properties={"electronic": True, "mass": 1.8})
        world = WorldState(
            entities={"cup": cup, "coffee": coffee, "laptop": laptop},
            relations=(),
        )
        from telos.physics import ALL_PRIMITIVES
        samples, names = generate_samples(world, ALL_PRIMITIVES, n=100)
        self.assertEqual(samples.shape[0], 100)
        self.assertGreater(samples.shape[1], 0)
        self.assertEqual(len(names), samples.shape[1])

    def test_generate_samples_variable_names_match_physics(self):
        """Variable names come from causal edges emitted by physics primitives."""
        from telos.world import Entity, WorldState
        from telos.physics import ALL_PRIMITIVES

        cup = Entity(id="cup", type="cup", properties={
            "mass": 0.25, "orientation": "inverted", "sealed": False,
            "contains": "coffee", "material": "ceramic",
        })
        coffee = Entity(id="coffee", type="liquid", properties={"conductive": True})
        laptop = Entity(id="laptop", type="laptop", properties={"electronic": True, "mass": 1.8})
        world = WorldState(
            entities={"cup": cup, "coffee": coffee, "laptop": laptop},
            relations=(
                __import__("telos.world", fromlist=["Relation"]).Relation("WILL_CONTACT", "coffee", "laptop"),
            ),
        )
        samples, names = generate_samples(world, ALL_PRIMITIVES, n=10)
        self.assertIn("cup.contents_escape", names)


class TestLearnGraph(unittest.TestCase):
    def test_learn_graph_returns_causal_graph(self):
        """PC algorithm output is converted to a CausalGraph."""
        # Simple chain: A -> B -> C
        rng = np.random.default_rng(42)
        n = 500
        a = rng.normal(size=n)
        b = a * 2.0 + rng.normal(size=n) * 0.1
        c = b * 3.0 + rng.normal(size=n) * 0.1
        samples = np.column_stack([a, b, c])
        names = ["a", "b", "c"]
        graph = learn_graph(samples, names)
        self.assertIsInstance(graph, CausalGraph)
        self.assertIn("a", graph.variables())
        self.assertIn("b", graph.variables())
        self.assertIn("c", graph.variables())

    def test_learn_graph_recovers_edges_in_linear_chain(self):
        """PC algorithm should recover A->B and B->C edges in a linear chain."""
        rng = np.random.default_rng(42)
        n = 1000
        a = rng.normal(size=n)
        b = a * 2.0 + rng.normal(size=n) * 0.1
        c = b * 3.0 + rng.normal(size=n) * 0.1
        samples = np.column_stack([a, b, c])
        names = ["a", "b", "c"]
        graph = learn_graph(samples, names)
        # b should have at least one incoming edge (from a)
        edges_into_b = graph.edges_into("b")
        self.assertGreater(len(edges_into_b), 0)


class TestCompareGraphs(unittest.TestCase):
    def test_identical_graphs_have_perfect_scores(self):
        g1 = CausalGraph()
        g1.add_variable("a")
        g1.add_variable("b")
        g1.add_mechanism("b", ["a"], lambda p: p["a"], label="a->b")

        g2 = CausalGraph()
        g2.add_variable("a")
        g2.add_variable("b")
        g2.add_mechanism("b", ["a"], lambda p: p["a"], label="a->b")

        metrics = compare_graphs(g1, g2)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_empty_learned_graph_has_zero_recall(self):
        g_true = CausalGraph()
        g_true.add_variable("a")
        g_true.add_variable("b")
        g_true.add_mechanism("b", ["a"], lambda p: p["a"], label="a->b")

        g_empty = CausalGraph()
        g_empty.add_variable("a")
        g_empty.add_variable("b")

        metrics = compare_graphs(g_empty, g_true)
        self.assertEqual(metrics["recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python3 -m pytest tests/test_structure_learner.py -v`
Expected: ImportError — `telos.structure_learner` does not exist yet.

---

### Task 3: Structure Learner — Implementation

**Files:**
- Create: `src/telos/structure_learner.py`

- [ ] **Step 1: Implement structure_learner.py**

```python
"""Learned causal structure: discover causal graphs from observational data.

Uses the PC algorithm from causal-learn to recover a DAG from samples,
then converts it into a telos CausalGraph.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from causallearn.search.ConstraintBased.PC import pc

from .causal_graph import CausalEdge, CausalGraph
from .physics import apply_all
from .world import Entity, WorldState


def generate_samples(
    world: WorldState,
    primitives: list[Callable[[WorldState], list[CausalEdge]]],
    n: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """Generate observational data by perturbing a world state and running physics.

    For each sample, randomly toggle root-cause conditions (support, orientation,
    sealing) and record which downstream effects fire. Returns an (n, num_vars)
    array and a list of variable names.
    """
    # First, get the variable names from a baseline run.
    baseline_edges = apply_all(world, primitives)
    all_vars: list[str] = []
    seen: set[str] = set()
    for edge in baseline_edges:
        for v in (*edge.parents, edge.effect):
            if v not in seen:
                all_vars.append(v)
                seen.add(v)

    if not all_vars:
        return np.zeros((n, 0)), []

    # Build a mapping from variable to its edges for fast lookup.
    effect_to_edge: dict[str, CausalEdge] = {}
    for edge in baseline_edges:
        effect_to_edge[edge.effect] = edge

    # Identify root variables (no incoming edges).
    effects = {e.effect for e in baseline_edges}
    roots = [v for v in all_vars if v not in effects]

    rng = np.random.default_rng(seed)
    data = np.zeros((n, len(all_vars)), dtype=float)

    for i in range(n):
        state: dict[str, float] = {}
        # Randomly set root variables.
        for v in roots:
            state[v] = float(rng.choice([0.0, 1.0]))

        # Propagate through edges in order.
        for v in all_vars:
            if v in state:
                continue
            edge = effect_to_edge.get(v)
            if edge is None:
                state[v] = float(rng.choice([0.0, 1.0]))
                continue
            parent_vals = {p: state.get(p, 0.0) for p in edge.parents}
            try:
                result = edge.mechanism(parent_vals)
                state[v] = float(bool(result))
            except Exception:
                state[v] = 0.0

        for j, v in enumerate(all_vars):
            data[i, j] = state.get(v, 0.0)

    return data, all_vars


def learn_graph(
    samples: np.ndarray,
    variable_names: list[str],
    alpha: float = 0.05,
) -> CausalGraph:
    """Run the PC algorithm on observational data and return a telos CausalGraph.

    Args:
        samples: (n_samples, n_variables) array of observations.
        variable_names: name for each column.
        alpha: significance level for conditional independence tests.
    """
    result = pc(samples, alpha=alpha, indep_test="fisherz", node_names=variable_names)
    adj_matrix = result.G.graph  # shape (n, n)

    graph = CausalGraph()
    for name in variable_names:
        graph.add_variable(name)

    n = len(variable_names)
    for i in range(n):
        for j in range(n):
            # causal-learn encodes: adj[i,j] = -1 and adj[j,i] = 1 means i -> j
            if adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1:
                parent = variable_names[i]
                child = variable_names[j]
                graph.add_mechanism(
                    child,
                    [parent],
                    mechanism=lambda p, pn=parent: p[pn],
                    label=f"learned({parent}->{child})",
                )

    return graph


def compare_graphs(
    learned: CausalGraph,
    ground_truth: CausalGraph,
) -> dict[str, float]:
    """Compare a learned graph against ground truth. Returns precision, recall, F1.

    Edges are compared as (parent, effect) pairs, ignoring mechanisms.
    """
    def _edge_set(g: CausalGraph) -> set[tuple[str, str]]:
        edges = set()
        for e in g.all_edges():
            for p in e.parents:
                edges.add((p, e.effect))
        return edges

    learned_edges = _edge_set(learned)
    true_edges = _edge_set(ground_truth)

    if not true_edges and not learned_edges:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    tp = len(learned_edges & true_edges)
    precision = tp / len(learned_edges) if learned_edges else 0.0
    recall = tp / len(true_edges) if true_edges else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `PYTHONPATH=src python3 -m pytest tests/test_structure_learner.py -v`
Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/telos/structure_learner.py tests/test_structure_learner.py
git commit -m "feat: add structure learner (PC algorithm via causal-learn)"
```

---

### Task 4: Structure Learner — Example

**Files:**
- Create: `examples/learned_structure.py`

- [ ] **Step 1: Write the example**

```python
"""Learned causal structure: recover the coffee cup causal graph from data."""

from telos import Entity, Relation, WorldState, CAWAAgent
from telos.physics import ALL_PRIMITIVES
from telos.structure_learner import generate_samples, learn_graph, compare_graphs


def run() -> None:
    print("=== Example: learning causal structure from observational data ===")

    # Build the coffee cup scene.
    cup = Entity(id="cup", type="cup", properties={
        "mass": 0.25, "orientation": "inverted", "sealed": False,
        "contains": "coffee", "material": "ceramic",
    })
    coffee = Entity(id="coffee", type="liquid", properties={"conductive": True})
    laptop = Entity(id="laptop", type="laptop", properties={"electronic": True, "mass": 1.8})
    world = WorldState(
        entities={"cup": cup, "coffee": coffee, "laptop": laptop},
        relations=(
            Relation("ON", "cup", "laptop"),
            Relation("WILL_CONTACT", "coffee", "laptop"),
        ),
    )

    # Generate observational samples.
    print("Generating 1000 observational samples from physics simulation...")
    samples, names = generate_samples(world, ALL_PRIMITIVES, n=1000)
    print(f"  Variables: {names}")
    print(f"  Sample shape: {samples.shape}")

    # Learn causal graph via PC algorithm.
    print("\nRunning PC algorithm to discover causal structure...")
    learned = learn_graph(samples, names)
    print(f"  Learned variables: {learned.variables()}")
    print(f"  Learned edges:")
    for edge in learned.all_edges():
        parents = ", ".join(edge.parents) if edge.parents else "(root)"
        print(f"    {parents} -> {edge.effect}  [{edge.label}]")

    # Compare against hand-built graph.
    print("\nComparing against hand-built causal graph...")
    agent = CAWAAgent()
    agent.perceive(world)
    handbuilt = agent.build_causal_graph()

    metrics = compare_graphs(learned, handbuilt)
    print(f"  Precision: {metrics['precision']:.2f}")
    print(f"  Recall:    {metrics['recall']:.2f}")
    print(f"  F1:        {metrics['f1']:.2f}")


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Run the example**

Run: `PYTHONPATH=src python3 -m examples.learned_structure`
Expected: prints sample generation, learned edges, and comparison metrics.

- [ ] **Step 3: Commit**

```bash
git add examples/learned_structure.py
git commit -m "examples: add learned_structure demo (PC algorithm vs hand-built)"
```

---

### Task 5: Perception — Test

**Files:**
- Create: `tests/test_perception.py`

- [ ] **Step 1: Write failing tests for perception**

```python
import unittest
from unittest.mock import patch, MagicMock
from telos.world import WorldState
from telos.perception import detect_objects, extract_relations, build_world


class TestDetectObjects(unittest.TestCase):
    @patch("telos.perception.YOLO")
    def test_detect_objects_returns_list_of_dicts(self, mock_yolo_cls):
        """detect_objects returns a list of detection dicts with label, confidence, bbox."""
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        # Simulate YOLO result.
        mock_box = MagicMock()
        mock_box.xyxy = MagicMock()
        mock_box.xyxy.cpu.return_value.numpy.return_value = [[100, 200, 300, 400]]
        mock_box.conf = MagicMock()
        mock_box.conf.cpu.return_value.numpy.return_value = [0.95]
        mock_box.cls = MagicMock()
        mock_box.cls.cpu.return_value.numpy.return_value = [41]  # 41 = "cup" in COCO

        mock_result = MagicMock()
        mock_result.boxes = mock_box
        mock_model.return_value = [mock_result]
        mock_model.names = {41: "cup"}

        detections = detect_objects("fake.jpg", model=mock_model)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["label"], "cup")
        self.assertAlmostEqual(detections[0]["confidence"], 0.95)
        self.assertEqual(detections[0]["bbox"], (100, 200, 300, 400))


class TestExtractRelations(unittest.TestCase):
    def test_on_relation_detected(self):
        """Object A sitting on top of object B produces an ON relation."""
        detections = [
            {"label": "cup", "confidence": 0.9, "bbox": (150, 80, 250, 150)},
            {"label": "dining table", "confidence": 0.9, "bbox": (50, 150, 400, 300)},
        ]
        relations = extract_relations(detections)
        rel_tuples = [(r.name, r.src, r.dst) for r in relations]
        self.assertIn(("ON", "cup_0", "dining table_1"), rel_tuples)

    def test_near_relation_detected(self):
        """Two objects close to each other produce a NEAR relation."""
        detections = [
            {"label": "cup", "confidence": 0.9, "bbox": (100, 100, 200, 200)},
            {"label": "laptop", "confidence": 0.9, "bbox": (210, 100, 400, 200)},
        ]
        relations = extract_relations(detections)
        rel_names = [r.name for r in relations]
        self.assertIn("NEAR", rel_names)

    def test_contains_relation_detected(self):
        """Object B inside object A produces a CONTAINS relation."""
        detections = [
            {"label": "bowl", "confidence": 0.9, "bbox": (50, 50, 350, 350)},
            {"label": "apple", "confidence": 0.9, "bbox": (100, 100, 200, 200)},
        ]
        relations = extract_relations(detections)
        rel_tuples = [(r.name, r.src, r.dst) for r in relations]
        self.assertIn(("CONTAINS", "bowl_0", "apple_1"), rel_tuples)


class TestBuildWorld(unittest.TestCase):
    @patch("telos.perception.detect_objects")
    def test_build_world_returns_world_state(self, mock_detect):
        mock_detect.return_value = [
            {"label": "cup", "confidence": 0.9, "bbox": (150, 80, 250, 150)},
            {"label": "dining table", "confidence": 0.9, "bbox": (50, 150, 400, 300)},
        ]
        world = build_world("fake.jpg")
        self.assertIsInstance(world, WorldState)
        self.assertIn("cup_0", world.entities)
        self.assertIn("dining table_1", world.entities)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python3 -m pytest tests/test_perception.py -v`
Expected: ImportError — `telos.perception` does not exist yet.

---

### Task 6: Perception — Implementation

**Files:**
- Create: `src/telos/perception.py`
- Create: `examples/assets/` directory (for sample images)

- [ ] **Step 1: Implement perception.py**

```python
"""Perception: image → WorldState via YOLOv8-nano object detection.

Detects objects with YOLO, derives spatial relations from bounding box
geometry, and constructs a telos WorldState.
"""

from __future__ import annotations

from ultralytics import YOLO

from .world import UNKNOWN, Entity, Relation, WorldState


_DEFAULT_MODEL: YOLO | None = None


def _get_model() -> YOLO:
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        _DEFAULT_MODEL = YOLO("yolov8n.pt")
    return _DEFAULT_MODEL


def detect_objects(
    image_path: str,
    model: YOLO | None = None,
    confidence_threshold: float = 0.3,
) -> list[dict]:
    """Run YOLOv8-nano on an image and return detections.

    Each detection is a dict with keys: label, confidence, bbox (x1, y1, x2, y2).
    """
    if model is None:
        model = _get_model()

    results = model(image_path, verbose=False)
    detections: list[dict] = []

    for result in results:
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        for i in range(len(xyxy)):
            conf = float(confs[i])
            if conf < confidence_threshold:
                continue
            cls_id = int(classes[i])
            label = model.names.get(cls_id, f"class_{cls_id}")
            x1, y1, x2, y2 = xyxy[i]
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
            })

    return detections


def extract_relations(
    detections: list[dict],
    near_threshold: float = 50.0,
    on_tolerance: float = 30.0,
    containment_ratio: float = 0.7,
) -> list[Relation]:
    """Derive spatial relations from bounding box geometry.

    Relations detected:
    - ON(A, B): A sits on top of B (A's bottom near B's top, A horizontally within B)
    - NEAR(A, B): bounding boxes within near_threshold pixels
    - CONTAINS(A, B): B's bbox mostly inside A's bbox
    """
    relations: list[Relation] = []

    for i, det_a in enumerate(detections):
        a_id = f"{det_a['label']}_{i}"
        ax1, ay1, ax2, ay2 = det_a["bbox"]
        a_cx = (ax1 + ax2) / 2
        a_cy = (ay1 + ay2) / 2
        a_area = (ax2 - ax1) * (ay2 - ay1)

        for j, det_b in enumerate(detections):
            if i == j:
                continue
            b_id = f"{det_b['label']}_{j}"
            bx1, by1, bx2, by2 = det_b["bbox"]
            b_cx = (bx1 + bx2) / 2
            b_cy = (by1 + by2) / 2
            b_area = (bx2 - bx1) * (by2 - by1)

            # ON: A's bottom edge near B's top edge, A horizontally overlaps B.
            if abs(ay2 - by1) < on_tolerance and ax1 >= bx1 - on_tolerance and ax2 <= bx2 + on_tolerance:
                relations.append(Relation("ON", a_id, b_id))

            # CONTAINS: B is mostly inside A.
            if a_area > b_area:
                overlap_x = max(0, min(ax2, bx2) - max(ax1, bx1))
                overlap_y = max(0, min(ay2, by2) - max(ay1, by1))
                overlap_area = overlap_x * overlap_y
                if b_area > 0 and overlap_area / b_area >= containment_ratio:
                    relations.append(Relation("CONTAINS", a_id, b_id))

            # NEAR: bounding boxes close to each other (avoid duplicates).
            if i < j:
                dist_x = max(0, max(ax1, bx1) - min(ax2, bx2))
                dist_y = max(0, max(ay1, by1) - min(ay2, by2))
                dist = (dist_x ** 2 + dist_y ** 2) ** 0.5
                if dist < near_threshold:
                    relations.append(Relation("NEAR", a_id, b_id))

    return relations


def build_world(
    image_path: str,
    model: YOLO | None = None,
) -> WorldState:
    """End-to-end: image → detect objects → extract relations → WorldState."""
    detections = detect_objects(image_path, model=model)

    entities: dict[str, Entity] = {}
    for i, det in enumerate(detections):
        eid = f"{det['label']}_{i}"
        entities[eid] = Entity(
            id=eid,
            type=det["label"],
            properties={
                "confidence": det["confidence"],
                "bbox": det["bbox"],
            },
        )

    relations = extract_relations(detections)

    return WorldState(entities=entities, relations=tuple(relations))
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `PYTHONPATH=src python3 -m pytest tests/test_perception.py -v`
Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/telos/perception.py tests/test_perception.py
git commit -m "feat: add perception module (YOLOv8-nano → WorldState)"
```

---

### Task 7: Perception — Example

**Files:**
- Create: `examples/perception_demo.py`
- Create: `examples/assets/` (directory for sample image)

- [ ] **Step 1: Create the assets directory**

```bash
mkdir -p examples/assets
```

- [ ] **Step 2: Write the example**

```python
"""Perception demo: image → WorldState → causal reasoning.

If run without an image argument, generates a simple test image using
Python's built-in libraries.
"""

import sys
import os

from telos import CAWAAgent
from telos.perception import build_world


def _create_sample_image(path: str) -> None:
    """Create a minimal PNG test image with colored rectangles (no PIL needed).

    Uses raw PNG encoding — just enough to give YOLO something to detect.
    If this does not produce detections, supply a real photograph instead.
    """
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (640, 480), (200, 200, 200))
        draw = ImageDraw.Draw(img)
        # Table-like rectangle.
        draw.rectangle([50, 250, 590, 450], fill=(139, 90, 43))
        # Cup-like rectangle on top of the table.
        draw.rectangle([250, 180, 350, 250], fill=(255, 255, 255))
        img.save(path)
    except ImportError:
        # Fallback: write a minimal 1x1 PNG so the example can still run.
        import struct, zlib
        def _chunk(ctype, data):
            c = ctype + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        raw = zlib.compress(b"\x00\xff\xff\xff")
        idat = _chunk(b"IDAT", raw)
        iend = _chunk(b"IEND", b"")
        with open(path, "wb") as f:
            f.write(sig + ihdr + idat + iend)
    print(f"  Created sample image: {path}")


def run() -> None:
    print("=== Example: perception → world state → causal reasoning ===")

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = os.path.join(os.path.dirname(__file__), "assets", "sample_scene.png")
        if not os.path.exists(image_path):
            print("No image provided. Creating a sample image...")
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            _create_sample_image(image_path)

    print(f"\nRunning YOLOv8-nano on: {image_path}")
    world = build_world(image_path)

    print(f"\nDetected {len(world.entities)} objects:")
    for eid, entity in world.entities.items():
        print(f"  {eid}: type={entity.type}, confidence={entity.get('confidence'):.2f}")

    print(f"\nExtracted {len(world.relations)} relations:")
    for rel in world.relations:
        print(f"  {rel.name}({rel.src}, {rel.dst})")

    if world.entities:
        print("\nBuilding causal graph from perceived world...")
        agent = CAWAAgent()
        agent.perceive(world)
        graph = agent.build_causal_graph()
        variables = graph.variables()
        if variables:
            print(f"  Causal variables: {variables}")
            state = graph.propagate()
            for var, val in state.items():
                print(f"    {var} = {val}")
        else:
            print("  No causal edges emitted (objects lack physics properties).")
            print("  This is expected for raw YOLO detections — the perception module")
            print("  provides entity detection; physics properties require domain knowledge.")
    else:
        print("\nNo objects detected in the image.")


if __name__ == "__main__":
    run()
```

- [ ] **Step 3: Run the example**

Run: `PYTHONPATH=src python3 -m examples.perception_demo`
Expected: creates sample image, runs YOLO, prints detected objects and relations.

- [ ] **Step 4: Commit**

```bash
git add examples/perception_demo.py examples/assets/
git commit -m "examples: add perception demo (YOLOv8-nano → WorldState)"
```

---

### Task 8: NLU — Test

**Files:**
- Create: `tests/test_nlu.py`

- [ ] **Step 1: Write failing tests for NLU**

```python
import unittest
from telos.world import WorldState
from telos.nlu import parse_scene, parse_query


class TestParseScene(unittest.TestCase):
    def test_simple_on_relation(self):
        """'A cup is on a table' extracts cup entity, table entity, ON relation."""
        world = parse_scene("A cup is on a table")
        self.assertIsInstance(world, WorldState)
        types = {e.type for e in world.entities.values()}
        self.assertIn("cup", types)
        self.assertIn("table", types)
        rel_names = [r.name for r in world.relations]
        self.assertIn("ON", rel_names)

    def test_near_relation(self):
        """'A laptop is near a cup' extracts NEAR relation."""
        world = parse_scene("A laptop is near a cup")
        rel_names = [r.name for r in world.relations]
        self.assertIn("NEAR", rel_names)

    def test_property_extraction(self):
        """'A heavy cup is on a wooden table' extracts adjective properties."""
        world = parse_scene("A heavy cup is on a wooden table")
        cup = None
        for e in world.entities.values():
            if e.type == "cup":
                cup = e
        self.assertIsNotNone(cup)
        self.assertIn("heavy", cup.properties.get("attributes", []))

    def test_multiple_entities(self):
        """'A cup is on a table near a laptop' extracts three entities."""
        world = parse_scene("A cup is on a table near a laptop")
        types = {e.type for e in world.entities.values()}
        self.assertTrue(len(types) >= 2)  # at least cup and table


class TestParseQuery(unittest.TestCase):
    def test_counterfactual_what_if(self):
        """'What happens if the cup falls?' is a counterfactual query."""
        query = parse_query("What happens if the cup falls?")
        self.assertEqual(query["type"], "counterfactual")
        self.assertIn("cup", str(query.get("subject", "")))

    def test_prediction_will(self):
        """'Will the laptop get damaged?' is a prediction query."""
        query = parse_query("Will the laptop get damaged?")
        self.assertEqual(query["type"], "prediction")
        self.assertIn("laptop", str(query.get("subject", "")))

    def test_counterfactual_what_would(self):
        """'What would happen if the cup were sealed?' is counterfactual."""
        query = parse_query("What would happen if the cup were sealed?")
        self.assertEqual(query["type"], "counterfactual")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python3 -m pytest tests/test_nlu.py -v`
Expected: ImportError — `telos.nlu` does not exist yet.

---

### Task 9: NLU — Implementation

**Files:**
- Create: `src/telos/nlu.py`

- [ ] **Step 1: Implement nlu.py**

```python
"""Natural Language Understanding: text → WorldState / structured queries.

Uses spaCy dependency parsing to extract entities, relations, and query intent
from natural language sentences.
"""

from __future__ import annotations

import spacy
from spacy.tokens import Doc

from .world import Entity, Relation, WorldState


_nlp: spacy.language.Language | None = None


def load_model() -> spacy.language.Language:
    """Load the spaCy English model (cached after first call)."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# Spatial prepositions → relation names.
_SPATIAL_MAP: dict[str, str] = {
    "on": "ON",
    "above": "ABOVE",
    "below": "BELOW",
    "near": "NEAR",
    "beside": "NEAR",
    "next": "NEAR",  # "next to"
    "inside": "CONTAINS",
    "in": "CONTAINS",
    "under": "BELOW",
    "over": "ABOVE",
}


def _extract_entities_and_relations(doc: Doc) -> tuple[dict[str, Entity], list[Relation]]:
    """Extract entities and spatial relations from a parsed sentence."""
    entities: dict[str, Entity] = {}
    relations: list[Relation] = []

    # Find noun chunks as candidate entities.
    chunks = list(doc.noun_chunks)
    entity_map: dict[int, str] = {}  # token index → entity id

    for chunk in chunks:
        # The root of the noun chunk is the head noun.
        head = chunk.root
        etype = head.lemma_.lower()
        eid = etype

        # Avoid duplicates by appending an index.
        if eid in entities:
            idx = 2
            while f"{eid}_{idx}" in entities:
                idx += 1
            eid = f"{eid}_{idx}"

        # Collect adjective modifiers as attributes.
        attrs = []
        for token in chunk:
            if token.pos_ == "ADJ":
                attrs.append(token.lemma_.lower())

        props: dict = {}
        if attrs:
            props["attributes"] = attrs

        entities[eid] = Entity(id=eid, type=etype, properties=props)
        for token in chunk:
            entity_map[token.i] = eid

    # Find spatial prepositions to derive relations.
    for token in doc:
        if token.dep_ == "prep" and token.lemma_.lower() in _SPATIAL_MAP:
            rel_name = _SPATIAL_MAP[token.lemma_.lower()]

            # Find the subject (entity before the prep).
            head_token = token.head
            src_id = None
            for idx, eid in entity_map.items():
                if idx == head_token.i or (head_token.i in range(
                    head_token.i - 3, head_token.i + 1
                )):
                    candidate = entity_map.get(head_token.i)
                    if candidate:
                        src_id = candidate
                        break
            if src_id is None:
                # Walk up to find the governing noun.
                h = head_token
                while h.head != h:
                    if h.i in entity_map:
                        src_id = entity_map[h.i]
                        break
                    h = h.head

            # Find the object of the preposition.
            dst_id = None
            for child in token.children:
                if child.dep_ == "pobj":
                    dst_id = entity_map.get(child.i)
                    if dst_id is None:
                        # Check if the pobj is part of a noun chunk.
                        for chunk in chunks:
                            if child.i >= chunk.start and child.i < chunk.end:
                                dst_id = entity_map.get(chunk.root.i)
                                break

            if src_id and dst_id:
                relations.append(Relation(rel_name, src_id, dst_id))

    return entities, relations


def parse_scene(text: str) -> WorldState:
    """Parse a natural language scene description into a WorldState.

    Example: "A coffee cup is on the edge of a wooden table"
    → WorldState with cup and table entities, ON relation.
    """
    nlp = load_model()
    doc = nlp(text)
    entities, relations = _extract_entities_and_relations(doc)

    return WorldState(
        entities=entities,
        relations=tuple(relations),
    )


def parse_query(text: str) -> dict:
    """Parse a natural language question into a structured query.

    Returns a dict with:
    - type: "counterfactual" or "prediction"
    - subject: the main entity mentioned
    - action: the verb/event mentioned (for counterfactuals)
    - target: what is being asked about (for predictions)
    """
    nlp = load_model()
    doc = nlp(text)
    text_lower = text.lower()

    # Determine query type.
    is_counterfactual = any(
        phrase in text_lower
        for phrase in ["what happens if", "what if", "what would happen"]
    )
    is_prediction = any(
        phrase in text_lower
        for phrase in ["will ", "is it going to", "is the", "are the", "does the"]
    )

    if is_counterfactual:
        query_type = "counterfactual"
    elif is_prediction:
        query_type = "prediction"
    else:
        query_type = "prediction"  # default

    # Extract subject and action from the sentence.
    subject = None
    action = None

    for chunk in doc.noun_chunks:
        # Skip question words.
        if chunk.root.lemma_.lower() in ("what", "which", "who"):
            continue
        if subject is None:
            subject = chunk.root.lemma_.lower()

    for token in doc:
        if token.pos_ == "VERB" and token.lemma_.lower() not in (
            "be", "do", "have", "happen", "would", "will", "can", "get",
        ):
            action = token.lemma_.lower()
            break

    return {
        "type": query_type,
        "subject": subject,
        "action": action,
    }
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `PYTHONPATH=src python3 -m pytest tests/test_nlu.py -v`
Expected: all tests pass (some pattern-matching tests may need tuning — adjust patterns in Step 3 if needed).

- [ ] **Step 3: Fix any failing tests by adjusting patterns**

If `parse_scene` or `parse_query` misparses specific sentences, adjust the extraction logic. spaCy dependency trees vary — debug by printing `[(t.text, t.dep_, t.head.text) for t in doc]`.

- [ ] **Step 4: Commit**

```bash
git add src/telos/nlu.py tests/test_nlu.py
git commit -m "feat: add NLU module (spaCy → WorldState / structured queries)"
```

---

### Task 10: NLU — Example

**Files:**
- Create: `examples/nlu_demo.py`

- [ ] **Step 1: Write the example**

```python
"""NLU demo: natural language → scene + query → causal explanation."""

from telos import CAWAAgent
from telos.nlu import parse_scene, parse_query


def run() -> None:
    print("=== Example: natural language understanding ===")

    # Describe a scene in plain English.
    scene_text = "A coffee cup is on a laptop"
    print(f"Scene description: {scene_text!r}")
    world = parse_scene(scene_text)

    print(f"\nParsed entities:")
    for eid, entity in world.entities.items():
        attrs = entity.properties.get("attributes", [])
        attr_str = f" [{', '.join(attrs)}]" if attrs else ""
        print(f"  {eid}: type={entity.type}{attr_str}")

    print(f"\nParsed relations:")
    for rel in world.relations:
        print(f"  {rel.name}({rel.src}, {rel.dst})")

    # Ask a counterfactual question.
    question = "What happens if the cup falls?"
    print(f"\nQuestion: {question!r}")
    query = parse_query(question)
    print(f"  Parsed query: {query}")

    # Ask a prediction question.
    question2 = "Will the laptop get damaged?"
    print(f"\nQuestion: {question2!r}")
    query2 = parse_query(question2)
    print(f"  Parsed query: {query2}")

    # Demonstrate the full pipeline with a richer scene.
    print("\n--- Full pipeline demo ---")
    rich_scene = "A heavy ceramic cup is on the edge of a wooden table near an open laptop"
    print(f"Scene: {rich_scene!r}")
    world2 = parse_scene(rich_scene)
    print(f"  Entities: {list(world2.entities.keys())}")
    print(f"  Relations: {[(r.name, r.src, r.dst) for r in world2.relations]}")


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Run the example**

Run: `PYTHONPATH=src python3 -m examples.nlu_demo`
Expected: prints parsed entities, relations, and query structures.

- [ ] **Step 3: Commit**

```bash
git add examples/nlu_demo.py
git commit -m "examples: add NLU demo (natural language → scene + query)"
```

---

### Task 11: Update Exports, Docs, and Makefile

**Files:**
- Modify: `src/telos/__init__.py`
- Modify: `Makefile`
- Modify: `docs/architecture.md`
- Modify: `README.md`

- [ ] **Step 1: Update __init__.py with new module exports**

Add to `src/telos/__init__.py`:

```python
from .structure_learner import generate_samples, learn_graph, compare_graphs
from .perception import detect_objects, extract_relations, build_world
from .nlu import parse_scene, parse_query, load_model
```

And add to `__all__`:

```python
"generate_samples", "learn_graph", "compare_graphs",
"detect_objects", "extract_relations", "build_world",
"parse_scene", "parse_query", "load_model",
```

- [ ] **Step 2: Update Makefile demo target**

```makefile
demo:
	@PYTHONPATH=src python3 -m examples.coffee_cup
	@echo
	@PYTHONPATH=src python3 -m examples.child_road
	@echo
	@PYTHONPATH=src python3 -m examples.salt_request
	@echo
	@PYTHONPATH=src python3 -m examples.novel_entity
	@echo
	@PYTHONPATH=src python3 -m examples.learned_structure
	@echo
	@PYTHONPATH=src python3 -m examples.nlu_demo
```

Note: `perception_demo` is excluded from `make demo` because it requires downloading the YOLO model on first run. It can be run separately with `PYTHONPATH=src python3 -m examples.perception_demo`.

- [ ] **Step 3: Update architecture.md**

Change the three "Not implemented" rows to:

| "Learned causal structure" | `src/telos/structure_learner.py` | Prototype: PC algorithm via causal-learn recovers DAG from observational samples. |
| "Perception from images/video" | `src/telos/perception.py` | Prototype: YOLOv8-nano detection → spatial relations → WorldState. |
| "Natural language understanding" | `src/telos/nlu.py` | Prototype: spaCy dependency parsing extracts entities, relations, and query intent. |

- [ ] **Step 4: Update README.md**

Add to the Examples section:

```markdown
- `examples/learned_structure.py` — causal discovery via PC algorithm
- `examples/perception_demo.py` — image → WorldState via YOLOv8-nano
- `examples/nlu_demo.py` — natural language → scene + query
```

Add an Install section before Quickstart:

```markdown
## Install

```bash
git clone https://github.com/Arunachalamkalimuthu/telos.git
cd telos
make install   # installs dependencies + spaCy model
```

And update Quickstart to remove the clone/cd lines (now in Install).

- [ ] **Step 5: Run all tests**

Run: `PYTHONPATH=src python3 -m pytest tests/ -v`
Expected: all tests pass (existing + new).

- [ ] **Step 6: Commit**

```bash
git add src/telos/__init__.py Makefile docs/architecture.md README.md
git commit -m "docs: update architecture, README, and exports for new modules"
```

---

### Task 12: Final Verification

- [ ] **Step 1: Run full test suite**

Run: `make test`
Expected: all tests pass.

- [ ] **Step 2: Run demo**

Run: `make demo`
Expected: all examples run without errors.

- [ ] **Step 3: Run perception demo separately**

Run: `PYTHONPATH=src python3 -m examples.perception_demo`
Expected: downloads YOLO model on first run, processes image, prints results.

- [ ] **Step 4: Final commit and push**

```bash
git push origin main
```
