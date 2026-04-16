# Remaining Features — Learned Structure, Perception, NLU

**Date:** 2026-04-16
**Status:** Approved

## Goal

Implement prototypes for the three features flagged as "not implemented" in `docs/architecture.md`:

1. Learned causal structure (automatic graph construction from data)
2. Perception from images/video (visual input → WorldState)
3. Natural language understanding (text → WorldState / queries)

Each prototype uses local, lightweight libraries. No cloud APIs.

## Constraints

- Each feature is a new module under `src/telos/`
- Each module integrates with existing types (`WorldState`, `CausalGraph`, etc.)
- Each has a corresponding example and test file
- New dependencies: `causal-learn`, `ultralytics`, `spacy`

---

## 1. Structure Learner — `src/telos/structure_learner.py`

**Library:** `causal-learn` (CMU causal discovery, PC algorithm)

**Functions:**

- `generate_samples(world, physics_primitives, n=500)` — perturb entity properties in a `WorldState`, run physics primitives to collect observational data, return as a 2D array (rows = samples, columns = variables).
- `learn_graph(samples, variable_names)` — run the PC algorithm via `causal-learn`, convert the resulting skeleton + orientations into a telos `CausalGraph`.
- `compare_graphs(learned, handbuilt)` — compute edge precision, recall, and F1 between two `CausalGraph` instances. Return a dict with metrics.

**Integration point:** Output is a standard `CausalGraph` — all existing downstream code (do-calculus, propagation, explanation) works unchanged.

**Example:** `examples/learned_structure.py`
- Generate samples from the coffee cup physics scenario
- Learn the causal graph via PC algorithm
- Compare learned graph against the hand-built graph from `examples/coffee_cup.py`
- Print precision/recall/F1

**Test:** `tests/test_structure_learner.py`
- Test sample generation produces expected shape
- Test learned graph recovers known edges from a simple deterministic scenario
- Test compare_graphs metrics on known identical and different graphs

---

## 2. Perception — `src/telos/perception.py`

**Library:** `ultralytics` (YOLOv8-nano, ~6MB model)

**Functions:**

- `detect_objects(image_path)` — run YOLOv8-nano on an image, return a list of dicts: `{"label": str, "confidence": float, "bbox": (x1, y1, x2, y2)}`.
- `extract_relations(detections)` — derive spatial relations from bounding box geometry:
  - `on(A, B)` — A's bottom edge is near B's top edge and A is horizontally within B
  - `above(A, B)` — A's center is above B's center with no overlap
  - `near(A, B)` — bounding boxes are within a threshold distance
  - `contains(A, B)` — B's bbox is mostly inside A's bbox
- `build_world(image_path)` — end-to-end pipeline: detect objects → extract relations → construct a `WorldState`.

**Integration point:** Output is a standard `WorldState` — can be fed directly into physics primitives, causal graph construction, or the agent orchestrator.

**Example:** `examples/perception_demo.py`
- Load a sample image (a cup on a table, or similar)
- Run `build_world` to get a `WorldState`
- Apply physics primitives and build causal graph
- Print the world state and causal chain

**Test:** `tests/test_perception.py`
- Test `detect_objects` returns expected structure (mock YOLO results for determinism)
- Test `extract_relations` correctly derives `on`, `above`, `near`, `contains` from known bounding boxes
- Test `build_world` end-to-end with mocked detections

**Note:** A sample image will be included in `examples/assets/` for the demo. Tests mock the YOLO model for determinism and CI-friendliness.

---

## 3. NLU — `src/telos/nlu.py`

**Library:** `spacy` with `en_core_web_sm` model (~15MB)

**Functions:**

- `parse_scene(text)` — parse a natural language scene description into a `WorldState`.
  - Input: "A coffee cup is on the edge of a wooden table"
  - Output: `WorldState` with entities (cup, table) and relations (on, edge_position)
  - Uses spaCy dependency parsing + `DependencyMatcher` patterns to extract subject-relation-object triples.
- `parse_query(text)` — parse a natural language question into a structured query.
  - Input: "What happens if the cup falls?"
  - Output: dict `{"type": "counterfactual", "intervention": {"cup_position": "falling"}, "target": None}` (None = explain full chain)
  - Input: "Will the laptop get damaged?"
  - Output: dict `{"type": "prediction", "target": "laptop_damaged"}`
- `load_model()` — load spaCy model, called once. Returns the nlp pipeline.

**Pattern categories:**
- Spatial patterns: "X is on/near/above/inside Y"
- Property patterns: "X is [adjective]" → entity properties
- Counterfactual patterns: "what happens if X", "what if X"
- Prediction patterns: "will X", "is X going to"

**Integration point:** `parse_scene` output is a `WorldState`; `parse_query` output drives `CausalGraph.do` or `CausalGraph.counterfactual`.

**Example:** `examples/nlu_demo.py`
- Describe the coffee cup scenario in plain English
- Parse into a WorldState
- Ask a counterfactual question in English
- Run causal reasoning and print explanation

**Test:** `tests/test_nlu.py`
- Test `parse_scene` extracts correct entities and relations from known sentences
- Test `parse_query` identifies counterfactual vs prediction queries
- Test integration: parsed scene → physics → causal graph → query → explanation

---

## Dependencies

Add to `pyproject.toml`:

```toml
dependencies = [
    "causal-learn",
    "ultralytics",
    "spacy",
]
```

Post-install step: `python -m spacy download en_core_web_sm`

Add to `Makefile`: an `install` target that runs `pip install -e .` and downloads the spaCy model.

---

## Files Changed/Created

**New files:**
- `src/telos/structure_learner.py`
- `src/telos/perception.py`
- `src/telos/nlu.py`
- `examples/learned_structure.py`
- `examples/perception_demo.py`
- `examples/nlu_demo.py`
- `examples/assets/` (sample image for perception demo)
- `tests/test_structure_learner.py`
- `tests/test_perception.py`
- `tests/test_nlu.py`

**Modified files:**
- `pyproject.toml` — add dependencies
- `Makefile` — add `install` target, update `demo` target to include new examples
- `docs/architecture.md` — update status of the three features from "Not implemented" to "Prototype implemented"
- `README.md` — add new examples and install instructions
