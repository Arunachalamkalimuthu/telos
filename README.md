# telos

A Python reference implementation of the **Causal Active World Architecture (CAWA)** described in the Medium article *[Beyond Token Prediction: How LLMs, JEPA, and CAWA See the World Differently](https://medium.com/@arunachalamk)*.

The name — Greek for "goal" or "purpose" — reflects the goal-directed action-selection at the heart of active inference.

**This is not AGI.** It is a closed-domain proof that causal graphs, physics primitives, theory of mind, and active inference compose cleanly into a working agent — plus prototypes for learned structure, perception, and natural language understanding. See [`docs/architecture.md`](docs/architecture.md) for an honest accounting of what this does and does not demonstrate.

## Install

```bash
git clone https://github.com/Arunachalamkalimuthu/telos.git
cd telos
make install   # installs dependencies + spaCy model
```

Requires **Python 3.10+**.

## Quickstart

```bash
make test      # run all 72 tests
make demo      # run all 6 example scenarios
```

## How It Works

```
perceive(WorldState)
    → build_causal_graph (physics primitives → CausalEdge DAG)
    → plan (active inference: minimise expected free energy)
    → explain (causal chain + counterfactuals)
```

The agent receives a `WorldState` (entities + relations), applies physics primitives to construct a causal graph, selects actions via expected free energy minimisation, and explains its reasoning through causal chains and counterfactual predictions.

## Examples

### Core Scenarios

| Example | Demonstrates |
|---------|-------------|
| [`coffee_cup.py`](examples/coffee_cup.py) | Physics primitives compose into a causal chain; counterfactuals propagate via do-operator |
| [`child_road.py`](examples/child_road.py) | Theory of mind — predicts a deaf child's action from their (wrong) beliefs; selects physical intercept over verbal signal |
| [`salt_request.py`](examples/salt_request.py) | Social inference — "Can you pass the salt?" interpreted as request, not capability question, based on asker's belief state |
| [`novel_entity.py`](examples/novel_entity.py) | Physics applies to unknown entities; unknown properties are flagged, not hallucinated |

### Prototype Scenarios

| Example | Demonstrates |
|---------|-------------|
| [`learned_structure.py`](examples/learned_structure.py) | PC algorithm recovers causal edges from observational data; precision/recall against hand-built graph |
| [`perception_demo.py`](examples/perception_demo.py) | YOLOv8-nano detects objects in an image → spatial relations → WorldState |
| [`nlu_demo.py`](examples/nlu_demo.py) | Natural language scene descriptions and questions parsed into WorldState and structured queries |

Run any example individually:

```bash
PYTHONPATH=src python3 -m examples.coffee_cup
PYTHONPATH=src python3 -m examples.perception_demo path/to/image.jpg
```

## Architecture

### Core Modules

| Module | Responsibility |
|--------|---------------|
| [`world.py`](src/telos/world.py) | Typed entities, relations, immutable snapshots; `UNKNOWN` sentinel for absent properties |
| [`physics.py`](src/telos/physics.py) | Axiomatic primitives (`gravity`, `containment`, `impact`, `liquid_damage`) that emit causal edges |
| [`causal_graph.py`](src/telos/causal_graph.py) | DAG with do-calculus (Pearl's do-operator), topological propagation, and causal explanation |
| [`theory_of_mind.py`](src/telos/theory_of_mind.py) | Agents with beliefs, goals, capabilities; action prediction from beliefs, not ground truth |
| [`active_inference.py`](src/telos/active_inference.py) | Action selection by expected free energy minimisation: `EFE = -(pragmatic + epistemic)` |
| [`agent.py`](src/telos/agent.py) | Orchestrator: `perceive → build_causal_graph → plan → explain` |

### Prototype Modules

| Module | Library | Responsibility |
|--------|---------|---------------|
| [`structure_learner.py`](src/telos/structure_learner.py) | `causal-learn` | Causal discovery from observational data via PC algorithm |
| [`perception.py`](src/telos/perception.py) | `ultralytics` | Image → object detection (YOLOv8-nano) → spatial relations → WorldState |
| [`nlu.py`](src/telos/nlu.py) | `spaCy` | Natural language → WorldState / structured queries via dependency parsing |

## Project Structure

```
src/telos/            # library code
examples/             # runnable scenarios
tests/                # 72 unit tests
docs/architecture.md  # article-to-code map + honest scoping
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `causal-learn` | PC algorithm for causal structure discovery |
| `ultralytics` | YOLOv8-nano object detection |
| `spacy` + `en_core_web_sm` | Dependency parsing for NLU |

All installed via `make install`.

## Limitations

This is a reference implementation, not a production system.

- **Core modules** are pure symbolic, hand-coded, closed-domain. Causal graphs and belief states are hand-built per scene.
- **Structure learner** works on small variable sets with synthetic linear data. No hidden variable support.
- **Perception** detects objects but doesn't infer physics properties (mass, fragility). Single-frame only, no video.
- **NLU** is pattern-based. Misses complex sentences, negation, and doesn't map parsed entities to physics properties automatically.
- No end-to-end pipeline connecting perception → NLU → causal reasoning.

See [`docs/architecture.md`](docs/architecture.md) for the full article-to-code map and what is explicitly out of scope.

## License

MIT
