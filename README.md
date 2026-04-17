# telos

**Causal Active World Architecture (CAWA)** — a cognitive architecture that reasons about the physical world through causal graphs, physics axioms, theory of mind, and active inference.

*telos* (Greek: "goal", "purpose") — named for the goal-directed action-selection at the heart of the system.

---

## What This Is

An agent that **understands why things happen**, not just what happens next.

Given a scene — a cup on a laptop, a child near a road, a dinner-table request — telos builds a causal model, simulates interventions, predicts other agents' behaviour from their beliefs (not ground truth), and selects actions that minimise expected free energy.

```
Scene → WorldState → Causal Graph → Active Inference → Action + Explanation
```

**What it is not:** This is not AGI. It is a closed-domain proof of concept demonstrating that causal reasoning, physics simulation, theory of mind, and active inference compose into a coherent cognitive architecture.

---

## Install

```bash
git clone https://github.com/Arunachalamkalimuthu/telos.git
cd telos
make install
```

Requires **Python 3.10+**. Installs `causal-learn`, `ultralytics`, `spacy`, and the `en_core_web_sm` language model.

## Run

```bash
make test       # 72 tests
make demo       # all scenarios
```

---

## The Agent Pipeline

```
perceive(WorldState)
    │
    ▼
build_causal_graph()          # physics axioms emit CausalEdges into a DAG
    │
    ▼
plan(goal, actions)           # expected free energy minimisation
    │                         # EFE = -(pragmatic + epistemic)
    ▼
explain(plan)                 # causal chain + counterfactual predictions
```

**Example:** An inverted coffee cup sits on a laptop.

```python
from telos import CAWAAgent, Entity, Relation, WorldState, Action

cup = Entity(id="cup", type="cup", properties={
    "mass": 0.25, "orientation": "inverted",
    "sealed": False, "contains": "coffee", "material": "ceramic",
})
coffee = Entity(id="coffee", type="liquid", properties={"conductive": True})
laptop = Entity(id="laptop", type="laptop", properties={"electronic": True})

world = WorldState(
    entities={"cup": cup, "coffee": coffee, "laptop": laptop},
    relations=(Relation("ON", "cup", "laptop"), Relation("WILL_CONTACT", "coffee", "laptop")),
)

agent = CAWAAgent()
agent.perceive(world)
graph = agent.build_causal_graph()

# The agent discovers: cup inverted → contents escape → liquid contacts laptop → damage
state = graph.propagate()
# {'laptop.falls': True, 'cup.contents_escape': True, 'laptop.damaged': True}

# Counterfactual: what if we sealed the cup?
graph.counterfactual({"cup.contents_escape": False})
# {'laptop.falls': True, 'cup.contents_escape': False, 'laptop.damaged': False}
```

The agent reasons through the causal chain, identifies that sealing the cup breaks the damage pathway, and selects it as the optimal intervention.

---

## Scenarios

### Core

| Scenario | What It Demonstrates |
|----------|---------------------|
| [**Coffee Cup**](examples/coffee_cup.py) | Physics primitives chain into causal graphs. Counterfactuals propagate correctly via Pearl's do-operator. Sealing the cup breaks the `spill → damage` pathway. |
| [**Child on Road**](examples/child_road.py) | Theory of mind: the agent predicts a deaf child will run into traffic because *the child believes* the road is safe. Selects physical intercept over shouting (no auditory channel). |
| [**Salt Request**](examples/salt_request.py) | Social inference: "Can you pass the salt?" is interpreted as a *request*, not a capability question, by reasoning about the asker's beliefs (arm in cast, salt out of reach). |
| [**Novel Entity**](examples/novel_entity.py) | A "frambulator" with unknown properties falls off a "zibbly". Physics applies (gravity); unknown properties are flagged with `UNKNOWN`, never hallucinated. |

### Prototypes

| Scenario | What It Demonstrates |
|----------|---------------------|
| [**Learned Structure**](examples/learned_structure.py) | PC algorithm (causal-learn) recovers the causal DAG from 1000 observational samples. Compares learned graph against hand-built ground truth with precision/recall/F1. |
| [**Perception**](examples/perception_demo.py) | YOLOv8-nano detects objects in an image, derives spatial relations (ON, NEAR, CONTAINS) from bounding box geometry, and builds a WorldState. |
| [**NLU**](examples/nlu_demo.py) | spaCy parses "A cup is on a laptop" into entities + relations, and classifies "What happens if the cup falls?" as a counterfactual query. |

```bash
# Run individually
PYTHONPATH=src python3 -m examples.coffee_cup
PYTHONPATH=src python3 -m examples.perception_demo path/to/image.jpg
```

---

## Architecture

```
src/telos/
├── world.py               # Entity, Relation, WorldState, UNKNOWN sentinel
├── physics.py             # gravity, containment, impact, liquid_damage → CausalEdges
├── causal_graph.py        # DAG + do-calculus + topological propagation + explain
├── theory_of_mind.py      # AgentMind, predict_action (from beliefs), intervention_effect
├── active_inference.py    # EFE = -(pragmatic + epistemic), action selection
├── agent.py               # CAWAAgent orchestrator
├── structure_learner.py   # PC / FCI / GES → CausalGraph (causal-learn)
├── perception.py          # YOLOv8-nano → WorldState (ultralytics)
└── nlu.py                 # text → WorldState / queries (spaCy)
```

### Design Principles

- **Immutable state.** `WorldState`, `Entity`, `Relation` are frozen dataclasses. State transitions produce new objects.
- **Composable primitives.** Physics rules are pure functions `WorldState → list[CausalEdge]`. New physics = new function, same interface.
- **Pearl's do-calculus.** Interventions sever incoming edges. Counterfactuals propagate through the modified graph.
- **Belief-based prediction.** Theory of mind predicts actions from the *agent's own beliefs*, not ground truth. A deaf child who believes the road is safe will run.
- **Expected free energy.** Actions are scored by `EFE = -(pragmatic + epistemic)`. The agent prefers actions that achieve goals *and* resolve uncertainty.
- **Honest uncertainty.** Unknown properties return `UNKNOWN`, never a guess. The system refuses to reason about what it doesn't know.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `causal-learn` | 0.1.x | PC algorithm for causal structure discovery |
| `ultralytics` | 8.x | YOLOv8-nano object detection |
| `spacy` | 3.x | Dependency parsing for NLU |

---

## Limitations

This is a reference implementation demonstrating architectural composition, not a production system.

**Core:** Causal graphs and belief states are hand-built per scene. Physics primitives are hand-coded axioms — extensible but closed-domain. No learning, no perception, no language in the core loop.

**Structure Learner:** Supports three algorithms (PC, FCI, GES) and both linear (Fisher-Z) and nonlinear (KCI) independence tests. FCI handles latent confounders. Tested on graphs up to 6 variables. Does not yet scale to hundreds of variables or learn from raw time-series data.

**Perception:** Detects objects but does not infer physics properties (mass, fragility, conductivity). Spatial relations are bounding-box heuristics. Single frame only.

**NLU:** Pattern-based extraction via dependency parsing. Handles simple spatial sentences. Does not handle negation, quantifiers, or complex clauses. Parsed entities lack physics properties.

See [`docs/architecture.md`](docs/architecture.md) for the full claim-to-code map and honest scoping.

---

## License

MIT
