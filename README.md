# telos

A Python reference implementation of the Causal Active World Architecture (CAWA) described in the Medium article *Beyond Token Prediction: How LLMs, JEPA, and CAWA See the World Differently*.

The name — Greek for "goal" or "purpose" — reflects the goal-directed action-selection at the heart of active inference.

**This is not AGI.** This is a closed-domain proof that causal graphs, physics primitives, theory of mind, and active inference compose cleanly into a working agent. See `docs/architecture.md` for an honest accounting of what this does and does not demonstrate.

## Quickstart

```bash
git clone https://github.com/Arunachalamkalimuthu/telos.git
cd telos
make test      # run all tests
make demo      # run all four example scenarios
```

## Examples

- `examples/coffee_cup.py` — physics + counterfactuals
- `examples/child_road.py` — theory of mind + intervention planning
- `examples/salt_request.py` — theory of mind + social inference
- `examples/novel_entity.py` — honest uncertainty with unknown entities

## Modules

- `src/telos/world.py` — typed entities, relations, immutable snapshots; `UNKNOWN` sentinel for absent properties
- `src/telos/physics.py` — axiomatic primitives (`gravity`, `containment`, `impact`, `liquid_damage`) that emit causal edges
- `src/telos/causal_graph.py` — DAG with do-calculus, propagation, and explanation
- `src/telos/theory_of_mind.py` — agents with beliefs, goals, capabilities; action prediction from beliefs, not ground truth
- `src/telos/active_inference.py` — action selection by expected free energy minimisation
- `src/telos/agent.py` — orchestrator: `perceive → build_causal_graph → plan → explain`

See `docs/architecture.md` for the article-to-code map and honest scoping of what is and is not implemented.

## Requirements

Python 3.10+. No runtime dependencies.
