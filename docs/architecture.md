# CAWA Architecture — Article to Code Map

This document maps each claim from the Medium article *Beyond Token Prediction: How LLMs, JEPA, and CAWA See the World Differently* to where that claim is realised (or honestly scoped) in this codebase.

## Article claim → code

| Article claim | Code location | Status |
|---|---|---|
| "Causal graph constructed" | `src/telos/causal_graph.py` | Implemented on hand-built graphs; automatic construction from perception is NOT implemented (open research problem). |
| "Physics primitives apply" | `src/telos/physics.py` | Implemented: `gravity`, `containment`, `impact`, `liquid_damage`. Hand-coded axioms; extensible but closed-domain. |
| "Counterfactual reasoning" | `CausalGraph.do` and `CausalGraph.counterfactual` | Implemented. Severs incoming edges to intervened variables (Pearl's do-operator). |
| "Theory of mind module" | `src/telos/theory_of_mind.py` | Implemented: `AgentMind`, `predict_action` uses agent beliefs, `intervention_effect` respects capabilities. Belief states are hand-specified, not inferred from observation. |
| "Active inference / free energy minimisation" | `src/telos/active_inference.py` | Implemented on small discrete state spaces. `EFE = -(pragmatic + epistemic)`. |
| "Honest uncertainty about unknown entities" | `src/telos/world.py` (`UNKNOWN`) and `examples/novel_entity.py` | Implemented as explicit sentinel; example demonstrates flagging rather than guessing. |
| "Learned causal structure" | `src/telos/structure_learner.py` | Prototype: PC algorithm via causal-learn recovers DAG from observational samples. |
| "Perception from images/video" | `src/telos/perception.py` | Prototype: YOLOv8-nano detection → spatial relations → WorldState. |
| "Natural language understanding" | `src/telos/nlu.py` | Prototype: spaCy dependency parsing extracts entities, relations, and query intent. |

## Module responsibilities

- **`world`** — typed entities, relations, immutable world snapshots. `UNKNOWN` sentinel for absent properties.
- **`physics`** — axiomatic primitives that emit causal edges from a world state. Primitives compose via `apply_all`.
- **`causal_graph`** — DAG over variables with do-calculus, propagation (Kahn's topological sort), and explanation.
- **`theory_of_mind`** — agents with their own beliefs, goals, capabilities, and action repertoire. `predict_action` uses beliefs, not ground truth; `intervention_effect` gates actions on perceptual channels.
- **`active_inference`** — action selection by expected free energy minimisation. `EFE = -(pragmatic + epistemic)`.
- **`agent`** — orchestrator that wires the above into `perceive → build_causal_graph → plan → explain`.
- **`structure_learner`** — (prototype) causal discovery from observational data via PC algorithm. `generate_samples` creates data from physics simulations; `learn_graph` recovers a `CausalGraph`; `compare_graphs` evaluates against ground truth.
- **`perception`** — (prototype) image → WorldState via YOLOv8-nano. `detect_objects` runs YOLO; `extract_relations` derives spatial relations from bounding box geometry; `build_world` composes both into a `WorldState`.
- **`nlu`** — (prototype) natural language → WorldState / structured queries via spaCy. `parse_scene` extracts entities and spatial relations; `parse_query` classifies questions as counterfactual or prediction.

## Example-to-architecture map

| Example | Primary architectural claim demonstrated |
|---|---|
| `examples/coffee_cup.py` | Physics primitives compose into a causal chain; counterfactuals propagate correctly (do-operator severs edges). |
| `examples/child_road.py` | Theory of mind — `predict_action` uses the child's (wrong) beliefs; `intervention_effect` correctly rules out a verbal signal for a deaf child. |
| `examples/salt_request.py` | Social inference — the utterance's meaning is grounded in the asker's belief state, not surface form. |
| `examples/novel_entity.py` | Physics applies regardless of entity identity; unknown properties are flagged, not hallucinated. |
| `examples/learned_structure.py` | PC algorithm recovers causal edges from observational samples; comparison against hand-built graph shows precision/recall. |
| `examples/perception_demo.py` | YOLOv8-nano detects objects in an image; spatial relations derived from bounding box geometry; result is a standard WorldState. |
| `examples/nlu_demo.py` | spaCy dependency parsing converts natural language scene descriptions into WorldState and classifies questions as counterfactual or prediction. |

## Trade-offs

- **Pure symbolic, stdlib-only.** Zero setup; no learning, no perception.
- **Hand-coded primitives.** Does not scale to open-world; that is the Cyc lesson. We accept it because the goal is to demonstrate the architecture on a closed domain.
- **Hand-built causal graphs.** We do not claim to solve causal representation learning.
- **Hand-specified belief states.** We do not claim to solve belief-state inference from observed behaviour.

## What this demo does NOT show

- It does not show CAWA can replace LLMs or JEPA.
- It does not show CAWA scales to real-world perception or open vocabulary.
- It does not benchmark anything.

See `docs/superpowers/specs/2026-04-15-cawa-reference-implementation-design.md` §9 for the full honest accounting.
