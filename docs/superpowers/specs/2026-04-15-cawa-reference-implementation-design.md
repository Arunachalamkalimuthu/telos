# CAWA Reference Implementation — Design

**Date:** 2026-04-15
**Status:** Approved for implementation planning
**Author:** Arunachalam K. (via Claude Code brainstorming session)

---

## 1. Motivation

The Causal Active World Architecture (CAWA) — combining causal graphs, physics primitives, theory of mind, and active inference — answers a fundamentally different question from LLMs or JEPA. CAWA does not exist as a complete system and is a long-horizon research programme.

This project builds a **reference implementation** that makes the architectural argument concrete and runnable. It is not an attempt to replace LLMs or JEPA. It is a proof that the four architectural ideas compose cleanly into a working agent on a closed domain, with honest accounting of what the approach does and does not cover.

## 2. Scope

### In scope

- A Python package (`src/cawa/`) implementing six modules: world, physics, causal graph, theory of mind, active inference, agent.
- Four runnable example scripts demonstrating the architecture on the three scenarios from the article (coffee cup, child near road, salt request) plus a novel-entity test (the "frambulator" case).
- Unit tests per module and end-to-end tests per example.
- Developer tooling: `pyproject.toml`, `Makefile`, `README.md`, architecture documentation.
- Zero runtime dependencies outside the Python standard library.

### Non-goals

- No learning from data (no training, no gradient descent, no neural networks).
- No perception from images, video, audio, or raw sensors.
- No natural-language understanding. Scenes are specified in Python.
- No claim of generality. The system works on the closed set of entities, properties, and physics primitives it is given.
- No attempt to scale causal-graph construction, theory-of-mind inference, or active inference to open-world state spaces. Those are open research problems (see §9).

## 3. Module architecture

Six modules, each with one clear purpose and a narrow interface.

### 3.1 `world`

Represents entities and their relationships at a point in time.

- `Entity(id, type, properties: dict[str, Any])` — typed properties include `mass`, `position`, `orientation`, `fragility`, `sealed`, `material`, `contains`, and domain extensions. Unknown properties are the sentinel value `UNKNOWN`, never guessed.
- `Relation(name, src, dst, attributes)` — e.g. `ON(cup, table)`, `NEAR(glass, edge)`.
- `WorldState(entities, relations, time)` — a snapshot. Immutable; interventions produce new `WorldState` values.

### 3.2 `physics`

Physics primitives as axiomatic pure functions. Each primitive has signature:

```python
def primitive(world: WorldState) -> list[CausalEdge]
```

Initial set of primitives:

- `gravity` — unsupported objects with mass fall.
- `containment` — contents of an inverted unsealed container escape.
- `impact` — a fragile object impacting at velocity above its threshold breaks.
- `support` — objects on a surface are supported iff their centre of mass is over the surface.
- `trajectory` — position evolves with velocity under applicable forces.

Primitives are composable: the causal graph for a scene is the union of edges emitted by all applicable primitives. Adding a new primitive is a ~20-line function; the architecture does not change.

### 3.3 `causal_graph`

A DAG over state variables, with do-calculus support.

- `CausalGraph` — nodes are variables, edges carry causal mechanisms (callables over parent values).
- `propagate()` — topological sort, evaluate each mechanism, return the resulting state.
- `do(var, value)` — returns a new graph with the variable pinned and its incoming edges severed (the `do`-operator).
- `counterfactual(interventions: dict)` — apply multiple `do` operations and propagate.
- `explain_path(target)` — return the causal chain from root causes to a target node, as a list of edges with their mechanisms.

### 3.4 `theory_of_mind`

Models other agents as having their own beliefs, goals, and perceptual capabilities — distinct from ground truth.

- `AgentMind(beliefs: dict, goals: list, capabilities: set, actions: list)`.
- `beliefs` is a partial view of the world (may be wrong or incomplete).
- `capabilities` includes perceptual channels (`visual`, `auditory`, `tactile`) and motor affordances.
- `predict_action(world)` — returns the action the agent would take, computed over *their beliefs*, not ground truth. This is what distinguishes ToM from behaviourist prediction.
- `intervention_effect(my_action, their_mind)` — whether an action by me can reach the agent. A verbal signal to a deaf agent returns no effect. A visual signal outside their field of view returns no effect.

### 3.5 `active_inference`

Action selection by expected free energy minimisation.

- Given a `WorldState`, a `goal` (desired partial world state), and a candidate action set, for each candidate:
  1. Simulate the resulting world via the causal graph.
  2. Compute **pragmatic value** — negative distance from the resulting state to the goal.
  3. Compute **epistemic value** — expected reduction in entropy over uncertain variables.
  4. `EFE(a) = -(pragmatic(a) + epistemic(a))`. (Sign convention: higher `pragmatic + epistemic` means lower expected free energy, i.e. more preferred.)
- Return `argmin EFE(a)` along with the causal chain used to evaluate it.
- Output includes the counterfactual rationale: "I chose A over B because under A the chain terminates at goal; under B it terminates at cost."

### 3.6 `agent`

Ties the modules together.

- `CAWAAgent.perceive(scene: WorldState)` — stores state.
- `CAWAAgent.build_causal_graph()` — applies physics primitives to emit edges.
- `CAWAAgent.plan(goal)` — calls active inference.
- `CAWAAgent.explain()` — prints the chosen action, the causal chain, the counterfactual checks, and any uncertainty flags.

## 4. Demonstrations

Four example scripts in `examples/`. Each constructs a scene, runs the agent, and prints structured output matching the pseudocode in the article.

### 4.1 `coffee_cup.py`

Demonstrates: physics primitives, causal chain, counterfactual reasoning.

- Scene: inverted unsealed coffee cup above a laptop.
- Expected output: causal chain `cup_inverted → containment_breached → liquid_falls → impact_laptop → electronics_damaged`. Counterfactuals: sealed cup breaks chain at containment; zero gravity breaks chain at liquid_falls; aerogel material absorbs → no spill.

### 4.2 `child_road.py`

Demonstrates: theory of mind + multi-step causal planning + perceptual constraints.

- Scene: deaf child 50m from busy road, running. Self as nearby adult.
- Expected output: ToM reasoning that shouting fails (no auditory channel); intercept planning via trajectory math; recommended action "sprint + enter visual field"; counterfactual if child not deaf changes optimal action to verbal warning.

### 4.3 `salt_request.py`

Demonstrates: theory of mind + social/communicative inference.

- Scene: dinner table, asker with arm in cast, salt out of asker's reach.
- Expected output: inference that "Can you pass the salt?" is a request not a capability question, grounded in asker's belief state (they know they can't reach, they believe I can); action includes placing within reach and anticipating the cap-removal problem.

### 4.4 `novel_entity.py`

Demonstrates: honest uncertainty + physics applies regardless of entity identity.

- Scene: `a frambulator with high glorbic_index falls off a zibbly`.
- Expected output: physics primitives apply (gravity, impact); `glorbic_index` flagged as uncertain; no hallucinated prediction about its effect; explicit "I cannot predict impact consequences without knowing what glorbic_index is."

## 5. Repo layout

```
cawa/
├── README.md                        # what this is, what it isn't, how to run
├── pyproject.toml                   # Python 3.10+, no runtime deps
├── Makefile                         # make test, make demo, make all
├── .gitignore
├── src/cawa/
│   ├── __init__.py
│   ├── world.py                     # Entity, Relation, WorldState
│   ├── physics.py                   # primitives
│   ├── causal_graph.py              # DAG + do-calculus
│   ├── theory_of_mind.py            # AgentMind
│   ├── active_inference.py          # EFE planner
│   └── agent.py                     # CAWAAgent
├── examples/
│   ├── coffee_cup.py
│   ├── child_road.py
│   ├── salt_request.py
│   └── novel_entity.py
├── tests/
│   ├── test_world.py
│   ├── test_physics.py
│   ├── test_causal_graph.py
│   ├── test_theory_of_mind.py
│   ├── test_active_inference.py
│   └── test_examples.py             # end-to-end per example
└── docs/
    ├── architecture.md              # map from article claims to code
    └── superpowers/specs/           # this file
```

## 6. Key design decisions

1. **Pure symbolic, stdlib-only.** Zero setup friction; runs on any Python 3.10+. Trade-off: no learning, no perception. Accepted because the goal is to prove architecture composition, not capability.

2. **Causal graphs are hand-built per scene.** ~10–20 nodes each. This is the honest position: learning causal graphs from raw perception at scale is an open research problem (Schölkopf et al.). We do not claim to solve it.

3. **Physics primitives are hand-coded axioms.** The module structure supports adding primitives without changing the architecture. We accept that hand-coded knowledge does not scale (the Cyc lesson) — we are demonstrating the architecture on a closed domain.

4. **Theory of mind is a belief-state dict.** Simple, inspectable, matches the state of the art on toy domains. No Bayesian inference over belief hierarchies — that can be added later.

5. **Active inference uses a simple expected free energy decomposition.** `EFE = -(pragmatic + epistemic)`. Pragmatic is goal distance; epistemic is entropy reduction. This is faithful to Friston's framework at toy scale.

6. **All outputs are explanatory.** The agent prints the causal chain, not just the action. The architecture is debuggable because causality is explicit.

7. **Immutable world states.** Interventions produce new `WorldState` values rather than mutating. This makes counterfactual reasoning and test assertions straightforward.

## 7. Testing strategy

- **Unit tests** per module:
  - `test_world.py` — entity construction, property access, UNKNOWN handling.
  - `test_physics.py` — each primitive emits correct edges on known inputs.
  - `test_causal_graph.py` — propagation, `do()` correctly severs incoming edges, counterfactual queries, `explain_path()`.
  - `test_theory_of_mind.py` — `predict_action` uses agent beliefs not ground truth; `intervention_effect` respects capabilities (deaf agent + shout = no effect).
  - `test_active_inference.py` — EFE decomposition, action selection on small problems with known answers.

- **End-to-end tests** per example in `test_examples.py`: asserts the causal chain contains expected edges and the chosen action matches the article's pseudocode.

- **`make test`** runs everything. Exit code 0 on green.

## 8. Build and run

```
make test          # run all tests
make demo          # run all four examples, print output
python -m examples.coffee_cup   # run one example
```

README includes a copy-paste-runnable quickstart.

## 9. Honest accounting: what this does NOT demonstrate

This section exists because the article makes strong claims and the code must not be mistaken for evidence of those claims beyond what it actually shows.

- **It does not show that CAWA scales.** The causal graphs are small and hand-built. The open problem — learning them from perception — is not addressed.
- **It does not show CAWA is better than LLMs or JEPA on any benchmark.** We are not comparing models; we are demonstrating architectural principles.
- **It does not show theory of mind works in the wild.** The belief-state representation is hand-specified. Inferring it from observation is an open problem.
- **It does not show active inference scales.** The state spaces in examples have <100 discrete states.
- **Novel-entity handling is shallow.** The `frambulator` test shows uncertainty is flagged honestly when an unknown property appears, but this is not a solution to open-vocabulary perception.

The contribution of this repo is: **a clean, runnable integration of the four ingredients on a closed domain, with explicit causal explanations and honest uncertainty.** That is useful as a teaching artefact, a research scaffold, and a counter to hand-wavy CAWA claims — but it is not CAWA-the-system the article imagines.

## 10. Out of scope for v1 (possible future work)

- LLM-grounded perception (parse a scene description into a `WorldState` via Claude/GPT).
- Learned physics primitives (the V-JEPA direction).
- Belief-state inference from observed actions (Bayesian ToM).
- Scaling active inference via amortised variational methods.
- A visual/interactive demo (gridworld with rendering).
