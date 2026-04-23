# Telos Architecture

Telos has two layers:

1. **Product layers (Phases 1-4)** — a causal reasoning and memory system for LLM coding assistants. Shipped and working, exposed via CLI and MCP.
2. **Research core (CAWA)** — the Causal Active World Architecture reference implementation. Demonstrates how causal graphs, physics primitives, theory of mind, and active inference compose.

The product layers build on primitives from the research core (`causal_graph.py` do-calculus, `theory_of_mind.py` AgentMind, `active_inference.py` EFE scoring).

---

## Product Architecture

### Phase 1: Code Graph + Impact Analysis

| Component | Location | Responsibility |
|-----------|----------|---------------|
| Parser orchestrator | `src/telos/code_parser/parser.py` | tree-sitter multi-language dispatch |
| Language extractors | `src/telos/code_parser/languages/*.py` | Python, JavaScript, TypeScript, Go, Java, Rust |
| Graph builder | `src/telos/code_parser/graph_builder.py` | AST → SQLite nodes + edges |
| Graph store | `src/telos/code_parser/store.py` | SQLite persistence at `.telos/graph.db` |
| Impact analyzer | `src/telos/impact/analyzer.py` | Transitive traversal with risk scoring |
| Counterfactual engine | `src/telos/impact/counterfactual.py` | Pearl's do-operator on the code graph |
| Reporter | `src/telos/impact/reporter.py` | Rich terminal output (trees, tables) |

**Edge weights:** `CALLS=1.0`, `INHERITS=0.9`, `DATA_FLOW=0.8`, `IMPORTS=0.6`. Risk along a path = product of edge weights.

### Phase 2: CLI + MCP Server

| Component | Location | Responsibility |
|-----------|----------|---------------|
| CLI | `src/telos/cli.py` | `telos init`, `impact`, `counterfactual`, `hotspots`, `info` |
| MCP server | `src/telos/mcp_server.py` | 19 tools exposed to any LLM via MCP protocol |

### Phase 3: Memory Layer

| Component | Location | Responsibility |
|-----------|----------|---------------|
| Event graph | `src/telos/memory/event_graph.py` | SQLite-backed causal graph over events (sessions, decisions, changes, outcomes) |
| Project memory | `src/telos/memory/project_memory.py` | Session management + structured event recording with auto-linking |
| Cross-session learner | `src/telos/memory/cross_session_learner.py` | Pattern detection across sessions (most-changed, failure-prone, co-changes) |

### Phase 4: Learning from History

| Component | Location | Responsibility |
|-----------|----------|---------------|
| Git learner | `src/telos/history/git_learner.py` | Parses `git log` for co-changes, churn, bug-prone files, coupling |
| Developer model | `src/telos/history/developer_model.py` | Builds `AgentMind` profiles from commit history; suggests reviewers |
| Fix evaluator | `src/telos/history/fix_evaluator.py` | Ranks candidate fixes by combining counterfactual + historical + developer evidence |

---

## MCP Tool Surface

19 tools across 4 groups:

| Group | Tools |
|-------|-------|
| **Impact Analysis (Phase 1-2)** | `telos_init`, `telos_impact`, `telos_counterfactual`, `telos_hotspots`, `telos_info` |
| **Memory (Phase 3)** | `telos_memory_start_session`, `telos_memory_record_decision`, `telos_memory_record_change`, `telos_memory_record_outcome`, `telos_memory_why`, `telos_memory_what_happened`, `telos_memory_patterns`, `telos_memory_search`, `telos_memory_recent` |
| **Git History (Phase 4)** | `telos_history_patterns`, `telos_history_bug_prone`, `telos_developer_profile`, `telos_developer_risk`, `telos_suggest_reviewers` |

Each tool returns JSON for the LLM to parse.

---

## Data Flow

```
Developer question ──▶ LLM ──▶ MCP tool call ──▶ Telos tool
                                                      │
                           ┌──────────────────────────┼──────────────────┐
                           ▼                          ▼                  ▼
                  Code graph (.telos/graph.db)  Memory (memory.db)  Git (subprocess)
                           │                          │                  │
                           └──────────────┬───────────┴──────────────────┘
                                          ▼
                          Structured JSON result ──▶ LLM ──▶ Developer
```

- Impact tools read code graph
- Memory tools read/write event graph
- History tools shell out to git (read-only)
- All results are JSON — LLM composes them into natural-language answers

---

## Research Core (CAWA)

The original reference implementation demonstrates the Causal Active World Architecture.

## Claim → code

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
| `examples/telos_product_demo.py` | End-to-end product demo spanning all 4 phases: impact analysis, counterfactuals, memory recording + causal chain traversal, git history patterns, developer expertise. |

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
