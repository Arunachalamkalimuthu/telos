# telos

**The reasoning and memory layer for AI coding assistants.**

Telos makes any LLM (Claude Code, Cursor, Copilot) provably aware of what its code changes will break, why past decisions were made, and who knows each part of the codebase — by building a causal graph over your code, a persistent event graph over your decisions, and a history graph from your git log.

> *telos* (Greek: "goal", "purpose") — named for the goal-directed action-selection at the heart of the system.

---

## What It Solves

LLMs write plausible code. Telos tells you what that code will actually do.

| Problem | How Telos Solves It |
|---------|---------------------|
| "Is this change safe?" | Causal graph — propagates impact through every dependent function |
| "Why did we make that decision?" | Event graph — traces causal chain back to the root reason |
| "Which fix is best?" | Counterfactual analysis — simulates each option's blast radius |
| "Who should review this?" | Developer model — finds contributors with expertise in the file |
| "What tends to break in this area?" | Git history — learns co-changes and bug-prone files |
| "I don't know that property" | `UNKNOWN` sentinel — refuses to hallucinate |

Works on **6 languages**: Python, JavaScript, TypeScript, Go, Java, Rust.

---

## Install

```bash
git clone https://github.com/Arunachalamkalimuthu/telos.git
cd telos
make install
```

Requires **Python 3.10+**. Installs all dependencies plus the spaCy `en_core_web_sm` language model.

---

## Quick Start

### CLI

```bash
telos init                                           # scan repo, build graph
telos impact src/auth.py:validate_token             # trace impact
telos impact src/auth.py:validate_token --fix X     # counterfactual
telos hotspots                                      # most depended-on code
telos info                                          # graph stats
```

### MCP Server (LLM integration)

Add to your Claude Code settings (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "telos": {
      "command": "python3",
      "args": ["-m", "telos.mcp_server"],
      "cwd": "/path/to/your/repo"
    }
  }
}
```

Claude, Cursor, and other MCP-aware LLMs can now ask telos directly:
- *"What breaks if I change `validate_token`?"*
- *"Why did we cap retries at 2?"* (traces back through session memory)
- *"Who should review changes to `src/payment/`?"*
- *"What files tend to break together?"*

---

## The Four Layers

```
┌────────────────────────────────────────────────────────────┐
│                     LLM Interface (Any LLM)                 │
│          MCP server exposes 19 tools for reasoning           │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
┌──────────────▼──────────────┐  ┌───────────▼───────────────┐
│   Phase 1: Code Graph        │  │   Phase 3: Memory          │
│                              │  │                            │
│  • tree-sitter parser         │  │  • event graph             │
│    (6 languages)              │  │    (sessions, decisions,   │
│  • SQLite dependency graph    │  │     changes, outcomes)     │
│  • Impact analyzer            │  │  • project memory          │
│    (transitive + risk)        │  │    ("why did we...?")      │
│  • Counterfactual engine      │  │  • cross-session learner   │
│    (Pearl's do-operator)      │  │    (failure patterns)      │
└──────────────────────────────┘  └────────────────────────────┘
               │                              │
┌──────────────▼──────────────────────────────▼───────────────┐
│              Phase 4: Learning from History                  │
│                                                              │
│  • git learner — co-changes, churn, bug-prone files         │
│  • developer model — AgentMind profiles from commits         │
│  • fix evaluator — rank fixes by historical + causal evidence│
└──────────────────────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│              Reference Core (Research Foundation)            │
│                                                              │
│  Causal graphs • Physics primitives • Theory of mind •      │
│  Active inference • Honest uncertainty                       │
└──────────────────────────────────────────────────────────────┘
```

---

## MCP Tools

### Impact Analysis (Phase 1-2)

| Tool | Description |
|------|-------------|
| `telos_init` | Scan codebase, build dependency graph |
| `telos_impact` | Trace full transitive impact of a change |
| `telos_counterfactual` | Compare blast radius with/without a fix |
| `telos_hotspots` | Most depended-on code (highest risk) |
| `telos_info` | Graph statistics and metadata |

### Memory (Phase 3)

| Tool | Description |
|------|-------------|
| `telos_memory_start_session` | Begin tracking a work session |
| `telos_memory_record_decision` | Capture WHY a decision was made |
| `telos_memory_record_change` | Track what changed and where |
| `telos_memory_record_outcome` | Record success/failure (auto-links to cause) |
| `telos_memory_why` | Root cause analysis — trace back through chain |
| `telos_memory_what_happened` | Full history for any file or function |
| `telos_memory_patterns` | Cross-session insights across all sessions |
| `telos_memory_search` | Keyword search across event summaries |
| `telos_memory_recent` | Most recent events |

### Git History (Phase 4)

| Tool | Description |
|------|-------------|
| `telos_history_patterns` | Co-changes, bug-prone files, recent hotspots |
| `telos_history_bug_prone` | Files with highest historical bug rate |
| `telos_developer_profile` | Expertise and activity for any contributor |
| `telos_developer_risk` | Is this developer qualified to change this file? |
| `telos_suggest_reviewers` | Best reviewers based on git expertise |

---

## Example: CLI Impact Analysis

```bash
$ telos impact src/telos/causal_graph.py:CausalGraph

Impact Analysis: src/telos/causal_graph.py:CausalGraph
══════════════════════════════════════════════════════

  src/telos/causal_graph.py:CausalGraph
  ├── CALLS → src/telos/agent.py:build_causal_graph     [risk: 1.0]
  │   └── CALLS → src/telos/agent.py:plan                [risk: 1.0]
  ├── CALLS → src/telos/structure_learner.py:learn_graph [risk: 1.0]
  └── IMPORTS → tests/test_causal_graph.py               [risk: 0.6]

Hottest path: CausalGraph → build_causal_graph → plan
Files affected: 8
```

## Example: Counterfactual

```bash
$ telos impact validate_token --fix add_fallback_at:middleware.py:require_auth

Without fix:  23 functions across 8 files affected
With fix:      2 functions across 1 file affected
Blast radius reduced: 91%
```

## Example: LLM + Memory

Developer asks Claude Code: *"Why did we cap retries at 2?"*

Claude calls `telos_memory_search("retry")` → finds decision event → calls `telos_memory_why(event_id)` → traces back:

```
Root cause (session_3, 2026-04-15):
  decision: Use retry pattern for API calls
  ↓ led_to
  change: Added retry with max_retries=5
  ↓ caused
  outcome: Connection pool exhausted in 12 seconds
  ↓ led_to
  decision: Cap retries at 2 (root of query)
```

---

## Reference Implementation

Telos grew out of research into the **Causal Active World Architecture (CAWA)** — causal graphs + physics primitives + theory of mind + active inference. The original reference implementation demonstrates these principles on closed-domain scenarios:

```bash
make test       # 227+ tests
make demo       # scenario demos
```

### Scenarios

| Scenario | What It Demonstrates |
|----------|---------------------|
| [**Coffee Cup**](examples/coffee_cup.py) | Physics + counterfactuals via Pearl's do-operator |
| [**Child on Road**](examples/child_road.py) | Theory of mind — predicts from beliefs, not ground truth |
| [**Salt Request**](examples/salt_request.py) | Social inference from belief state |
| [**Novel Entity**](examples/novel_entity.py) | Honest uncertainty — `UNKNOWN`, not hallucination |
| [**Learned Structure**](examples/learned_structure.py) | PC/FCI/GES algorithms recover causal DAG from data |
| [**Perception**](examples/perception_demo.py) | YOLOv8-nano → WorldState with physics property KB |
| [**NLU**](examples/nlu_demo.py) | spaCy dependency parsing → executable causal queries |

See [`docs/architecture.md`](docs/architecture.md) for the full claim-to-code map.

---

## Architecture

```
src/telos/
│
│  # Core reasoning (reference implementation)
├── world.py                  # Entity, Relation, WorldState, UNKNOWN
├── physics.py                # gravity, containment, impact, liquid_damage
├── causal_graph.py           # DAG + do-calculus + propagation
├── theory_of_mind.py         # AgentMind, belief-based prediction
├── active_inference.py       # EFE = -(pragmatic + epistemic)
├── agent.py                  # CAWAAgent orchestrator
├── structure_learner.py      # PC / FCI / GES (causal-learn)
├── perception.py             # YOLOv8-nano → WorldState
├── nlu.py                    # spaCy → WorldState / executable queries
│
│  # Phase 1: Code parser
├── code_parser/
│   ├── parser.py             # tree-sitter orchestrator
│   ├── graph_builder.py      # AST → SQLite dependency graph
│   ├── store.py              # SQLite graph store
│   └── languages/            # Python, JS, TS, Go, Java, Rust
│
│  # Phase 1: Impact analysis
├── impact/
│   ├── analyzer.py           # transitive traversal + risk scoring
│   ├── counterfactual.py     # Pearl's do-operator on code
│   └── reporter.py           # rich terminal output
│
│  # Phase 2: CLI + MCP
├── cli.py                    # typer CLI (init, impact, hotspots, graph, info)
├── mcp_server.py             # 19 MCP tools for LLM integration
│
│  # Phase 3: Memory
├── memory/
│   ├── event_graph.py        # causal graph over events
│   ├── project_memory.py     # session management + recording
│   └── cross_session_learner.py  # patterns across sessions
│
│  # Phase 4: Learning from history
└── history/
    ├── git_learner.py        # git log analysis
    ├── developer_model.py    # AgentMind from commits
    └── fix_evaluator.py      # rank fixes by historical evidence
```

---

## Design Principles

- **Proven, not guessed.** Every answer traces back through an explicit causal chain.
- **Composable primitives.** Each layer is a standalone module with a clear interface.
- **Pearl's do-calculus.** Interventions sever edges, counterfactuals propagate through the modified graph.
- **Belief-based prediction.** Model the *agent's own beliefs*, not ground truth.
- **Expected free energy.** Actions scored by `EFE = -(pragmatic + epistemic)`.
- **Honest uncertainty.** `UNKNOWN` sentinel — refuse to reason about what you don't know.
- **Git history is truth.** Learn from what actually happened, not what was supposed to.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `tree-sitter` + 6 grammars | Code parsing (Python, JS, TS, Go, Java, Rust) |
| `typer` + `rich` | CLI framework and colored output |
| `mcp` | MCP server for LLM integration |
| `causal-learn` | PC / FCI / GES structure discovery |
| `ultralytics` | YOLOv8-nano object detection |
| `spacy` | Dependency parsing for NLU |

---

## Limitations

This is a working reference implementation, not a mature production system.

**Code Parser:** Supports 6 languages with basic function/class/import/call extraction. Does not yet resolve cross-module references, dynamic dispatch, or macro expansion.

**Impact Analysis:** Edge weights are static (CALLS=1.0, INHERITS=0.9, DATA_FLOW=0.8, IMPORTS=0.6). Phase 4 enables tuning these from git history but tuning is not yet automatic.

**Memory:** SQLite-backed persistence works reliably. No synchronization yet for multi-user or multi-machine teams.

**Git Learning:** Co-change and bug-rate analysis work well. Developer expertise modeling is heuristic — treats all commits equally without weighting recency or PR approval.

**Reference Core:** Causal graphs and belief states are hand-built per scenario. Physics primitives are hand-coded axioms. See [`docs/architecture.md`](docs/architecture.md) for honest scoping.

---

## License

MIT
