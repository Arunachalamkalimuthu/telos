# Telos Product — Phase 1: Causal Impact Analyzer

**Date:** 2026-04-16
**Status:** Approved

## Goal

Build `telos` as a production CLI tool that parses codebases into causal dependency graphs and provides impact analysis with counterfactual reasoning. Ship as `pip install telos-cli`.

## Product Definition

Telos is a causal impact analyzer for codebases. It parses source code into a causal dependency graph using tree-sitter, traces full transitive impact chains for any code change, and runs counterfactual analysis (Pearl's do-operator) to evaluate interventions before they're applied.

### CLI Commands

```
telos init [--path .] [--force]
telos impact <target> [--fix <strategy>] [--depth N] [--min-risk 0.5]
telos hotspots [--top N]
telos graph <target> [--depth N]
telos info
```

### Not in Phase 1

- MCP server for LLM integration (Phase 2)
- Memory / event graph layer (Phase 3)
- Git history learning / edge weight tuning (Phase 4)
- Live file watching (Phase 2)
- Developer belief modeling (Phase 4)

---

## Architecture

### Code Parser

tree-sitter parses source files into ASTs. Language-specific extractors pull four types of causal edges:

| Edge type | Meaning | Default weight |
|-----------|---------|---------------|
| CALLS | Function A calls function B | 1.0 |
| DATA_FLOW | Function A passes data to function B | 0.8 |
| IMPORTS | Module A imports from module B | 0.6 |
| INHERITS | Class A extends class B | 0.9 |

Each language gets a thin extraction layer implementing a shared interface:

```python
class LanguageExtractor:
    def extract_functions(self, tree) -> list[Function]
    def extract_imports(self, tree) -> list[Import]
    def extract_calls(self, tree) -> list[Call]
    def extract_classes(self, tree) -> list[Class]
```

**Phase 1 languages:** Python, JavaScript, TypeScript, Go, Java, Rust.

### Impact Analysis Engine

When `telos impact <target>` runs:

1. **Find the node** — look up target in the graph
2. **Transitive propagation** — walk the full causal graph via topological sort, collect every downstream dependent
3. **Risk scoring** — impact score per downstream node = product of edge weights along the path
4. **Counterfactual analysis** (with `--fix`) — apply do-intervention (sever edge at specified point), re-propagate, compare blast radius

### Storage

SQLite database at `.telos/graph.db`:

```sql
nodes (
    id          TEXT PRIMARY KEY,    -- "auth.py:validate_token"
    file_path   TEXT,
    name        TEXT,
    kind        TEXT,                -- "function" | "class" | "module"
    language    TEXT,
    line_start  INTEGER,
    line_end    INTEGER
)

edges (
    id          INTEGER PRIMARY KEY,
    source      TEXT REFERENCES nodes(id),
    target      TEXT REFERENCES nodes(id),
    kind        TEXT,                -- "CALLS" | "IMPORTS" | "DATA_FLOW" | "INHERITS"
    weight      REAL DEFAULT 1.0,
    file_path   TEXT,
    line        INTEGER
)

meta (
    key         TEXT PRIMARY KEY,
    value       TEXT
)
```

Recursive CTEs for transitive traversal. Single file, no server process. Rebuild with `telos init --force`.

---

## CLI Interface

Built with `typer` + `rich` for colored terminal output.

### `telos init`

Scans repository, parses all supported files, builds dependency graph.

Output:
```
$ telos init

Scanning repository...
  Languages detected: Python (42 files), TypeScript (18 files)
  Parsing ASTs... done (60 files in 1.2s)

Building dependency graph...
  Functions:  284
  Imports:    156
  Call edges: 723
  Data flows: 89

Stored graph: .telos/graph.db (1.2 MB)

Hotspots (most depended-on):
  1. src/auth/token.py:validate_token    → 23 downstream dependents
  2. src/db/connection.py:get_pool       → 19 downstream dependents
  3. src/api/middleware.py:require_auth   → 15 downstream dependents

Ready. Run: telos impact <file:function> to trace impact.
```

### `telos impact`

Traces full causal impact of changing a target.

Output:
```
$ telos impact auth.py:validate_token

Impact Analysis: auth.py:validate_token
════════════════════════════════════════

Causal chain (23 functions affected):

  auth.py:validate_token
  ├── CALLS → api/middleware.py:require_auth          [risk: 1.0]
  │   ├── CALLS → payment/process.py:charge_card      [risk: 1.0]
  │   │   └── CALLS → billing/webhook.py:notify        [risk: 1.0]
  │   ├── CALLS → session/manager.py:refresh_session   [risk: 1.0]
  │   └── DATA_FLOW → api/routes.py:user_context       [risk: 0.8]
  ├── IMPORTS → tests/test_auth.py                     [risk: 0.6]
  └── INHERITS → auth/oauth.py:OAuthValidator          [risk: 0.9]

Hottest path: validate_token → require_auth → charge_card → notify
  Severity: HIGH (payment flow affected)

Files affected: 8

Run: telos impact auth.py:validate_token --fix "add_fallback_at:<function>"
     to simulate interventions.
```

### `telos impact --fix` (counterfactual)

```
$ telos impact auth.py:validate_token --fix "add_fallback_at:api/middleware.py:require_auth"

Impact Analysis: auth.py:validate_token (with intervention)
═══════════════════════════════════════════════════════════

Without fix:
  23 functions across 8 files affected

With fix (fallback at api/middleware.py:require_auth):
  auth.py:validate_token
  └── CALLS → api/middleware.py:require_auth          [risk: 1.0]
      ✗ cascade stopped (intervention applied)

  2 functions across 1 file affected

Blast radius reduced: 23 → 2 functions (91% reduction)
```

### `telos hotspots`

```
$ telos hotspots --top 5

Dependency Hotspots
═══════════════════

  #  Function                              Dependents  Risk
  1  src/auth/token.py:validate_token      23          HIGH
  2  src/db/connection.py:get_pool         19          HIGH
  3  src/api/middleware.py:require_auth     15          HIGH
  4  src/models/user.py:User               12          MEDIUM
  5  src/utils/config.py:get_setting        8          MEDIUM
```

### `telos graph`

```
$ telos graph auth.py:validate_token --depth 2

Dependency Graph: auth.py:validate_token (depth=2)
══════════════════════════════════════════════════

  auth.py:validate_token
  ├── CALLS → api/middleware.py:require_auth
  │   ├── CALLS → payment/process.py:charge_card
  │   ├── CALLS → session/manager.py:refresh_session
  │   └── DATA_FLOW → api/routes.py:user_context
  ├── IMPORTS → tests/test_auth.py
  └── INHERITS → auth/oauth.py:OAuthValidator
      └── CALLS → auth/oauth.py:refresh_oauth_token
```

### `telos info`

```
$ telos info

Telos Graph: .telos/graph.db
════════════════════════════

  Repository:  /Users/dev/myproject
  Last scan:   2026-04-16 14:30:22
  Languages:   Python (42), TypeScript (18)
  Files:       60
  Nodes:       284 (231 functions, 38 classes, 15 modules)
  Edges:       723 (512 CALLS, 89 DATA_FLOW, 84 IMPORTS, 38 INHERITS)
  DB size:     1.2 MB
```

---

## File Structure

```
src/telos/
│
│  # Existing reasoning core (unchanged)
├── world.py
├── causal_graph.py
├── physics.py
├── theory_of_mind.py
├── active_inference.py
├── agent.py
├── structure_learner.py
├── perception.py
├── nlu.py
│
│  # NEW — Code Parser
├── code_parser/
│   ├── __init__.py
│   ├── parser.py              # tree-sitter orchestration
│   ├── graph_builder.py       # AST extractions → CausalGraph
│   ├── store.py               # SQLite persistence (.telos/graph.db)
│   └── languages/
│       ├── __init__.py
│       ├── python.py           # Python extractor
│       ├── javascript.py       # JavaScript extractor
│       ├── typescript.py       # TypeScript extractor
│       ├── go.py               # Go extractor
│       ├── java.py             # Java extractor
│       └── rust.py             # Rust extractor
│
│  # NEW — Impact Analysis
├── impact/
│   ├── __init__.py
│   ├── analyzer.py            # transitive propagation + risk scoring
│   ├── counterfactual.py      # do-interventions on code graph
│   └── reporter.py            # rich terminal output formatting
│
│  # NEW — CLI
├── cli.py                     # typer app: init, impact, hotspots, graph, info
```

Tests:
```
tests/
├── test_code_parser/
│   ├── test_parser.py
│   ├── test_graph_builder.py
│   ├── test_store.py
│   └── test_languages/
│       ├── test_python.py
│       ├── test_javascript.py
│       ├── test_typescript.py
│       ├── test_go.py
│       ├── test_java.py
│       └── test_rust.py
├── test_impact/
│   ├── test_analyzer.py
│   ├── test_counterfactual.py
│   └── test_reporter.py
└── test_cli.py
```

---

## Dependencies

```toml
[project]
dependencies = [
    # Existing
    "causal-learn",
    "ultralytics",
    "spacy",
    # New — Phase 1
    "typer",
    "rich",
    "tree-sitter",
    "tree-sitter-python",
    "tree-sitter-javascript",
    "tree-sitter-typescript",
    "tree-sitter-go",
    "tree-sitter-java",
    "tree-sitter-rust",
]

[project.scripts]
telos = "telos.cli:app"
```

---

## Success Criteria

1. `telos init` completes on a 100-file Python+TypeScript repo in under 5 seconds
2. `telos impact` traces full transitive chains and displays colored tree output
3. `telos impact --fix` correctly applies do-intervention and shows reduced blast radius
4. `telos hotspots` ranks nodes by downstream dependent count
5. All 6 language extractors parse real-world files without crashing
6. SQLite graph handles 1000+ nodes and 5000+ edges without performance issues
7. `pip install telos-cli` works and provides the `telos` command
