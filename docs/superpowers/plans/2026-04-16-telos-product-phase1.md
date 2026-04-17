# Telos Product Phase 1: Causal Impact Analyzer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `telos` as a production CLI tool that parses codebases into causal dependency graphs via tree-sitter and provides impact analysis with counterfactual reasoning.

**Architecture:** tree-sitter parses 6 languages into ASTs → language extractors pull functions/classes/imports/calls → graph_builder creates CausalGraph edges → SQLite persists the graph → impact analyzer traverses transitively with risk scoring → counterfactual engine applies do-interventions → rich CLI outputs colored tree views.

**Tech Stack:** tree-sitter (0.25), tree-sitter-{python,javascript,typescript,go,java,rust}, typer, rich, sqlite3 (stdlib)

**API notes (tree-sitter 0.25):** No query/capture API. Use `node.type`, `node.children`, `node.named_children`, `node.child_by_field_name()`, `node.text.decode()`. Language grammars expose a `language()` function (typescript has `language_typescript()` and `language_tsx()`). Parser takes Language in constructor: `Parser(Language(grammar.language()))`.

---

### Task 1: Dependencies and Project Structure

**Files:**
- Modify: `pyproject.toml`
- Create: `src/telos/code_parser/__init__.py`
- Create: `src/telos/code_parser/languages/__init__.py`
- Create: `src/telos/impact/__init__.py`

- [ ] **Step 1: Update pyproject.toml**

Add new dependencies and CLI entry point:

```toml
[project]
name = "telos"
version = "0.2.0"
description = "telos — causal impact analyzer for codebases"
requires-python = ">=3.10"
dependencies = [
    "causal-learn",
    "ultralytics",
    "spacy",
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

- [ ] **Step 2: Create package directories**

```bash
mkdir -p src/telos/code_parser/languages src/telos/impact
```

Create empty `__init__.py` files:

`src/telos/code_parser/__init__.py`:
```python
"""Code parser: tree-sitter AST → causal dependency graph."""
```

`src/telos/code_parser/languages/__init__.py`:
```python
"""Language-specific tree-sitter extractors."""
```

`src/telos/impact/__init__.py`:
```python
"""Impact analysis: transitive traversal, risk scoring, counterfactuals."""
```

- [ ] **Step 3: Install dependencies**

```bash
pip install -e .
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml src/telos/code_parser/ src/telos/impact/
git commit -m "build: add Phase 1 dependencies and package structure"
```

---

### Task 2: SQLite Store

**Files:**
- Create: `src/telos/code_parser/store.py`
- Create: `tests/test_code_parser/test_store.py`

- [ ] **Step 1: Write failing tests**

```python
"""tests/test_code_parser/test_store.py"""
import os
import tempfile
import unittest

from telos.code_parser.store import GraphStore


class TestGraphStore(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, ".telos", "graph.db")
        self.store = GraphStore(self.db_path)

    def tearDown(self):
        self.store.close()

    def test_creates_db_and_tables(self):
        self.assertTrue(os.path.exists(self.db_path))

    def test_add_and_get_node(self):
        self.store.add_node(
            id="auth.py:validate_token",
            file_path="src/auth.py",
            name="validate_token",
            kind="function",
            language="python",
            line_start=10,
            line_end=25,
        )
        node = self.store.get_node("auth.py:validate_token")
        self.assertEqual(node["name"], "validate_token")
        self.assertEqual(node["kind"], "function")
        self.assertEqual(node["language"], "python")

    def test_add_and_get_edge(self):
        self.store.add_node(id="a", file_path="a.py", name="a", kind="function", language="python")
        self.store.add_node(id="b", file_path="b.py", name="b", kind="function", language="python")
        self.store.add_edge(source="a", target="b", kind="CALLS", weight=1.0, file_path="a.py", line=5)
        edges = self.store.get_edges_from("a")
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["target"], "b")
        self.assertEqual(edges[0]["kind"], "CALLS")

    def test_get_edges_to(self):
        self.store.add_node(id="a", file_path="a.py", name="a", kind="function", language="python")
        self.store.add_node(id="b", file_path="b.py", name="b", kind="function", language="python")
        self.store.add_edge(source="a", target="b", kind="CALLS", weight=1.0, file_path="a.py", line=5)
        edges = self.store.get_edges_to("b")
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["source"], "a")

    def test_get_all_nodes(self):
        self.store.add_node(id="a", file_path="a.py", name="a", kind="function", language="python")
        self.store.add_node(id="b", file_path="b.py", name="b", kind="function", language="python")
        nodes = self.store.get_all_nodes()
        self.assertEqual(len(nodes), 2)

    def test_set_and_get_meta(self):
        self.store.set_meta("last_scan", "2026-04-16")
        self.assertEqual(self.store.get_meta("last_scan"), "2026-04-16")

    def test_clear_removes_all(self):
        self.store.add_node(id="a", file_path="a.py", name="a", kind="function", language="python")
        self.store.clear()
        self.assertEqual(len(self.store.get_all_nodes()), 0)

    def test_get_stats(self):
        self.store.add_node(id="a", file_path="a.py", name="a", kind="function", language="python")
        self.store.add_node(id="b", file_path="b.py", name="b", kind="class", language="python")
        self.store.add_edge(source="a", target="b", kind="CALLS", weight=1.0, file_path="a.py", line=1)
        stats = self.store.get_stats()
        self.assertEqual(stats["node_count"], 2)
        self.assertEqual(stats["edge_count"], 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
mkdir -p tests/test_code_parser
touch tests/test_code_parser/__init__.py
PYTHONPATH=src python3.11 -m pytest tests/test_code_parser/test_store.py -v
```
Expected: ImportError

- [ ] **Step 3: Implement store.py**

```python
"""src/telos/code_parser/store.py — SQLite persistence for the dependency graph."""

from __future__ import annotations

import os
import sqlite3
from typing import Any


class GraphStore:
    """SQLite-backed storage for code dependency graph."""

    def __init__(self, db_path: str) -> None:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id          TEXT PRIMARY KEY,
                file_path   TEXT,
                name        TEXT,
                kind        TEXT,
                language    TEXT,
                line_start  INTEGER DEFAULT 0,
                line_end    INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS edges (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source      TEXT REFERENCES nodes(id),
                target      TEXT REFERENCES nodes(id),
                kind        TEXT,
                weight      REAL DEFAULT 1.0,
                file_path   TEXT,
                line        INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
        """)
        self._conn.commit()

    def add_node(self, id: str, file_path: str, name: str, kind: str,
                 language: str, line_start: int = 0, line_end: int = 0) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO nodes (id, file_path, name, kind, language, line_start, line_end) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (id, file_path, name, kind, language, line_start, line_end),
        )
        self._conn.commit()

    def get_node(self, id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM nodes WHERE id = ?", (id,)).fetchone()
        return dict(row) if row else None

    def get_all_nodes(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._conn.execute("SELECT * FROM nodes").fetchall()]

    def add_edge(self, source: str, target: str, kind: str,
                 weight: float = 1.0, file_path: str = "", line: int = 0) -> None:
        self._conn.execute(
            "INSERT INTO edges (source, target, kind, weight, file_path, line) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (source, target, kind, weight, file_path, line),
        )
        self._conn.commit()

    def get_edges_from(self, source: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM edges WHERE source = ?", (source,)
        ).fetchall()]

    def get_edges_to(self, target: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM edges WHERE target = ?", (target,)
        ).fetchall()]

    def get_all_edges(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._conn.execute("SELECT * FROM edges").fetchall()]

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value)
        )
        self._conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def get_stats(self) -> dict[str, int]:
        node_count = self._conn.execute("SELECT COUNT(*) as c FROM nodes").fetchone()["c"]
        edge_count = self._conn.execute("SELECT COUNT(*) as c FROM edges").fetchone()["c"]
        return {"node_count": node_count, "edge_count": edge_count}

    def clear(self) -> None:
        self._conn.executescript("DELETE FROM edges; DELETE FROM nodes; DELETE FROM meta;")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=src python3.11 -m pytest tests/test_code_parser/test_store.py -v
```
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/telos/code_parser/store.py tests/test_code_parser/
git commit -m "feat: add SQLite graph store"
```

---

### Task 3: Tree-sitter Parser Orchestrator

**Files:**
- Create: `src/telos/code_parser/parser.py`
- Create: `tests/test_code_parser/test_parser.py`

- [ ] **Step 1: Write failing tests**

```python
"""tests/test_code_parser/test_parser.py"""
import unittest

from telos.code_parser.parser import TelosParser


class TestTelosParser(unittest.TestCase):

    def test_detect_language_python(self):
        self.assertEqual(TelosParser.detect_language("src/auth.py"), "python")

    def test_detect_language_javascript(self):
        self.assertEqual(TelosParser.detect_language("app/index.js"), "javascript")

    def test_detect_language_typescript(self):
        self.assertEqual(TelosParser.detect_language("app/index.ts"), "typescript")

    def test_detect_language_tsx(self):
        self.assertEqual(TelosParser.detect_language("app/Component.tsx"), "tsx")

    def test_detect_language_go(self):
        self.assertEqual(TelosParser.detect_language("main.go"), "go")

    def test_detect_language_java(self):
        self.assertEqual(TelosParser.detect_language("Main.java"), "java")

    def test_detect_language_rust(self):
        self.assertEqual(TelosParser.detect_language("lib.rs"), "rust")

    def test_detect_language_unknown(self):
        self.assertIsNone(TelosParser.detect_language("README.md"))

    def test_parse_python_source(self):
        parser = TelosParser()
        tree = parser.parse(b"def hello(): pass", "python")
        self.assertEqual(tree.root_node.type, "module")

    def test_supported_languages(self):
        parser = TelosParser()
        langs = parser.supported_languages()
        self.assertIn("python", langs)
        self.assertIn("javascript", langs)
        self.assertIn("typescript", langs)
        self.assertIn("go", langs)
        self.assertIn("java", langs)
        self.assertIn("rust", langs)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src python3.11 -m pytest tests/test_code_parser/test_parser.py -v
```

- [ ] **Step 3: Implement parser.py**

```python
"""src/telos/code_parser/parser.py — tree-sitter orchestration for multiple languages."""

from __future__ import annotations

from tree_sitter import Language, Parser, Tree

import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_typescript
import tree_sitter_go
import tree_sitter_java
import tree_sitter_rust


_EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
}

_GRAMMAR_MAP: dict[str, Language] = {}


def _get_language(name: str) -> Language:
    if name not in _GRAMMAR_MAP:
        grammars = {
            "python": tree_sitter_python.language(),
            "javascript": tree_sitter_javascript.language(),
            "typescript": tree_sitter_typescript.language_typescript(),
            "tsx": tree_sitter_typescript.language_tsx(),
            "go": tree_sitter_go.language(),
            "java": tree_sitter_java.language(),
            "rust": tree_sitter_rust.language(),
        }
        if name not in grammars:
            raise ValueError(f"unsupported language: {name}")
        _GRAMMAR_MAP[name] = Language(grammars[name])
    return _GRAMMAR_MAP[name]


class TelosParser:
    """Multi-language tree-sitter parser."""

    @staticmethod
    def detect_language(file_path: str) -> str | None:
        import os
        _, ext = os.path.splitext(file_path)
        return _EXTENSION_MAP.get(ext.lower())

    @staticmethod
    def supported_languages() -> list[str]:
        return ["python", "javascript", "tsx", "typescript", "go", "java", "rust"]

    def parse(self, source: bytes, language: str) -> Tree:
        lang = _get_language(language)
        parser = Parser(lang)
        return parser.parse(source)
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=src python3.11 -m pytest tests/test_code_parser/test_parser.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/telos/code_parser/parser.py tests/test_code_parser/test_parser.py
git commit -m "feat: add tree-sitter parser orchestrator"
```

---

### Task 4: Python Language Extractor

**Files:**
- Create: `src/telos/code_parser/languages/python.py`
- Create: `tests/test_code_parser/test_languages/test_python.py`

- [ ] **Step 1: Write failing tests**

```python
"""tests/test_code_parser/test_languages/test_python.py"""
import unittest

from telos.code_parser.parser import TelosParser
from telos.code_parser.languages.python import PythonExtractor


class TestPythonExtractor(unittest.TestCase):

    def setUp(self):
        self.parser = TelosParser()
        self.extractor = PythonExtractor()

    def _parse(self, code: str):
        return self.parser.parse(code.encode(), "python")

    def test_extract_functions(self):
        tree = self._parse("def hello(name):\n    pass\n\ndef world():\n    pass")
        funcs = self.extractor.extract_functions(tree, "test.py")
        names = [f["name"] for f in funcs]
        self.assertIn("hello", names)
        self.assertIn("world", names)
        hello = [f for f in funcs if f["name"] == "hello"][0]
        self.assertEqual(hello["kind"], "function")

    def test_extract_classes(self):
        tree = self._parse("class MyClass(Base):\n    def method(self):\n        pass")
        classes = self.extractor.extract_classes(tree, "test.py")
        self.assertEqual(len(classes), 1)
        self.assertEqual(classes[0]["name"], "MyClass")
        self.assertEqual(classes[0]["bases"], ["Base"])

    def test_extract_imports(self):
        tree = self._parse("import os\nfrom pathlib import Path\nfrom . import utils")
        imports = self.extractor.extract_imports(tree, "test.py")
        modules = [i["module"] for i in imports]
        self.assertIn("os", modules)
        self.assertIn("pathlib", modules)

    def test_extract_calls(self):
        tree = self._parse("def main():\n    hello()\n    obj.method()\n    print('hi')")
        calls = self.extractor.extract_calls(tree, "test.py")
        call_names = [c["name"] for c in calls]
        self.assertIn("hello", call_names)
        self.assertIn("print", call_names)

    def test_extract_methods_inside_class(self):
        tree = self._parse("class Foo:\n    def bar(self):\n        pass\n    def baz(self):\n        pass")
        funcs = self.extractor.extract_functions(tree, "test.py")
        names = [f["name"] for f in funcs]
        self.assertIn("bar", names)
        self.assertIn("baz", names)

    def test_extract_decorated_function(self):
        tree = self._parse("@app.route('/api')\ndef handler():\n    pass")
        funcs = self.extractor.extract_functions(tree, "test.py")
        self.assertEqual(len(funcs), 1)
        self.assertEqual(funcs[0]["name"], "handler")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
mkdir -p tests/test_code_parser/test_languages
touch tests/test_code_parser/test_languages/__init__.py
PYTHONPATH=src python3.11 -m pytest tests/test_code_parser/test_languages/test_python.py -v
```

- [ ] **Step 3: Implement Python extractor**

```python
"""src/telos/code_parser/languages/python.py — Python AST extraction via tree-sitter."""

from __future__ import annotations

from tree_sitter import Node, Tree
from typing import Any


class PythonExtractor:
    """Extract functions, classes, imports, and calls from a Python AST."""

    def extract_functions(self, tree: Tree, file_path: str) -> list[dict[str, Any]]:
        funcs: list[dict[str, Any]] = []
        self._walk_for_functions(tree.root_node, file_path, funcs)
        return funcs

    def _walk_for_functions(self, node: Node, file_path: str, results: list) -> None:
        if node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                results.append({
                    "name": name_node.text.decode(),
                    "kind": "function",
                    "file_path": file_path,
                    "line_start": node.start_point[0] + 1,
                    "line_end": node.end_point[0] + 1,
                })
        for child in node.children:
            self._walk_for_functions(child, file_path, results)

    def extract_classes(self, tree: Tree, file_path: str) -> list[dict[str, Any]]:
        classes: list[dict[str, Any]] = []
        self._walk_for_classes(tree.root_node, file_path, classes)
        return classes

    def _walk_for_classes(self, node: Node, file_path: str, results: list) -> None:
        if node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                bases: list[str] = []
                superclasses = node.child_by_field_name("superclasses")
                if superclasses:
                    for child in superclasses.named_children:
                        bases.append(child.text.decode())
                results.append({
                    "name": name_node.text.decode(),
                    "kind": "class",
                    "file_path": file_path,
                    "line_start": node.start_point[0] + 1,
                    "line_end": node.end_point[0] + 1,
                    "bases": bases,
                })
        for child in node.children:
            self._walk_for_classes(child, file_path, results)

    def extract_imports(self, tree: Tree, file_path: str) -> list[dict[str, Any]]:
        imports: list[dict[str, Any]] = []
        self._walk_for_imports(tree.root_node, file_path, imports)
        return imports

    def _walk_for_imports(self, node: Node, file_path: str, results: list) -> None:
        if node.type == "import_statement":
            for child in node.named_children:
                if child.type == "dotted_name":
                    results.append({
                        "module": child.text.decode(),
                        "names": [],
                        "file_path": file_path,
                        "line": node.start_point[0] + 1,
                    })
        elif node.type == "import_from_statement":
            module_node = node.child_by_field_name("module_name")
            module = module_node.text.decode() if module_node else ""
            names: list[str] = []
            for child in node.named_children:
                if child.type == "dotted_name" and child != module_node:
                    names.append(child.text.decode())
            results.append({
                "module": module,
                "names": names,
                "file_path": file_path,
                "line": node.start_point[0] + 1,
            })
        for child in node.children:
            if node.type not in ("import_statement", "import_from_statement"):
                self._walk_for_imports(child, file_path, results)

    def extract_calls(self, tree: Tree, file_path: str) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        self._walk_for_calls(tree.root_node, file_path, calls)
        return calls

    def _walk_for_calls(self, node: Node, file_path: str, results: list) -> None:
        if node.type == "call":
            func_node = node.child_by_field_name("function")
            if func_node:
                name = func_node.text.decode()
                # For attribute calls like obj.method(), extract just the method name
                if func_node.type == "attribute":
                    attr = func_node.child_by_field_name("attribute")
                    if attr:
                        name = attr.text.decode()
                elif func_node.type == "identifier":
                    name = func_node.text.decode()
                results.append({
                    "name": name,
                    "full_name": func_node.text.decode(),
                    "file_path": file_path,
                    "line": node.start_point[0] + 1,
                })
        for child in node.children:
            self._walk_for_calls(child, file_path, results)
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=src python3.11 -m pytest tests/test_code_parser/test_languages/test_python.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/telos/code_parser/languages/python.py tests/test_code_parser/test_languages/
git commit -m "feat: add Python language extractor"
```

---

### Task 5: JavaScript Language Extractor

**Files:**
- Create: `src/telos/code_parser/languages/javascript.py`
- Create: `tests/test_code_parser/test_languages/test_javascript.py`

- [ ] **Step 1: Write failing tests**

```python
"""tests/test_code_parser/test_languages/test_javascript.py"""
import unittest

from telos.code_parser.parser import TelosParser
from telos.code_parser.languages.javascript import JavaScriptExtractor


class TestJavaScriptExtractor(unittest.TestCase):

    def setUp(self):
        self.parser = TelosParser()
        self.extractor = JavaScriptExtractor()

    def _parse(self, code: str):
        return self.parser.parse(code.encode(), "javascript")

    def test_extract_functions(self):
        tree = self._parse("function hello(name) { return name; }\nfunction world() {}")
        funcs = self.extractor.extract_functions(tree, "test.js")
        names = [f["name"] for f in funcs]
        self.assertIn("hello", names)
        self.assertIn("world", names)

    def test_extract_arrow_functions(self):
        tree = self._parse("const greet = (name) => { return name; };")
        funcs = self.extractor.extract_functions(tree, "test.js")
        names = [f["name"] for f in funcs]
        self.assertIn("greet", names)

    def test_extract_classes(self):
        tree = self._parse("class MyClass extends Base { constructor() {} }")
        classes = self.extractor.extract_classes(tree, "test.js")
        self.assertEqual(len(classes), 1)
        self.assertEqual(classes[0]["name"], "MyClass")

    def test_extract_imports(self):
        tree = self._parse("import { foo } from 'bar';\nconst x = require('baz');")
        imports = self.extractor.extract_imports(tree, "test.js")
        modules = [i["module"] for i in imports]
        self.assertIn("bar", modules)

    def test_extract_calls(self):
        tree = self._parse("function main() { hello(); console.log('hi'); }")
        calls = self.extractor.extract_calls(tree, "test.js")
        call_names = [c["name"] for c in calls]
        self.assertIn("hello", call_names)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Implement JavaScript extractor**

```python
"""src/telos/code_parser/languages/javascript.py — JavaScript/JSX extraction via tree-sitter."""

from __future__ import annotations

from tree_sitter import Node, Tree
from typing import Any


class JavaScriptExtractor:
    """Extract functions, classes, imports, and calls from JavaScript AST."""

    def extract_functions(self, tree: Tree, file_path: str) -> list[dict[str, Any]]:
        funcs: list[dict[str, Any]] = []
        self._walk_for_functions(tree.root_node, file_path, funcs)
        return funcs

    def _walk_for_functions(self, node: Node, file_path: str, results: list) -> None:
        if node.type == "function_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                results.append({
                    "name": name_node.text.decode(),
                    "kind": "function",
                    "file_path": file_path,
                    "line_start": node.start_point[0] + 1,
                    "line_end": node.end_point[0] + 1,
                })
        elif node.type in ("lexical_declaration", "variable_declaration"):
            for child in node.named_children:
                if child.type == "variable_declarator":
                    name_node = child.child_by_field_name("name")
                    value_node = child.child_by_field_name("value")
                    if name_node and value_node and value_node.type == "arrow_function":
                        results.append({
                            "name": name_node.text.decode(),
                            "kind": "function",
                            "file_path": file_path,
                            "line_start": node.start_point[0] + 1,
                            "line_end": node.end_point[0] + 1,
                        })
        for child in node.children:
            self._walk_for_functions(child, file_path, results)

    def extract_classes(self, tree: Tree, file_path: str) -> list[dict[str, Any]]:
        classes: list[dict[str, Any]] = []
        self._walk_for_classes(tree.root_node, file_path, classes)
        return classes

    def _walk_for_classes(self, node: Node, file_path: str, results: list) -> None:
        if node.type == "class_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                bases: list[str] = []
                for child in node.children:
                    if child.type == "class_heritage":
                        for c in child.named_children:
                            bases.append(c.text.decode())
                results.append({
                    "name": name_node.text.decode(),
                    "kind": "class",
                    "file_path": file_path,
                    "line_start": node.start_point[0] + 1,
                    "line_end": node.end_point[0] + 1,
                    "bases": bases,
                })
        for child in node.children:
            self._walk_for_classes(child, file_path, results)

    def extract_imports(self, tree: Tree, file_path: str) -> list[dict[str, Any]]:
        imports: list[dict[str, Any]] = []
        self._walk_for_imports(tree.root_node, file_path, imports)
        return imports

    def _walk_for_imports(self, node: Node, file_path: str, results: list) -> None:
        if node.type == "import_statement":
            source = node.child_by_field_name("source")
            if source:
                module = source.text.decode().strip("'\"")
                results.append({
                    "module": module,
                    "names": [],
                    "file_path": file_path,
                    "line": node.start_point[0] + 1,
                })
        for child in node.children:
            if node.type != "import_statement":
                self._walk_for_imports(child, file_path, results)

    def extract_calls(self, tree: Tree, file_path: str) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        self._walk_for_calls(tree.root_node, file_path, calls)
        return calls

    def _walk_for_calls(self, node: Node, file_path: str, results: list) -> None:
        if node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            if func_node:
                name = func_node.text.decode()
                if func_node.type == "member_expression":
                    prop = func_node.child_by_field_name("property")
                    if prop:
                        name = prop.text.decode()
                elif func_node.type == "identifier":
                    name = func_node.text.decode()
                results.append({
                    "name": name,
                    "full_name": func_node.text.decode(),
                    "file_path": file_path,
                    "line": node.start_point[0] + 1,
                })
        for child in node.children:
            self._walk_for_calls(child, file_path, results)
```

- [ ] **Step 3: Run tests**

```bash
PYTHONPATH=src python3.11 -m pytest tests/test_code_parser/test_languages/test_javascript.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/telos/code_parser/languages/javascript.py tests/test_code_parser/test_languages/test_javascript.py
git commit -m "feat: add JavaScript language extractor"
```

---

### Task 6: TypeScript, Go, Java, Rust Extractors

**Files:**
- Create: `src/telos/code_parser/languages/typescript.py`
- Create: `src/telos/code_parser/languages/go.py`
- Create: `src/telos/code_parser/languages/java.py`
- Create: `src/telos/code_parser/languages/rust.py`
- Create: `tests/test_code_parser/test_languages/test_typescript.py`
- Create: `tests/test_code_parser/test_languages/test_go.py`
- Create: `tests/test_code_parser/test_languages/test_java.py`
- Create: `tests/test_code_parser/test_languages/test_rust.py`

Each extractor follows the same interface as PythonExtractor and JavaScriptExtractor: `extract_functions`, `extract_classes`, `extract_imports`, `extract_calls`. The tree-sitter node types differ per language:

**TypeScript** — same as JavaScript but parses with `language_typescript()` / `language_tsx()`. Adds `type_alias_declaration` and `interface_declaration` as class-like nodes. Import syntax identical to JS ES6.

**Go** — `function_declaration` for functions, `method_declaration` for methods, `type_declaration` with `struct_type` for classes, `import_declaration` for imports, `call_expression` for calls.

**Java** — `method_declaration` for functions, `class_declaration` for classes, `import_declaration` for imports, `method_invocation` for calls.

**Rust** — `function_item` for functions, `struct_item`/`impl_item` for classes, `use_declaration` for imports, `call_expression` for calls.

- [ ] **Step 1: Implement all four extractors**

Each extractor is ~80-120 lines following the same walk pattern as Python/JavaScript. The key difference is the tree-sitter node type names per language.

- [ ] **Step 2: Write tests for each (one test class per language, 4-5 tests each)**

Each test class verifies: extract_functions, extract_classes, extract_imports, extract_calls on minimal valid source code for that language.

- [ ] **Step 3: Run all language tests**

```bash
PYTHONPATH=src python3.11 -m pytest tests/test_code_parser/test_languages/ -v
```

- [ ] **Step 4: Commit**

```bash
git add src/telos/code_parser/languages/ tests/test_code_parser/test_languages/
git commit -m "feat: add TypeScript, Go, Java, Rust language extractors"
```

---

### Task 7: Graph Builder

**Files:**
- Create: `src/telos/code_parser/graph_builder.py`
- Create: `tests/test_code_parser/test_graph_builder.py`

- [ ] **Step 1: Write failing tests**

```python
"""tests/test_code_parser/test_graph_builder.py"""
import os
import tempfile
import unittest

from telos.code_parser.graph_builder import GraphBuilder
from telos.code_parser.store import GraphStore


class TestGraphBuilder(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, ".telos", "graph.db")
        self.store = GraphStore(self.db_path)
        self.builder = GraphBuilder(self.store)

    def tearDown(self):
        self.store.close()

    def test_scan_python_file(self):
        py_file = os.path.join(self.tmpdir, "example.py")
        with open(py_file, "w") as f:
            f.write("def hello():\n    pass\n\ndef world():\n    hello()\n")
        self.builder.scan_file(py_file, self.tmpdir)
        nodes = self.store.get_all_nodes()
        names = [n["name"] for n in nodes]
        self.assertIn("hello", names)
        self.assertIn("world", names)

    def test_scan_creates_call_edges(self):
        py_file = os.path.join(self.tmpdir, "example.py")
        with open(py_file, "w") as f:
            f.write("def hello():\n    pass\n\ndef world():\n    hello()\n")
        self.builder.scan_file(py_file, self.tmpdir)
        edges = self.store.get_all_edges()
        call_edges = [e for e in edges if e["kind"] == "CALLS"]
        self.assertGreater(len(call_edges), 0)

    def test_scan_directory(self):
        py_file1 = os.path.join(self.tmpdir, "a.py")
        py_file2 = os.path.join(self.tmpdir, "b.py")
        with open(py_file1, "w") as f:
            f.write("def func_a():\n    pass\n")
        with open(py_file2, "w") as f:
            f.write("def func_b():\n    func_a()\n")
        stats = self.builder.scan_directory(self.tmpdir)
        self.assertEqual(stats["files_scanned"], 2)
        self.assertGreater(stats["nodes"], 0)

    def test_scan_ignores_unsupported_files(self):
        md_file = os.path.join(self.tmpdir, "README.md")
        with open(md_file, "w") as f:
            f.write("# Hello")
        stats = self.builder.scan_directory(self.tmpdir)
        self.assertEqual(stats["files_scanned"], 0)

    def test_scan_creates_import_edges(self):
        py_file = os.path.join(self.tmpdir, "example.py")
        with open(py_file, "w") as f:
            f.write("import os\nfrom pathlib import Path\n")
        self.builder.scan_file(py_file, self.tmpdir)
        edges = self.store.get_all_edges()
        import_edges = [e for e in edges if e["kind"] == "IMPORTS"]
        self.assertGreater(len(import_edges), 0)

    def test_scan_creates_inherits_edges(self):
        py_file = os.path.join(self.tmpdir, "example.py")
        with open(py_file, "w") as f:
            f.write("class Base:\n    pass\n\nclass Child(Base):\n    pass\n")
        self.builder.scan_file(py_file, self.tmpdir)
        edges = self.store.get_all_edges()
        inherit_edges = [e for e in edges if e["kind"] == "INHERITS"]
        self.assertGreater(len(inherit_edges), 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Implement graph_builder.py**

```python
"""src/telos/code_parser/graph_builder.py — scan files and build dependency graph."""

from __future__ import annotations

import os
from typing import Any

from .parser import TelosParser
from .store import GraphStore
from .languages.python import PythonExtractor
from .languages.javascript import JavaScriptExtractor


# Edge type weights per the spec.
_EDGE_WEIGHTS: dict[str, float] = {
    "CALLS": 1.0,
    "DATA_FLOW": 0.8,
    "IMPORTS": 0.6,
    "INHERITS": 0.9,
}

# Map language names to extractor instances.
_EXTRACTORS: dict[str, Any] = {}


def _get_extractor(language: str):
    if not _EXTRACTORS:
        _EXTRACTORS["python"] = PythonExtractor()
        _EXTRACTORS["javascript"] = JavaScriptExtractor()
        _EXTRACTORS["tsx"] = JavaScriptExtractor()  # JSX/TSX use same node types
        # Import remaining extractors lazily to avoid circular issues during build.
        try:
            from .languages.typescript import TypeScriptExtractor
            _EXTRACTORS["typescript"] = TypeScriptExtractor()
        except ImportError:
            _EXTRACTORS["typescript"] = JavaScriptExtractor()
        try:
            from .languages.go import GoExtractor
            _EXTRACTORS["go"] = GoExtractor()
        except ImportError:
            pass
        try:
            from .languages.java import JavaExtractor
            _EXTRACTORS["java"] = JavaExtractor()
        except ImportError:
            pass
        try:
            from .languages.rust import RustExtractor
            _EXTRACTORS["rust"] = RustExtractor()
        except ImportError:
            pass
    return _EXTRACTORS.get(language)


def _make_node_id(file_path: str, name: str, repo_root: str) -> str:
    rel = os.path.relpath(file_path, repo_root)
    return f"{rel}:{name}"


class GraphBuilder:
    """Scan source files and populate the graph store with nodes and edges."""

    def __init__(self, store: GraphStore) -> None:
        self.store = store
        self.parser = TelosParser()
        # Track all function/class names per file for call resolution.
        self._symbol_table: dict[str, str] = {}  # name → node_id

    def scan_file(self, file_path: str, repo_root: str) -> None:
        language = self.parser.detect_language(file_path)
        if language is None:
            return

        extractor = _get_extractor(language)
        if extractor is None:
            return

        with open(file_path, "rb") as f:
            source = f.read()

        tree = self.parser.parse(source, language)
        rel_path = os.path.relpath(file_path, repo_root)

        # Add module node.
        module_id = rel_path
        self.store.add_node(
            id=module_id, file_path=rel_path, name=os.path.basename(rel_path),
            kind="module", language=language,
        )

        # Extract and add function nodes.
        functions = extractor.extract_functions(tree, rel_path)
        for func in functions:
            node_id = f"{rel_path}:{func['name']}"
            self.store.add_node(
                id=node_id, file_path=rel_path, name=func["name"],
                kind=func.get("kind", "function"), language=language,
                line_start=func.get("line_start", 0), line_end=func.get("line_end", 0),
            )
            self._symbol_table[func["name"]] = node_id

        # Extract and add class nodes.
        classes = extractor.extract_classes(tree, rel_path)
        for cls in classes:
            node_id = f"{rel_path}:{cls['name']}"
            self.store.add_node(
                id=node_id, file_path=rel_path, name=cls["name"],
                kind="class", language=language,
                line_start=cls.get("line_start", 0), line_end=cls.get("line_end", 0),
            )
            self._symbol_table[cls["name"]] = node_id

            # INHERITS edges.
            for base in cls.get("bases", []):
                base_id = self._symbol_table.get(base)
                if base_id:
                    self.store.add_edge(
                        source=node_id, target=base_id, kind="INHERITS",
                        weight=_EDGE_WEIGHTS["INHERITS"], file_path=rel_path,
                        line=cls.get("line_start", 0),
                    )

        # Extract imports → IMPORTS edges.
        imports = extractor.extract_imports(tree, rel_path)
        for imp in imports:
            target_id = imp["module"]
            self.store.add_edge(
                source=module_id, target=target_id, kind="IMPORTS",
                weight=_EDGE_WEIGHTS["IMPORTS"], file_path=rel_path,
                line=imp.get("line", 0),
            )

        # Extract calls → CALLS edges.
        calls = extractor.extract_calls(tree, rel_path)
        for call in calls:
            caller_id = self._find_enclosing_function(functions, call.get("line", 0), rel_path)
            if caller_id is None:
                caller_id = module_id
            target_id = self._symbol_table.get(call["name"])
            if target_id and target_id != caller_id:
                self.store.add_edge(
                    source=caller_id, target=target_id, kind="CALLS",
                    weight=_EDGE_WEIGHTS["CALLS"], file_path=rel_path,
                    line=call.get("line", 0),
                )

    def _find_enclosing_function(
        self, functions: list[dict], line: int, rel_path: str,
    ) -> str | None:
        for func in functions:
            if func.get("line_start", 0) <= line <= func.get("line_end", 0):
                return f"{rel_path}:{func['name']}"
        return None

    def scan_directory(
        self, root: str, exclude_dirs: set[str] | None = None,
    ) -> dict[str, int]:
        if exclude_dirs is None:
            exclude_dirs = {".git", ".telos", "node_modules", "__pycache__", ".venv", "venv", "target", "build", "dist"}

        files_scanned = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                lang = self.parser.detect_language(fpath)
                if lang is not None:
                    self.scan_file(fpath, root)
                    files_scanned += 1

        stats = self.store.get_stats()
        return {
            "files_scanned": files_scanned,
            "nodes": stats["node_count"],
            "edges": stats["edge_count"],
        }
```

- [ ] **Step 3: Run tests**

```bash
PYTHONPATH=src python3.11 -m pytest tests/test_code_parser/test_graph_builder.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/telos/code_parser/graph_builder.py tests/test_code_parser/test_graph_builder.py
git commit -m "feat: add graph builder (scan files → nodes + edges)"
```

---

### Task 8: Impact Analyzer

**Files:**
- Create: `src/telos/impact/analyzer.py`
- Create: `tests/test_impact/test_analyzer.py`

- [ ] **Step 1: Write failing tests**

```python
"""tests/test_impact/test_analyzer.py"""
import os
import tempfile
import unittest

from telos.code_parser.store import GraphStore
from telos.impact.analyzer import ImpactAnalyzer


class TestImpactAnalyzer(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, ".telos", "graph.db")
        self.store = GraphStore(self.db_path)
        self._build_test_graph()
        self.analyzer = ImpactAnalyzer(self.store)

    def _build_test_graph(self):
        # a → b → c → d, e → c
        for n in ["a", "b", "c", "d", "e"]:
            self.store.add_node(id=n, file_path=f"{n}.py", name=n, kind="function", language="python")
        self.store.add_edge(source="a", target="b", kind="CALLS", weight=1.0, file_path="a.py", line=1)
        self.store.add_edge(source="b", target="c", kind="CALLS", weight=1.0, file_path="b.py", line=1)
        self.store.add_edge(source="c", target="d", kind="DATA_FLOW", weight=0.8, file_path="c.py", line=1)
        self.store.add_edge(source="e", target="c", kind="CALLS", weight=1.0, file_path="e.py", line=1)

    def tearDown(self):
        self.store.close()

    def test_direct_impact(self):
        result = self.analyzer.analyze("a")
        affected = [r["node_id"] for r in result["affected"]]
        self.assertIn("b", affected)

    def test_transitive_impact(self):
        result = self.analyzer.analyze("a")
        affected = [r["node_id"] for r in result["affected"]]
        self.assertIn("b", affected)
        self.assertIn("c", affected)
        self.assertIn("d", affected)

    def test_risk_scores_decrease_along_chain(self):
        result = self.analyzer.analyze("a")
        scores = {r["node_id"]: r["risk"] for r in result["affected"]}
        self.assertGreaterEqual(scores["b"], scores["d"])

    def test_depth_limit(self):
        result = self.analyzer.analyze("a", max_depth=1)
        affected = [r["node_id"] for r in result["affected"]]
        self.assertIn("b", affected)
        self.assertNotIn("d", affected)

    def test_min_risk_filter(self):
        result = self.analyzer.analyze("a", min_risk=0.9)
        # d has risk 1.0 * 1.0 * 0.8 = 0.8, should be filtered
        affected = [r["node_id"] for r in result["affected"]]
        self.assertNotIn("d", affected)

    def test_hottest_path(self):
        result = self.analyzer.analyze("a")
        path = result["hottest_path"]
        self.assertEqual(path[0], "a")
        self.assertIn("d", path)

    def test_affected_files(self):
        result = self.analyzer.analyze("a")
        self.assertIn("b.py", result["files_affected"])

    def test_nonexistent_node(self):
        result = self.analyzer.analyze("nonexistent")
        self.assertEqual(len(result["affected"]), 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Implement analyzer.py**

```python
"""src/telos/impact/analyzer.py — transitive impact analysis with risk scoring."""

from __future__ import annotations

from collections import deque
from typing import Any

from telos.code_parser.store import GraphStore


class ImpactAnalyzer:
    """Trace transitive impact of changing a code node."""

    def __init__(self, store: GraphStore) -> None:
        self.store = store

    def analyze(
        self,
        target: str,
        max_depth: int | None = None,
        min_risk: float = 0.0,
    ) -> dict[str, Any]:
        node = self.store.get_node(target)
        if node is None:
            return {"target": target, "affected": [], "hottest_path": [], "files_affected": []}

        # BFS with risk propagation.
        affected: list[dict[str, Any]] = []
        visited: dict[str, float] = {}  # node_id → risk
        parent_map: dict[str, str] = {}  # node_id → parent_id (for path reconstruction)
        queue: deque[tuple[str, float, int]] = deque()
        queue.append((target, 1.0, 0))
        visited[target] = 1.0

        while queue:
            current, risk, depth = queue.popleft()
            edges = self.store.get_edges_from(current)

            for edge in edges:
                next_node = edge["target"]
                next_risk = risk * edge["weight"]

                if max_depth is not None and depth + 1 > max_depth:
                    continue
                if next_risk < min_risk:
                    continue
                if next_node in visited and visited[next_node] >= next_risk:
                    continue

                visited[next_node] = next_risk
                parent_map[next_node] = current
                queue.append((next_node, next_risk, depth + 1))

                next_node_data = self.store.get_node(next_node)
                affected.append({
                    "node_id": next_node,
                    "risk": next_risk,
                    "depth": depth + 1,
                    "edge_kind": edge["kind"],
                    "file_path": next_node_data["file_path"] if next_node_data else edge.get("file_path", ""),
                    "via": current,
                })

        # Deduplicate (keep highest risk per node).
        seen: dict[str, dict] = {}
        for item in affected:
            nid = item["node_id"]
            if nid not in seen or item["risk"] > seen[nid]["risk"]:
                seen[nid] = item
        affected = sorted(seen.values(), key=lambda x: (-x["risk"], x["depth"]))

        # Hottest path: trace back from the deepest high-risk node.
        hottest_path = self._find_hottest_path(target, affected, parent_map)

        # Affected files.
        files = sorted(set(a["file_path"] for a in affected if a["file_path"]))

        return {
            "target": target,
            "affected": affected,
            "hottest_path": hottest_path,
            "files_affected": files,
        }

    def _find_hottest_path(
        self, target: str, affected: list[dict], parent_map: dict[str, str],
    ) -> list[str]:
        if not affected:
            return [target]
        # Find the deepest node with highest risk.
        deepest = max(affected, key=lambda x: (x["depth"], x["risk"]))
        path = [deepest["node_id"]]
        current = deepest["node_id"]
        while current in parent_map:
            current = parent_map[current]
            path.append(current)
        path.reverse()
        return path

    def hotspots(self, top_n: int = 10) -> list[dict[str, Any]]:
        """Return the most depended-on nodes."""
        nodes = self.store.get_all_nodes()
        results: list[dict[str, Any]] = []
        for node in nodes:
            edges_to = self.store.get_edges_to(node["id"])
            if edges_to:
                results.append({
                    "node_id": node["id"],
                    "name": node["name"],
                    "file_path": node["file_path"],
                    "dependent_count": len(edges_to),
                })
        results.sort(key=lambda x: -x["dependent_count"])
        return results[:top_n]
```

- [ ] **Step 3: Run tests**

```bash
mkdir -p tests/test_impact
touch tests/test_impact/__init__.py
PYTHONPATH=src python3.11 -m pytest tests/test_impact/test_analyzer.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/telos/impact/analyzer.py tests/test_impact/
git commit -m "feat: add impact analyzer (transitive traversal + risk scoring)"
```

---

### Task 9: Counterfactual Engine

**Files:**
- Create: `src/telos/impact/counterfactual.py`
- Create: `tests/test_impact/test_counterfactual.py`

- [ ] **Step 1: Write failing tests**

```python
"""tests/test_impact/test_counterfactual.py"""
import os
import tempfile
import unittest

from telos.code_parser.store import GraphStore
from telos.impact.analyzer import ImpactAnalyzer
from telos.impact.counterfactual import CounterfactualAnalyzer


class TestCounterfactualAnalyzer(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, ".telos", "graph.db")
        self.store = GraphStore(self.db_path)
        self._build_test_graph()
        self.analyzer = ImpactAnalyzer(self.store)
        self.cf = CounterfactualAnalyzer(self.store, self.analyzer)

    def _build_test_graph(self):
        for n in ["a", "b", "c", "d", "e"]:
            self.store.add_node(id=n, file_path=f"{n}.py", name=n, kind="function", language="python")
        self.store.add_edge(source="a", target="b", kind="CALLS", weight=1.0, file_path="a.py", line=1)
        self.store.add_edge(source="b", target="c", kind="CALLS", weight=1.0, file_path="b.py", line=1)
        self.store.add_edge(source="c", target="d", kind="CALLS", weight=1.0, file_path="c.py", line=1)
        self.store.add_edge(source="e", target="c", kind="CALLS", weight=1.0, file_path="e.py", line=1)

    def tearDown(self):
        self.store.close()

    def test_intervention_reduces_blast_radius(self):
        result = self.cf.analyze("a", intervention_at="b")
        self.assertLess(
            len(result["with_fix"]["affected"]),
            len(result["without_fix"]["affected"]),
        )

    def test_intervention_report_has_reduction(self):
        result = self.cf.analyze("a", intervention_at="b")
        self.assertIn("reduction", result)
        self.assertGreater(result["reduction"], 0)

    def test_intervention_at_leaf_no_change(self):
        result = self.cf.analyze("a", intervention_at="d")
        self.assertEqual(
            len(result["with_fix"]["affected"]),
            len(result["without_fix"]["affected"]),
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Implement counterfactual.py**

```python
"""src/telos/impact/counterfactual.py — do-interventions on the code dependency graph."""

from __future__ import annotations

from typing import Any

from telos.code_parser.store import GraphStore
from .analyzer import ImpactAnalyzer


class CounterfactualAnalyzer:
    """Apply Pearl's do-operator to the code graph and compare impact."""

    def __init__(self, store: GraphStore, analyzer: ImpactAnalyzer) -> None:
        self.store = store
        self.analyzer = analyzer

    def analyze(
        self,
        target: str,
        intervention_at: str,
    ) -> dict[str, Any]:
        # Without fix: normal impact analysis.
        without = self.analyzer.analyze(target)

        # With fix: temporarily remove outgoing edges from intervention point.
        # We do this by analyzing with the intervention node as a "wall" — 
        # filter the affected list to only nodes NOT reachable through the intervention.
        with_fix_affected = [
            a for a in without["affected"]
            if not self._is_reachable_only_through(target, a["node_id"], intervention_at)
        ]

        without_count = len(without["affected"])
        with_count = len(with_fix_affected)

        with_fix_result = dict(without)
        with_fix_result["affected"] = with_fix_affected
        with_fix_result["files_affected"] = sorted(set(
            a["file_path"] for a in with_fix_affected if a["file_path"]
        ))

        reduction = without_count - with_count

        return {
            "target": target,
            "intervention_at": intervention_at,
            "without_fix": without,
            "with_fix": with_fix_result,
            "reduction": reduction,
            "without_count": without_count,
            "with_count": with_count,
        }

    def _is_reachable_only_through(
        self, start: str, end: str, wall: str,
    ) -> bool:
        """Check if end is only reachable from start through wall.

        If removing wall from the graph makes end unreachable from start,
        then the cascade stops at wall.
        """
        # BFS from start to end, skipping outgoing edges from wall.
        from collections import deque
        visited: set[str] = set()
        queue: deque[str] = deque([start])
        visited.add(start)

        while queue:
            current = queue.popleft()
            if current == end:
                return False  # Reachable without going through wall's outgoing edges? No intervention needed.
            if current == wall:
                continue  # Don't traverse outgoing edges from the wall.
            edges = self.store.get_edges_from(current)
            for edge in edges:
                next_node = edge["target"]
                if next_node not in visited:
                    visited.add(next_node)
                    queue.append(next_node)

        return True  # Not reachable when wall is in place → intervention works.
```

- [ ] **Step 3: Run tests**

```bash
PYTHONPATH=src python3.11 -m pytest tests/test_impact/test_counterfactual.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/telos/impact/counterfactual.py tests/test_impact/test_counterfactual.py
git commit -m "feat: add counterfactual analyzer (do-interventions on code graph)"
```

---

### Task 10: Rich Terminal Reporter

**Files:**
- Create: `src/telos/impact/reporter.py`
- Create: `tests/test_impact/test_reporter.py`

- [ ] **Step 1: Write failing tests**

```python
"""tests/test_impact/test_reporter.py"""
import unittest

from telos.impact.reporter import format_impact, format_hotspots, format_counterfactual, format_info


class TestReporter(unittest.TestCase):

    def test_format_impact_produces_string(self):
        result = {
            "target": "auth.py:validate",
            "affected": [
                {"node_id": "api.py:handler", "risk": 1.0, "depth": 1, "edge_kind": "CALLS", "file_path": "api.py", "via": "auth.py:validate"},
            ],
            "hottest_path": ["auth.py:validate", "api.py:handler"],
            "files_affected": ["api.py"],
        }
        output = format_impact(result)
        self.assertIn("auth.py:validate", output)
        self.assertIn("api.py:handler", output)

    def test_format_hotspots_produces_string(self):
        hotspots = [
            {"node_id": "auth.py:validate", "name": "validate", "file_path": "auth.py", "dependent_count": 23},
        ]
        output = format_hotspots(hotspots)
        self.assertIn("validate", output)
        self.assertIn("23", output)

    def test_format_counterfactual_produces_string(self):
        result = {
            "target": "a",
            "intervention_at": "b",
            "without_count": 10,
            "with_count": 3,
            "reduction": 7,
            "without_fix": {"affected": [{"node_id": f"n{i}", "risk": 0.5, "depth": 1, "edge_kind": "CALLS", "file_path": "x.py", "via": "a"} for i in range(10)]},
            "with_fix": {"affected": [{"node_id": f"n{i}", "risk": 0.5, "depth": 1, "edge_kind": "CALLS", "file_path": "x.py", "via": "a"} for i in range(3)]},
        }
        output = format_counterfactual(result)
        self.assertIn("10", output)
        self.assertIn("3", output)

    def test_format_info_produces_string(self):
        info = {
            "repo_root": "/home/dev/project",
            "last_scan": "2026-04-16",
            "node_count": 100,
            "edge_count": 500,
            "db_size": "1.2 MB",
        }
        output = format_info(info)
        self.assertIn("100", output)
        self.assertIn("500", output)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Implement reporter.py**

```python
"""src/telos/impact/reporter.py — rich terminal output formatting."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.tree import Tree
from rich.table import Table
from rich.panel import Panel
from rich.text import Text


def _risk_color(risk: float) -> str:
    if risk >= 0.9:
        return "red"
    elif risk >= 0.6:
        return "yellow"
    return "green"


def _risk_label(risk: float) -> str:
    if risk >= 0.9:
        return "HIGH"
    elif risk >= 0.6:
        return "MEDIUM"
    return "LOW"


def format_impact(result: dict[str, Any]) -> str:
    console = Console(record=True, width=100)
    target = result["target"]
    affected = result["affected"]

    console.print()
    console.print(f"Impact Analysis: [bold]{target}[/bold]", style="bold white")
    console.print("=" * 60)

    if not affected:
        console.print("[green]No downstream dependencies found.[/green]")
        return console.export_text()

    console.print(f"\nCausal chain ({len(affected)} nodes affected):\n")

    tree = Tree(f"[bold]{target}[/bold]")
    # Group by depth for tree display.
    by_via: dict[str, list] = {}
    for a in affected:
        via = a.get("via", target)
        by_via.setdefault(via, []).append(a)

    def _build_tree(parent_tree, parent_id, depth=0, max_depth=5):
        if depth > max_depth:
            return
        children = by_via.get(parent_id, [])
        for child in children:
            color = _risk_color(child["risk"])
            label = f"{child['edge_kind']} → [bold]{child['node_id']}[/bold] [{color}]risk: {child['risk']:.1f}[/{color}]"
            branch = parent_tree.add(label)
            _build_tree(branch, child["node_id"], depth + 1, max_depth)

    _build_tree(tree, target)
    console.print(tree)

    # Hottest path.
    path = result.get("hottest_path", [])
    if len(path) > 1:
        console.print(f"\nHottest path: [bold]{' → '.join(path)}[/bold]")

    # Files affected.
    files = result.get("files_affected", [])
    if files:
        console.print(f"\nFiles affected: {len(files)}")
        for f in files:
            console.print(f"  {f}")

    return console.export_text()


def format_hotspots(hotspots: list[dict[str, Any]]) -> str:
    console = Console(record=True, width=100)
    console.print()
    console.print("Dependency Hotspots", style="bold white")
    console.print("=" * 40)

    table = Table()
    table.add_column("#", style="dim")
    table.add_column("Function", style="bold")
    table.add_column("File")
    table.add_column("Dependents", justify="right")
    table.add_column("Risk", justify="center")

    for i, h in enumerate(hotspots, 1):
        count = h["dependent_count"]
        risk = _risk_label(min(count / 20.0, 1.0))
        color = _risk_color(min(count / 20.0, 1.0))
        table.add_row(str(i), h["name"], h["file_path"], str(count), f"[{color}]{risk}[/{color}]")

    console.print(table)
    return console.export_text()


def format_counterfactual(result: dict[str, Any]) -> str:
    console = Console(record=True, width=100)
    console.print()
    console.print(f"Impact Analysis: [bold]{result['target']}[/bold] (with intervention)", style="bold white")
    console.print("=" * 60)

    console.print(f"\nWithout fix: [red]{result['without_count']} nodes affected[/red]")
    console.print(f"With fix (intervention at {result['intervention_at']}): [green]{result['with_count']} nodes affected[/green]")

    reduction = result["reduction"]
    if result["without_count"] > 0:
        pct = reduction / result["without_count"] * 100
        console.print(f"\nBlast radius reduced: {result['without_count']} → {result['with_count']} ({pct:.0f}% reduction)")
    else:
        console.print("\nNo impact to reduce.")

    return console.export_text()


def format_info(info: dict[str, Any]) -> str:
    console = Console(record=True, width=100)
    console.print()
    console.print("Telos Graph Info", style="bold white")
    console.print("=" * 30)

    for key, value in info.items():
        label = key.replace("_", " ").title()
        console.print(f"  {label}: [bold]{value}[/bold]")

    return console.export_text()
```

- [ ] **Step 3: Run tests**

```bash
PYTHONPATH=src python3.11 -m pytest tests/test_impact/test_reporter.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/telos/impact/reporter.py tests/test_impact/test_reporter.py
git commit -m "feat: add rich terminal reporter"
```

---

### Task 11: CLI Application

**Files:**
- Create: `src/telos/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

```python
"""tests/test_cli.py"""
import os
import tempfile
import unittest
from unittest.mock import patch

from typer.testing import CliRunner

from telos.cli import app


class TestCLI(unittest.TestCase):

    def setUp(self):
        self.runner = CliRunner()
        self.tmpdir = tempfile.mkdtemp()
        # Create a simple Python file.
        with open(os.path.join(self.tmpdir, "example.py"), "w") as f:
            f.write("def hello():\n    pass\n\ndef world():\n    hello()\n")

    def test_init_creates_db(self):
        result = self.runner.invoke(app, ["init", "--path", self.tmpdir])
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, ".telos", "graph.db")))

    def test_init_shows_summary(self):
        result = self.runner.invoke(app, ["init", "--path", self.tmpdir])
        self.assertIn("Scanning", result.output)

    def test_info_after_init(self):
        self.runner.invoke(app, ["init", "--path", self.tmpdir])
        result = self.runner.invoke(app, ["info", "--path", self.tmpdir])
        self.assertEqual(result.exit_code, 0)

    def test_hotspots_after_init(self):
        self.runner.invoke(app, ["init", "--path", self.tmpdir])
        result = self.runner.invoke(app, ["hotspots", "--path", self.tmpdir])
        self.assertEqual(result.exit_code, 0)

    def test_impact_after_init(self):
        self.runner.invoke(app, ["init", "--path", self.tmpdir])
        result = self.runner.invoke(app, ["impact", "example.py:hello", "--path", self.tmpdir])
        self.assertEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Implement cli.py**

```python
"""src/telos/cli.py — typer CLI for telos causal impact analyzer."""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Optional

import typer
from rich.console import Console

from telos.code_parser.store import GraphStore
from telos.code_parser.graph_builder import GraphBuilder
from telos.impact.analyzer import ImpactAnalyzer
from telos.impact.counterfactual import CounterfactualAnalyzer
from telos.impact.reporter import format_impact, format_hotspots, format_counterfactual, format_info

app = typer.Typer(help="telos — causal impact analyzer for codebases")
console = Console()


def _get_db_path(path: str) -> str:
    return os.path.join(path, ".telos", "graph.db")


def _require_init(path: str) -> GraphStore:
    db_path = _get_db_path(path)
    if not os.path.exists(db_path):
        console.print("[red]Error: telos not initialized. Run 'telos init' first.[/red]")
        raise typer.Exit(1)
    return GraphStore(db_path)


@app.command()
def init(
    path: str = typer.Option(".", help="Repository root path"),
    force: bool = typer.Option(False, help="Rebuild graph from scratch"),
) -> None:
    """Scan repository and build dependency graph."""
    path = os.path.abspath(path)
    db_path = _get_db_path(path)

    console.print("\n[bold]Scanning repository...[/bold]")

    store = GraphStore(db_path)
    if force:
        store.clear()

    builder = GraphBuilder(store)
    start = time.time()
    stats = builder.scan_directory(path)
    elapsed = time.time() - start

    store.set_meta("last_scan", datetime.now().isoformat())
    store.set_meta("repo_root", path)

    console.print(f"  Files scanned: {stats['files_scanned']}")
    console.print(f"  Parsing time: {elapsed:.1f}s")
    console.print(f"\n[bold]Building dependency graph...[/bold]")
    console.print(f"  Nodes: {stats['nodes']}")
    console.print(f"  Edges: {stats['edges']}")

    db_size = os.path.getsize(db_path)
    size_str = f"{db_size / 1024:.1f} KB" if db_size < 1024 * 1024 else f"{db_size / 1024 / 1024:.1f} MB"
    console.print(f"\n  Stored graph: {db_path} ({size_str})")

    # Show hotspots.
    analyzer = ImpactAnalyzer(store)
    hotspots = analyzer.hotspots(top_n=5)
    if hotspots:
        console.print("\n[bold]Hotspots (most depended-on):[/bold]")
        for i, h in enumerate(hotspots, 1):
            console.print(f"  {i}. {h['node_id']} → {h['dependent_count']} dependents")

    console.print(f"\n[green]Ready. Run: telos impact <file:function> to trace impact.[/green]")
    store.close()


@app.command()
def impact(
    target: str = typer.Argument(help="Target node (file.py:function)"),
    path: str = typer.Option(".", help="Repository root path"),
    fix: Optional[str] = typer.Option(None, help="Intervention point for counterfactual"),
    depth: Optional[int] = typer.Option(None, help="Max traversal depth"),
    min_risk: float = typer.Option(0.0, help="Minimum risk threshold"),
) -> None:
    """Trace causal impact of changing a target."""
    path = os.path.abspath(path)
    store = _require_init(path)
    analyzer = ImpactAnalyzer(store)

    if fix:
        cf = CounterfactualAnalyzer(store, analyzer)
        result = cf.analyze(target, intervention_at=fix)
        output = format_counterfactual(result)
    else:
        result = analyzer.analyze(target, max_depth=depth, min_risk=min_risk)
        output = format_impact(result)

    console.print(output)
    store.close()


@app.command()
def hotspots(
    path: str = typer.Option(".", help="Repository root path"),
    top: int = typer.Option(10, help="Number of hotspots to show"),
) -> None:
    """Show most depended-on code nodes."""
    path = os.path.abspath(path)
    store = _require_init(path)
    analyzer = ImpactAnalyzer(store)
    results = analyzer.hotspots(top_n=top)
    output = format_hotspots(results)
    console.print(output)
    store.close()


@app.command()
def graph(
    target: str = typer.Argument(help="Target node (file.py:function)"),
    path: str = typer.Option(".", help="Repository root path"),
    depth: int = typer.Option(3, help="Max traversal depth"),
) -> None:
    """Show dependency subgraph around a target."""
    path = os.path.abspath(path)
    store = _require_init(path)
    analyzer = ImpactAnalyzer(store)
    result = analyzer.analyze(target, max_depth=depth)
    output = format_impact(result)
    console.print(output)
    store.close()


@app.command()
def info(
    path: str = typer.Option(".", help="Repository root path"),
) -> None:
    """Show graph statistics."""
    path = os.path.abspath(path)
    store = _require_init(path)
    stats = store.get_stats()
    info_dict = {
        "repo_root": store.get_meta("repo_root") or path,
        "last_scan": store.get_meta("last_scan") or "never",
        "node_count": stats["node_count"],
        "edge_count": stats["edge_count"],
        "db_path": _get_db_path(path),
    }
    db_path = _get_db_path(path)
    if os.path.exists(db_path):
        size = os.path.getsize(db_path)
        info_dict["db_size"] = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / 1024 / 1024:.1f} MB"
    output = format_info(info_dict)
    console.print(output)
    store.close()


if __name__ == "__main__":
    app()
```

- [ ] **Step 3: Run tests**

```bash
PYTHONPATH=src python3.11 -m pytest tests/test_cli.py -v
```

- [ ] **Step 4: Test CLI manually**

```bash
pip install -e .
telos init --path .
telos info --path .
telos hotspots --path .
telos impact src/telos/causal_graph.py:CausalGraph --path .
```

- [ ] **Step 5: Commit**

```bash
git add src/telos/cli.py tests/test_cli.py
git commit -m "feat: add telos CLI (init, impact, hotspots, graph, info)"
```

---

### Task 12: Final Verification and Push

- [ ] **Step 1: Run full test suite**

```bash
PYTHONPATH=src python3.11 -m pytest tests/ -v
```

- [ ] **Step 2: Test CLI end-to-end on the telos repo itself**

```bash
telos init
telos info
telos hotspots
telos impact src/telos/causal_graph.py:CausalGraph
```

- [ ] **Step 3: Update README**

Add a "Product" section describing the CLI tool and its commands.

- [ ] **Step 4: Commit and push**

```bash
git add .
git commit -m "feat: telos Phase 1 complete — causal impact analyzer CLI"
git push origin main
```
