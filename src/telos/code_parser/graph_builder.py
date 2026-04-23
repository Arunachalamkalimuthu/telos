"""Scan source files and populate a GraphStore with nodes and edges."""

from __future__ import annotations

import os

from telos.code_parser.parser import TelosParser
from telos.code_parser.store import GraphStore
from telos.code_parser.languages.python import PythonExtractor
from telos.code_parser.languages.javascript import JavaScriptExtractor
from telos.code_parser.languages.typescript import TypeScriptExtractor
from telos.code_parser.languages.go import GoExtractor
from telos.code_parser.languages.java import JavaExtractor
from telos.code_parser.languages.rust import RustExtractor

# Edge weights from the spec.
EDGE_WEIGHTS = {
    "CALLS": 1.0,
    "DATA_FLOW": 0.8,
    "IMPORTS": 0.6,
    "INHERITS": 0.9,
}

# Map language name to extractor instance.
_EXTRACTOR_MAP = {
    "python": PythonExtractor(),
    "javascript": JavaScriptExtractor(),
    "typescript": TypeScriptExtractor(),
    "tsx": TypeScriptExtractor(),
    "go": GoExtractor(),
    "java": JavaExtractor(),
    "rust": RustExtractor(),
}

_DEFAULT_EXCLUDE_DIRS: set[str] = {
    ".git",
    ".telos",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "target",
    "build",
    "dist",
}


class GraphBuilder:
    """Scans source files and populates a GraphStore with nodes and edges."""

    def __init__(self, store: GraphStore):
        self._store = store
        self._parser = TelosParser()
        # Global symbol table: simple name -> node_id (for call resolution).
        self._symbols: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_file(self, file_path: str, repo_root: str) -> None:
        """Parse *file_path* and add all discovered nodes/edges to the store."""
        language = TelosParser.detect_language(file_path)
        if language is None:
            return

        extractor = _EXTRACTOR_MAP.get(language)
        if extractor is None:
            return

        try:
            with open(file_path, "rb") as fh:
                source = fh.read()
        except (OSError, IOError):
            return

        tree = self._parser.parse(source, language)
        rel_path = os.path.relpath(file_path, repo_root)

        # 1. Module node
        self._store.add_node(
            id=rel_path,
            file_path=rel_path,
            name=rel_path,
            kind="module",
            language=language,
        )

        # 2. Functions
        functions = extractor.extract_functions(tree, rel_path)
        for fn in functions:
            node_id = f"{rel_path}:{fn['name']}"
            self._store.add_node(
                id=node_id,
                file_path=rel_path,
                name=fn["name"],
                kind="function",
                language=language,
                line_start=fn.get("line_start", 0),
                line_end=fn.get("line_end", 0),
            )
            self._symbols[fn["name"]] = node_id

        # 3. Classes
        classes = extractor.extract_classes(tree, rel_path)
        for cls in classes:
            node_id = f"{rel_path}:{cls['name']}"
            self._store.add_node(
                id=node_id,
                file_path=rel_path,
                name=cls["name"],
                kind="class",
                language=language,
                line_start=cls.get("line_start", 0),
                line_end=cls.get("line_end", 0),
            )
            self._symbols[cls["name"]] = node_id

            # INHERITS edges for base classes.
            for base in cls.get("bases", []):
                base_id = self._symbols.get(base, base)
                self._store.add_edge(
                    source=node_id,
                    target=base_id,
                    kind="INHERITS",
                    weight=EDGE_WEIGHTS["INHERITS"],
                    file_path=rel_path,
                    line=cls.get("line_start", 0),
                )

        # 4. Imports
        imports = extractor.extract_imports(tree, rel_path)
        for imp in imports:
            module = imp.get("module", "")
            if module:
                self._store.add_edge(
                    source=rel_path,
                    target=module,
                    kind="IMPORTS",
                    weight=EDGE_WEIGHTS["IMPORTS"],
                    file_path=rel_path,
                    line=imp.get("line", 0),
                )

        # 5. Calls – resolve caller by line range, callee by symbol table.
        # Build a lookup of functions/classes with their line ranges for caller
        # resolution.
        containers: list[dict] = []
        for fn in functions:
            containers.append({
                "id": f"{rel_path}:{fn['name']}",
                "line_start": fn.get("line_start", 0),
                "line_end": fn.get("line_end", 0),
            })
        for cls in classes:
            containers.append({
                "id": f"{rel_path}:{cls['name']}",
                "line_start": cls.get("line_start", 0),
                "line_end": cls.get("line_end", 0),
            })

        calls = extractor.extract_calls(tree, rel_path)
        for call in calls:
            call_line = call.get("line", 0)
            call_name = call.get("name", "")

            # Find the containing function/class for this call.
            caller_id = self._resolve_caller(containers, call_line, rel_path)

            # Resolve callee via symbol table.
            callee_id = self._symbols.get(call_name)
            if callee_id is None:
                continue

            # Don't add self-calls as edges (e.g. recursive definition
            # artifacts) unless caller != callee.
            if caller_id == callee_id:
                continue

            self._store.add_edge(
                source=caller_id,
                target=callee_id,
                kind="CALLS",
                weight=EDGE_WEIGHTS["CALLS"],
                file_path=rel_path,
                line=call_line,
            )

    def scan_directory(
        self,
        root: str,
        exclude_dirs: set[str] | None = None,
    ) -> dict[str, int]:
        """Walk *root*, scan every supported file, return summary stats."""
        if exclude_dirs is None:
            exclude_dirs = _DEFAULT_EXCLUDE_DIRS

        files_scanned = 0

        for dirpath, dirnames, filenames in os.walk(root):
            # Prune excluded directories in-place so os.walk doesn't descend.
            dirnames[:] = [
                d for d in dirnames if d not in exclude_dirs
            ]

            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                language = TelosParser.detect_language(fpath)
                if language is None:
                    continue
                self.scan_file(fpath, root)
                files_scanned += 1

        stats = self._store.get_stats()
        return {
            "files_scanned": files_scanned,
            "nodes": stats["node_count"],
            "edges": stats["edge_count"],
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_caller(
        containers: list[dict],
        line: int,
        rel_path: str,
    ) -> str:
        """Return the node_id of the tightest container enclosing *line*.

        Falls back to the module node (*rel_path*) if none match.
        """
        best: dict | None = None
        for c in containers:
            if c["line_start"] <= line <= c["line_end"]:
                if best is None or (c["line_end"] - c["line_start"]) < (
                    best["line_end"] - best["line_start"]
                ):
                    best = c
        return best["id"] if best else rel_path
