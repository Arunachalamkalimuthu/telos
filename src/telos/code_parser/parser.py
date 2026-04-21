"""Multi-language tree-sitter parser orchestrator."""

from __future__ import annotations

import os

from tree_sitter import Language, Parser, Tree
import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_typescript
import tree_sitter_go
import tree_sitter_java
import tree_sitter_rust

# Module-level cache of Language objects keyed by language name.
_GRAMMAR_MAP: dict[str, Language] = {
    "python":     Language(tree_sitter_python.language()),
    "javascript": Language(tree_sitter_javascript.language()),
    "typescript": Language(tree_sitter_typescript.language_typescript()),
    "tsx":        Language(tree_sitter_typescript.language_tsx()),
    "go":         Language(tree_sitter_go.language()),
    "java":       Language(tree_sitter_java.language()),
    "rust":       Language(tree_sitter_rust.language()),
}

_EXTENSION_MAP: dict[str, str] = {
    ".py":  "python",
    ".js":  "javascript",
    ".jsx": "javascript",
    ".ts":  "typescript",
    ".tsx": "tsx",
    ".go":  "go",
    ".java": "java",
    ".rs":  "rust",
}


class TelosParser:
    """Orchestrates tree-sitter parsing across multiple languages."""

    @staticmethod
    def detect_language(file_path: str) -> str | None:
        """Return the language name for *file_path*, or None if unsupported."""
        _, ext = os.path.splitext(file_path)
        return _EXTENSION_MAP.get(ext.lower())

    @staticmethod
    def supported_languages() -> list[str]:
        """Return the list of supported language names."""
        return ["python", "javascript", "tsx", "typescript", "go", "java", "rust"]

    def parse(self, source: bytes, language: str) -> Tree:
        """Parse *source* bytes using the grammar for *language* and return the Tree."""
        lang = _GRAMMAR_MAP[language]
        parser = Parser(lang)
        return parser.parse(source)
