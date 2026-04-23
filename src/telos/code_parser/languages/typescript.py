"""TypeScript-specific tree-sitter extractor.

Reuses JavaScriptExtractor for the common JS/TS AST, then adds handling
for ``interface_declaration`` and ``type_alias_declaration``.
"""

from __future__ import annotations

from tree_sitter import Tree

from telos.code_parser.languages.javascript import JavaScriptExtractor


class TypeScriptExtractor(JavaScriptExtractor):
    """Extract functions, classes, imports, and calls from a TypeScript AST.

    TypeScript's tree-sitter grammar is a superset of JavaScript's, so we
    inherit all four extraction methods and extend ``extract_classes`` to
    cover interfaces and type aliases.
    """

    def extract_classes(self, tree: Tree, file_path: str) -> list[dict]:
        """Return records for classes, interfaces, and type aliases.

        TypeScript's ``class_heritage`` wraps the base class inside an
        ``extends_clause`` node, unlike JavaScript where the identifier is
        a direct child of ``class_heritage``.  We therefore re-implement the
        class-extraction logic rather than delegating to the parent.
        """
        results: list[dict] = []

        # class_declaration  →  kind="class"
        for node in self._walk(tree.root_node, "class_declaration"):
            name_node = node.child_by_field_name("name")
            if name_node is None:
                continue

            bases: list[str] = []
            for child in node.children:
                if child.type == "class_heritage":
                    # TS: class_heritage → extends_clause → identifier
                    for clause in child.named_children:
                        if clause.type == "extends_clause":
                            for base in clause.named_children:
                                bases.append(base.text.decode())
                        else:
                            # Fallback for unexpected structure.
                            bases.append(clause.text.decode())

            results.append(
                {
                    "name": name_node.text.decode(),
                    "kind": "class",
                    "file_path": file_path,
                    "line_start": node.start_point[0] + 1,
                    "line_end": node.end_point[0] + 1,
                    "bases": bases,
                }
            )

        # interface_declaration  →  kind="interface"
        for node in self._walk(tree.root_node, "interface_declaration"):
            name_node = node.child_by_field_name("name")
            if name_node is None:
                continue
            results.append(
                {
                    "name": name_node.text.decode(),
                    "kind": "interface",
                    "file_path": file_path,
                    "line_start": node.start_point[0] + 1,
                    "line_end": node.end_point[0] + 1,
                    "bases": [],
                }
            )

        # type_alias_declaration  →  kind="type_alias"
        for node in self._walk(tree.root_node, "type_alias_declaration"):
            name_node = node.child_by_field_name("name")
            if name_node is None:
                continue
            results.append(
                {
                    "name": name_node.text.decode(),
                    "kind": "type_alias",
                    "file_path": file_path,
                    "line_start": node.start_point[0] + 1,
                    "line_end": node.end_point[0] + 1,
                    "bases": [],
                }
            )

        return results
