"""Go-specific tree-sitter extractor."""

from __future__ import annotations

from tree_sitter import Tree


class GoExtractor:
    """Extract functions, classes (structs), imports, and calls from a Go AST."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _walk(node, node_type: str):
        """Yield every descendant (including *node* itself) whose type matches."""
        if node.type == node_type:
            yield node
        for child in node.children:
            yield from GoExtractor._walk(child, node_type)

    @staticmethod
    def _walk_multi(node, node_types: set[str]):
        """Yield every descendant whose type is in *node_types*."""
        if node.type in node_types:
            yield node
        for child in node.children:
            yield from GoExtractor._walk_multi(child, node_types)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_functions(self, tree: Tree, file_path: str) -> list[dict]:
        """Return a record for every function/method declaration in the tree."""
        results: list[dict] = []

        # function_declaration  →  func hello(...)
        for node in self._walk(tree.root_node, "function_declaration"):
            name_node = node.child_by_field_name("name")
            if name_node is None:
                continue
            results.append(
                {
                    "name": name_node.text.decode(),
                    "kind": "function",
                    "file_path": file_path,
                    "line_start": node.start_point[0] + 1,
                    "line_end": node.end_point[0] + 1,
                }
            )

        # method_declaration  →  func (p *Point) Move(...)
        for node in self._walk(tree.root_node, "method_declaration"):
            name_node = node.child_by_field_name("name")
            if name_node is None:
                continue
            results.append(
                {
                    "name": name_node.text.decode(),
                    "kind": "method",
                    "file_path": file_path,
                    "line_start": node.start_point[0] + 1,
                    "line_end": node.end_point[0] + 1,
                }
            )

        return results

    def extract_classes(self, tree: Tree, file_path: str) -> list[dict]:
        """Return a record for every struct type declaration (treated as a class)."""
        results: list[dict] = []

        # type_declaration  →  type Point struct { ... }
        # Contains a type_spec child whose child is a struct_type.
        for td_node in self._walk(tree.root_node, "type_declaration"):
            for ts_node in self._walk(td_node, "type_spec"):
                # Check if this type_spec contains a struct_type.
                has_struct = any(
                    c.type == "struct_type" for c in ts_node.named_children
                )
                if not has_struct:
                    continue
                name_node = ts_node.child_by_field_name("name")
                if name_node is None:
                    continue
                results.append(
                    {
                        "name": name_node.text.decode(),
                        "kind": "struct",
                        "file_path": file_path,
                        "line_start": td_node.start_point[0] + 1,
                        "line_end": td_node.end_point[0] + 1,
                        "bases": [],
                    }
                )

        return results

    def extract_imports(self, tree: Tree, file_path: str) -> list[dict]:
        """Return a record for every import path in the tree."""
        results: list[dict] = []

        for node in self._walk(tree.root_node, "import_declaration"):
            for spec in self._walk(node, "import_spec"):
                path_node = spec.child_by_field_name("path")
                if path_node is None:
                    continue
                # Strip surrounding double-quotes from the module path.
                module = path_node.text.decode().strip('"')
                results.append(
                    {
                        "module": module,
                        "file_path": file_path,
                        "line": spec.start_point[0] + 1,
                    }
                )

        return results

    def extract_calls(self, tree: Tree, file_path: str) -> list[dict]:
        """Return a record for every call expression in the tree."""
        results: list[dict] = []

        for node in self._walk(tree.root_node, "call_expression"):
            callee = node.child_by_field_name("function")
            if callee is None:
                continue

            full_name = callee.text.decode()

            if callee.type == "selector_expression":
                # fmt.Println(...)  →  name = "Println"
                field_node = callee.child_by_field_name("field")
                name = field_node.text.decode() if field_node is not None else full_name
            elif callee.type == "identifier":
                name = full_name
            else:
                name = full_name

            results.append(
                {
                    "name": name,
                    "full_name": full_name,
                    "file_path": file_path,
                    "line": node.start_point[0] + 1,
                }
            )

        return results
