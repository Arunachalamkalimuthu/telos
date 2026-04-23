"""Rust-specific tree-sitter extractor."""

from __future__ import annotations

from tree_sitter import Tree


class RustExtractor:
    """Extract functions, classes (structs/enums/impls), imports, and calls from a Rust AST."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _walk(node, node_type: str):
        """Yield every descendant (including *node* itself) whose type matches."""
        if node.type == node_type:
            yield node
        for child in node.children:
            yield from RustExtractor._walk(child, node_type)

    @staticmethod
    def _walk_multi(node, node_types: set[str]):
        """Yield every descendant whose type is in *node_types*."""
        if node.type in node_types:
            yield node
        for child in node.children:
            yield from RustExtractor._walk_multi(child, node_types)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_functions(self, tree: Tree, file_path: str) -> list[dict]:
        """Return a record for every function_item in the tree."""
        results: list[dict] = []

        for node in self._walk(tree.root_node, "function_item"):
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

        return results

    def extract_classes(self, tree: Tree, file_path: str) -> list[dict]:
        """Return records for structs, enums, and impl blocks."""
        results: list[dict] = []

        # struct_item  →  struct Point { ... }
        for node in self._walk(tree.root_node, "struct_item"):
            name_node = node.child_by_field_name("name")
            if name_node is None:
                continue
            results.append(
                {
                    "name": name_node.text.decode(),
                    "kind": "struct",
                    "file_path": file_path,
                    "line_start": node.start_point[0] + 1,
                    "line_end": node.end_point[0] + 1,
                    "bases": [],
                }
            )

        # enum_item  →  enum Color { ... }
        for node in self._walk(tree.root_node, "enum_item"):
            name_node = node.child_by_field_name("name")
            if name_node is None:
                continue
            results.append(
                {
                    "name": name_node.text.decode(),
                    "kind": "enum",
                    "file_path": file_path,
                    "line_start": node.start_point[0] + 1,
                    "line_end": node.end_point[0] + 1,
                    "bases": [],
                }
            )

        # impl_item  →  impl Point { ... }
        for node in self._walk(tree.root_node, "impl_item"):
            type_node = node.child_by_field_name("type")
            if type_node is None:
                continue
            results.append(
                {
                    "name": type_node.text.decode(),
                    "kind": "impl",
                    "file_path": file_path,
                    "line_start": node.start_point[0] + 1,
                    "line_end": node.end_point[0] + 1,
                    "bases": [],
                }
            )

        return results

    def extract_imports(self, tree: Tree, file_path: str) -> list[dict]:
        """Return a record for every use_declaration in the tree."""
        results: list[dict] = []

        for node in self._walk(tree.root_node, "use_declaration"):
            # The argument field holds the use path (scoped_identifier,
            # use_wildcard, use_list, etc.).
            arg = node.child_by_field_name("argument")
            if arg is None:
                continue
            module = arg.text.decode()
            results.append(
                {
                    "module": module,
                    "file_path": file_path,
                    "line": node.start_point[0] + 1,
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

            if callee.type == "scoped_identifier":
                # Point::new(...)  →  name = "new"
                name_node = callee.child_by_field_name("name")
                name = name_node.text.decode() if name_node is not None else full_name
            elif callee.type == "field_expression":
                # obj.method(...)  →  name = "method"
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
