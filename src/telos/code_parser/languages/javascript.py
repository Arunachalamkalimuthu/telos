"""JavaScript-specific tree-sitter extractor."""

from __future__ import annotations

from tree_sitter import Tree


class JavaScriptExtractor:
    """Extract functions, classes, imports, and calls from a JavaScript AST."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _walk(node, node_type: str):
        """Yield every descendant (including *node* itself) whose type matches."""
        if node.type == node_type:
            yield node
        for child in node.children:
            yield from JavaScriptExtractor._walk(child, node_type)

    @staticmethod
    def _walk_multi(node, node_types: set[str]):
        """Yield every descendant whose type is in *node_types*."""
        if node.type in node_types:
            yield node
        for child in node.children:
            yield from JavaScriptExtractor._walk_multi(child, node_types)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_functions(self, tree: Tree, file_path: str) -> list[dict]:
        """Return a record for every function declaration (including arrow functions) in the tree."""
        results: list[dict] = []

        # Named function declarations: function hello(name) { ... }
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

        # Arrow functions assigned to variables:
        # const greet = (name) => { ... }
        # let foo = () => ...
        for decl_node in self._walk_multi(
            tree.root_node,
            {"lexical_declaration", "variable_declaration"},
        ):
            for declarator in self._walk(decl_node, "variable_declarator"):
                value_node = declarator.child_by_field_name("value")
                if value_node is None or value_node.type != "arrow_function":
                    continue
                name_node = declarator.child_by_field_name("name")
                if name_node is None:
                    continue
                results.append(
                    {
                        "name": name_node.text.decode(),
                        "kind": "function",
                        "file_path": file_path,
                        "line_start": decl_node.start_point[0] + 1,
                        "line_end": decl_node.end_point[0] + 1,
                    }
                )

        return results

    def extract_classes(self, tree: Tree, file_path: str) -> list[dict]:
        """Return a record for every class_declaration in the tree."""
        results: list[dict] = []
        for node in self._walk(tree.root_node, "class_declaration"):
            name_node = node.child_by_field_name("name")
            if name_node is None:
                continue

            # Collect base class names from class_heritage (extends clause).
            bases: list[str] = []
            for child in node.children:
                if child.type == "class_heritage":
                    # class_heritage contains the expression after "extends"
                    for heritage_child in child.named_children:
                        bases.append(heritage_child.text.decode())

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
        return results

    def extract_imports(self, tree: Tree, file_path: str) -> list[dict]:
        """Return a record for every import_statement in the tree."""
        results: list[dict] = []

        for node in self._walk(tree.root_node, "import_statement"):
            source_node = node.child_by_field_name("source")
            if source_node is None:
                continue
            # Strip surrounding quotes from the module string.
            module = source_node.text.decode().strip("'\"")
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

            if callee.type == "member_expression":
                # obj.method(...)  →  name = "method"
                prop_node = callee.child_by_field_name("property")
                name = prop_node.text.decode() if prop_node is not None else full_name
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
