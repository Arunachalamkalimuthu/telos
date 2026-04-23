"""Java-specific tree-sitter extractor."""

from __future__ import annotations

from tree_sitter import Tree


class JavaExtractor:
    """Extract functions (methods), classes, imports, and calls from a Java AST."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _walk(node, node_type: str):
        """Yield every descendant (including *node* itself) whose type matches."""
        if node.type == node_type:
            yield node
        for child in node.children:
            yield from JavaExtractor._walk(child, node_type)

    @staticmethod
    def _walk_multi(node, node_types: set[str]):
        """Yield every descendant whose type is in *node_types*."""
        if node.type in node_types:
            yield node
        for child in node.children:
            yield from JavaExtractor._walk_multi(child, node_types)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_functions(self, tree: Tree, file_path: str) -> list[dict]:
        """Return a record for every method_declaration in the tree."""
        results: list[dict] = []

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
        """Return a record for every class and interface declaration."""
        results: list[dict] = []

        for node in self._walk(tree.root_node, "class_declaration"):
            name_node = node.child_by_field_name("name")
            if name_node is None:
                continue

            # Collect base class / implemented interfaces.
            bases: list[str] = []
            for child in node.children:
                if child.type == "superclass":
                    # superclass contains a type_identifier
                    for nc in child.named_children:
                        bases.append(nc.text.decode())
                elif child.type == "super_interfaces":
                    for nc in child.named_children:
                        # type_list contains type_identifiers
                        if nc.type == "type_list":
                            for ti in nc.named_children:
                                bases.append(ti.text.decode())
                        else:
                            bases.append(nc.text.decode())

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

        return results

    def extract_imports(self, tree: Tree, file_path: str) -> list[dict]:
        """Return a record for every import_declaration in the tree."""
        results: list[dict] = []

        for node in self._walk(tree.root_node, "import_declaration"):
            # The import path lives in a scoped_identifier child.
            module = ""
            for child in node.named_children:
                if child.type == "scoped_identifier":
                    module = child.text.decode()
                    break
                elif child.type == "identifier":
                    module = child.text.decode()
                    break

            if module:
                results.append(
                    {
                        "module": module,
                        "file_path": file_path,
                        "line": node.start_point[0] + 1,
                    }
                )

        return results

    def extract_calls(self, tree: Tree, file_path: str) -> list[dict]:
        """Return a record for every method invocation in the tree."""
        results: list[dict] = []

        for node in self._walk(tree.root_node, "method_invocation"):
            name_node = node.child_by_field_name("name")
            if name_node is None:
                continue

            name = name_node.text.decode()

            obj_node = node.child_by_field_name("object")
            if obj_node is not None:
                full_name = f"{obj_node.text.decode()}.{name}"
            else:
                full_name = name

            results.append(
                {
                    "name": name,
                    "full_name": full_name,
                    "file_path": file_path,
                    "line": node.start_point[0] + 1,
                }
            )

        return results
