"""Python-specific tree-sitter extractor."""

from __future__ import annotations

from tree_sitter import Tree


class PythonExtractor:
    """Extract functions, classes, imports, and calls from a Python AST."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _walk(node, node_type: str):
        """Yield every descendant (including *node* itself) whose type matches."""
        if node.type == node_type:
            yield node
        for child in node.children:
            yield from PythonExtractor._walk(child, node_type)

    @staticmethod
    def _walk_multi(node, node_types: set[str]):
        """Yield every descendant whose type is in *node_types*."""
        if node.type in node_types:
            yield node
        for child in node.children:
            yield from PythonExtractor._walk_multi(child, node_types)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_functions(self, tree: Tree, file_path: str) -> list[dict]:
        """Return a record for every function_definition in the tree."""
        results: list[dict] = []
        for node in self._walk(tree.root_node, "function_definition"):
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
        """Return a record for every class_definition in the tree."""
        results: list[dict] = []
        for node in self._walk(tree.root_node, "class_definition"):
            name_node = node.child_by_field_name("name")
            if name_node is None:
                continue

            # Collect base class names from the "superclasses" field.
            bases: list[str] = []
            superclasses_node = node.child_by_field_name("superclasses")
            if superclasses_node is not None:
                for base in superclasses_node.named_children:
                    bases.append(base.text.decode())

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
        """Return a record for every import_statement / import_from_statement."""
        results: list[dict] = []

        for node in self._walk_multi(
            tree.root_node,
            {"import_statement", "import_from_statement"},
        ):
            line = node.start_point[0] + 1

            if node.type == "import_statement":
                # import os
                # import os, sys
                # import os as operating_system
                # Children include dotted_name / aliased_import nodes.
                for child in node.named_children:
                    if child.type == "dotted_name":
                        module = child.text.decode()
                        results.append(
                            {
                                "module": module,
                                "names": [],
                                "file_path": file_path,
                                "line": line,
                            }
                        )
                    elif child.type == "aliased_import":
                        # aliased_import: dotted_name "as" identifier
                        inner = child.child_by_field_name("name")
                        if inner is not None:
                            module = inner.text.decode()
                            results.append(
                                {
                                    "module": module,
                                    "names": [],
                                    "file_path": file_path,
                                    "line": line,
                                }
                            )

            else:  # import_from_statement
                # from pathlib import Path
                # from os.path import join, exists
                module_node = node.child_by_field_name("module_name")
                module = module_node.text.decode() if module_node is not None else ""

                names: list[str] = []
                for child in node.named_children:
                    # Skip the module_name node itself.
                    if child == module_node:
                        continue
                    if child.type == "dotted_name":
                        names.append(child.text.decode())
                    elif child.type == "aliased_import":
                        inner = child.child_by_field_name("name")
                        if inner is not None:
                            names.append(inner.text.decode())
                    elif child.type == "wildcard_import":
                        names.append("*")

                results.append(
                    {
                        "module": module,
                        "names": names,
                        "file_path": file_path,
                        "line": line,
                    }
                )

        return results

    def extract_calls(self, tree: Tree, file_path: str) -> list[dict]:
        """Return a record for every call expression in the tree."""
        results: list[dict] = []

        for node in self._walk(tree.root_node, "call"):
            callee = node.child_by_field_name("function")
            if callee is None:
                continue

            full_name = callee.text.decode()

            if callee.type == "attribute":
                # obj.method(...)  →  name = "method"
                attr_node = callee.child_by_field_name("attribute")
                name = attr_node.text.decode() if attr_node is not None else full_name
            elif callee.type == "identifier":
                name = full_name
            else:
                # Handles subscript, call chaining, etc. – use full text.
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
