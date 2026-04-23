"""Tests for the graph builder (scan files -> nodes + edges)."""

import os
import tempfile
import textwrap
import unittest

from telos.code_parser.graph_builder import GraphBuilder
from telos.code_parser.store import GraphStore


class _GraphBuilderTestBase(unittest.TestCase):
    """Common setup: temp dir + fresh GraphStore."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(self._tmpdir, "test_graph.db")
        self.store = GraphStore(db_path)
        self.builder = GraphBuilder(self.store)

    def tearDown(self):
        self.store.close()

    def _write(self, rel_path: str, content: str) -> str:
        """Write *content* to <tmpdir>/<rel_path> and return the full path."""
        full = os.path.join(self._tmpdir, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as fh:
            fh.write(textwrap.dedent(content))
        return full


class TestScanPythonFile(_GraphBuilderTestBase):
    def test_scan_python_file(self):
        """Two functions in one file produce three nodes (module + 2 funcs)."""
        fpath = self._write("app.py", """\
            def greet(name):
                return f"Hello, {name}"

            def main():
                greet("world")
        """)
        self.builder.scan_file(fpath, self._tmpdir)

        nodes = self.store.get_all_nodes()
        node_ids = {n["id"] for n in nodes}

        self.assertIn("app.py", node_ids)            # module
        self.assertIn("app.py:greet", node_ids)       # function
        self.assertIn("app.py:main", node_ids)        # function
        self.assertEqual(len(nodes), 3)


class TestScanCreatesCallEdges(_GraphBuilderTestBase):
    def test_scan_creates_call_edges(self):
        """main() calls greet() -> a CALLS edge must exist."""
        fpath = self._write("app.py", """\
            def greet(name):
                return f"Hello, {name}"

            def main():
                greet("world")
        """)
        self.builder.scan_file(fpath, self._tmpdir)

        edges = self.store.get_all_edges()
        call_edges = [e for e in edges if e["kind"] == "CALLS"]

        self.assertTrue(len(call_edges) >= 1)
        sources = {e["source"] for e in call_edges}
        targets = {e["target"] for e in call_edges}
        self.assertIn("app.py:main", sources)
        self.assertIn("app.py:greet", targets)


class TestScanDirectory(_GraphBuilderTestBase):
    def test_scan_directory(self):
        """Scanning a directory with two .py files reports correct count."""
        self._write("a.py", """\
            def foo():
                pass
        """)
        self._write("b.py", """\
            def bar():
                pass
        """)

        result = self.builder.scan_directory(self._tmpdir)

        self.assertEqual(result["files_scanned"], 2)
        self.assertGreaterEqual(result["nodes"], 4)  # 2 modules + 2 funcs
        self.assertGreaterEqual(result["edges"], 0)


class TestScanIgnoresUnsupportedFiles(_GraphBuilderTestBase):
    def test_scan_ignores_unsupported_files(self):
        """A .md file should not be scanned."""
        self._write("README.md", "# Hello\n")

        result = self.builder.scan_directory(self._tmpdir)

        self.assertEqual(result["files_scanned"], 0)


class TestScanCreatesImportEdges(_GraphBuilderTestBase):
    def test_scan_creates_import_edges(self):
        """import os -> an IMPORTS edge from the module to 'os'."""
        fpath = self._write("importer.py", """\
            import os
            from pathlib import Path

            def run():
                pass
        """)
        self.builder.scan_file(fpath, self._tmpdir)

        edges = self.store.get_all_edges()
        import_edges = [e for e in edges if e["kind"] == "IMPORTS"]

        self.assertTrue(len(import_edges) >= 2)
        targets = {e["target"] for e in import_edges}
        self.assertIn("os", targets)
        self.assertIn("pathlib", targets)

        # All import edges should originate from the module node.
        sources = {e["source"] for e in import_edges}
        self.assertEqual(sources, {"importer.py"})


class TestScanCreatesInheritsEdges(_GraphBuilderTestBase):
    def test_scan_creates_inherits_edges(self):
        """class Child(Base) -> an INHERITS edge from Child to Base."""
        fpath = self._write("hierarchy.py", """\
            class Base:
                pass

            class Child(Base):
                pass
        """)
        self.builder.scan_file(fpath, self._tmpdir)

        edges = self.store.get_all_edges()
        inherits_edges = [e for e in edges if e["kind"] == "INHERITS"]

        self.assertEqual(len(inherits_edges), 1)
        e = inherits_edges[0]
        self.assertEqual(e["source"], "hierarchy.py:Child")
        self.assertEqual(e["target"], "hierarchy.py:Base")
        self.assertAlmostEqual(e["weight"], 0.9)


if __name__ == "__main__":
    unittest.main()
