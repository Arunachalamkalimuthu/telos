"""Tests for the SQLite graph store."""

import os
import tempfile
import unittest

from telos.code_parser.store import GraphStore


class TestGraphStore(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(self._tmpdir, "test_graph.db")
        self.store = GraphStore(db_path)

    def tearDown(self):
        self.store.close()

    # ------------------------------------------------------------------
    # Schema / creation
    # ------------------------------------------------------------------

    def test_creates_db_and_tables(self):
        # get_stats works without error only if all three tables exist
        stats = self.store.get_stats()
        self.assertIn("node_count", stats)
        self.assertIn("edge_count", stats)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def test_add_and_get_node(self):
        self.store.add_node(
            id="n1",
            file_path="src/foo.py",
            name="Foo",
            kind="class",
            language="python",
            line_start=10,
            line_end=30,
        )
        node = self.store.get_node("n1")
        self.assertIsNotNone(node)
        self.assertEqual(node["id"], "n1")
        self.assertEqual(node["file_path"], "src/foo.py")
        self.assertEqual(node["name"], "Foo")
        self.assertEqual(node["kind"], "class")
        self.assertEqual(node["language"], "python")
        self.assertEqual(node["line_start"], 10)
        self.assertEqual(node["line_end"], 30)

    def test_get_node_returns_none_for_missing(self):
        self.assertIsNone(self.store.get_node("does_not_exist"))

    def test_get_all_nodes(self):
        self.store.add_node("n1", "a.py", "A", "function", "python")
        self.store.add_node("n2", "b.py", "B", "function", "python")
        nodes = self.store.get_all_nodes()
        ids = {n["id"] for n in nodes}
        self.assertIn("n1", ids)
        self.assertIn("n2", ids)
        self.assertEqual(len(nodes), 2)

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    def test_add_and_get_edge(self):
        self.store.add_node("n1", "a.py", "A", "function", "python")
        self.store.add_node("n2", "b.py", "B", "function", "python")
        self.store.add_edge(
            source="n1",
            target="n2",
            kind="calls",
            weight=0.9,
            file_path="a.py",
            line=15,
        )
        edges = self.store.get_edges_from("n1")
        self.assertEqual(len(edges), 1)
        e = edges[0]
        self.assertEqual(e["source"], "n1")
        self.assertEqual(e["target"], "n2")
        self.assertEqual(e["kind"], "calls")
        self.assertAlmostEqual(e["weight"], 0.9)
        self.assertEqual(e["file_path"], "a.py")
        self.assertEqual(e["line"], 15)

    def test_get_edges_to(self):
        self.store.add_node("n1", "a.py", "A", "function", "python")
        self.store.add_node("n2", "b.py", "B", "function", "python")
        self.store.add_node("n3", "c.py", "C", "function", "python")
        self.store.add_edge("n1", "n3", "calls")
        self.store.add_edge("n2", "n3", "calls")
        edges = self.store.get_edges_to("n3")
        sources = {e["source"] for e in edges}
        self.assertEqual(sources, {"n1", "n2"})

    def test_get_edges_from_returns_empty_for_unknown_source(self):
        self.assertEqual(self.store.get_edges_from("ghost"), [])

    def test_get_all_edges(self):
        self.store.add_node("n1", "a.py", "A", "function", "python")
        self.store.add_node("n2", "b.py", "B", "function", "python")
        self.store.add_edge("n1", "n2", "imports")
        self.store.add_edge("n2", "n1", "imports")
        edges = self.store.get_all_edges()
        self.assertEqual(len(edges), 2)

    # ------------------------------------------------------------------
    # Meta
    # ------------------------------------------------------------------

    def test_set_and_get_meta(self):
        self.store.set_meta("version", "1.0")
        self.assertEqual(self.store.get_meta("version"), "1.0")

    def test_get_meta_returns_none_for_missing_key(self):
        self.assertIsNone(self.store.get_meta("no_such_key"))

    def test_set_meta_overwrites_existing(self):
        self.store.set_meta("k", "old")
        self.store.set_meta("k", "new")
        self.assertEqual(self.store.get_meta("k"), "new")

    # ------------------------------------------------------------------
    # Stats & clear
    # ------------------------------------------------------------------

    def test_get_stats(self):
        self.store.add_node("n1", "a.py", "A", "function", "python")
        self.store.add_node("n2", "b.py", "B", "function", "python")
        self.store.add_edge("n1", "n2", "calls")
        stats = self.store.get_stats()
        self.assertEqual(stats["node_count"], 2)
        self.assertEqual(stats["edge_count"], 1)

    def test_clear_removes_all(self):
        self.store.add_node("n1", "a.py", "A", "function", "python")
        self.store.add_edge("n1", "n1", "self_ref")
        self.store.set_meta("k", "v")
        self.store.clear()
        stats = self.store.get_stats()
        self.assertEqual(stats["node_count"], 0)
        self.assertEqual(stats["edge_count"], 0)
        self.assertIsNone(self.store.get_meta("k"))


if __name__ == "__main__":
    unittest.main()
