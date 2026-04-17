import unittest

import numpy as np

from telos.causal_graph import CausalGraph
from telos.world import Entity, Relation, WorldState
from telos.structure_learner import generate_samples, learn_graph, compare_graphs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cup_on_laptop_world() -> WorldState:
    """Classic telos scenario: inverted cup of coffee above a laptop."""
    cup = Entity(
        id="cup",
        type="cup",
        properties={
            "mass": 0.25,
            "orientation": "inverted",
            "sealed": False,
            "contains": "coffee",
        },
    )
    coffee = Entity(
        id="coffee",
        type="liquid",
        properties={"conductive": True},
    )
    laptop = Entity(
        id="laptop",
        type="laptop",
        properties={"electronic": True, "mass": 2.0},
    )
    return WorldState(
        entities={"cup": cup, "coffee": coffee, "laptop": laptop},
        relations=(
            Relation("WILL_CONTACT", "coffee", "laptop"),
        ),
    )


def _make_simple_chain_graph() -> CausalGraph:
    """Ground-truth A -> B -> C graph."""
    g = CausalGraph()
    g.add_variable("A")
    g.add_variable("B")
    g.add_variable("C")
    g.add_mechanism("B", ["A"], mechanism=lambda p: p["A"] * 2, label="A->B")
    g.add_mechanism("C", ["B"], mechanism=lambda p: p["B"] * 3, label="B->C")
    return g


# ---------------------------------------------------------------------------
# generate_samples
# ---------------------------------------------------------------------------

class TestGenerateSamples(unittest.TestCase):

    def test_returns_correct_shape_and_names(self):
        world = _make_cup_on_laptop_world()
        samples, names = generate_samples(world, n=100, seed=0)
        self.assertIsInstance(samples, np.ndarray)
        self.assertEqual(samples.shape[0], 100)
        self.assertEqual(samples.shape[1], len(names))
        self.assertGreater(len(names), 0)

    def test_variable_names_match_columns(self):
        world = _make_cup_on_laptop_world()
        samples, names = generate_samples(world, n=50, seed=1)
        # Each name should be a string and number of names equals columns.
        for name in names:
            self.assertIsInstance(name, str)
        self.assertEqual(len(names), samples.shape[1])

    def test_deterministic_with_seed(self):
        world = _make_cup_on_laptop_world()
        s1, n1 = generate_samples(world, n=30, seed=42)
        s2, n2 = generate_samples(world, n=30, seed=42)
        self.assertEqual(n1, n2)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_differ(self):
        world = _make_cup_on_laptop_world()
        s1, _ = generate_samples(world, n=30, seed=0)
        s2, _ = generate_samples(world, n=30, seed=99)
        self.assertFalse(np.array_equal(s1, s2))

    def test_no_edges_returns_empty(self):
        """A world with no applicable physics produces an empty array."""
        empty = WorldState(
            entities={"rock": Entity(id="rock", type="rock", properties={})},
        )
        samples, names = generate_samples(empty, n=10)
        self.assertEqual(samples.shape, (10, 0))
        self.assertEqual(names, [])


# ---------------------------------------------------------------------------
# learn_graph
# ---------------------------------------------------------------------------

class TestLearnGraph(unittest.TestCase):

    def test_returned_graph_has_correct_variables(self):
        names = ["A", "B", "C"]
        data = np.random.default_rng(0).standard_normal((200, 3))
        g = learn_graph(data, names)
        self.assertIsInstance(g, CausalGraph)
        self.assertEqual(sorted(g.variables()), sorted(names))

    def test_recovers_linear_chain_skeleton(self):
        """PC should find edges between A-B and B-C but NOT between A-C."""
        rng = np.random.default_rng(42)
        n = 2000
        a = rng.standard_normal(n)
        b = a * 2 + rng.standard_normal(n) * 0.5
        c = b * 3 + rng.standard_normal(n) * 0.5
        data = np.column_stack([a, b, c])
        names = ["A", "B", "C"]

        g = learn_graph(data, names, alpha=0.01)
        # Extract all (parent, effect) pairs.
        edge_pairs = set()
        for edge in g.all_edges():
            for parent in edge.parents:
                edge_pairs.add((parent, edge.effect))

        # Skeleton: A--B and B--C should be connected (in at least one direction).
        ab_connected = ("A", "B") in edge_pairs or ("B", "A") in edge_pairs
        bc_connected = ("B", "C") in edge_pairs or ("C", "B") in edge_pairs
        ac_connected = ("A", "C") in edge_pairs or ("C", "A") in edge_pairs

        self.assertTrue(ab_connected, "Expected edge between A and B")
        self.assertTrue(bc_connected, "Expected edge between B and C")
        self.assertFalse(ac_connected, "Should not have direct edge A-C")

    def test_recovers_v_structure(self):
        """PC should orient edges toward the collider in A -> C <- B."""
        rng = np.random.default_rng(7)
        n = 3000
        a = rng.standard_normal(n)
        b = rng.standard_normal(n)
        c = a * 2 + b * 3 + rng.standard_normal(n) * 0.1
        data = np.column_stack([a, b, c])
        names = ["A", "B", "C"]

        g = learn_graph(data, names, alpha=0.05)
        edge_pairs = set()
        for edge in g.all_edges():
            for parent in edge.parents:
                edge_pairs.add((parent, edge.effect))

        # Directed edges A->C and B->C expected.
        self.assertIn(("A", "C"), edge_pairs)
        self.assertIn(("B", "C"), edge_pairs)
        # No edge between A and B.
        self.assertNotIn(("A", "B"), edge_pairs)
        self.assertNotIn(("B", "A"), edge_pairs)

    def test_empty_data_returns_edgeless_graph(self):
        names = ["X", "Y"]
        data = np.empty((0, 2))
        g = learn_graph(data, names)
        self.assertEqual(g.all_edges(), [])

    def test_single_variable(self):
        names = ["X"]
        data = np.random.default_rng(0).standard_normal((100, 1))
        g = learn_graph(data, names)
        self.assertEqual(g.variables(), ["X"])
        self.assertEqual(g.all_edges(), [])


# ---------------------------------------------------------------------------
# compare_graphs
# ---------------------------------------------------------------------------

class TestCompareGraphs(unittest.TestCase):

    def test_identical_graphs_perfect_scores(self):
        g1 = _make_simple_chain_graph()
        g2 = _make_simple_chain_graph()
        result = compare_graphs(g1, g2)
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["f1"], 1.0)

    def test_empty_learned_vs_nonempty_truth(self):
        learned = CausalGraph()
        learned.add_variable("A")
        learned.add_variable("B")
        learned.add_variable("C")
        truth = _make_simple_chain_graph()
        result = compare_graphs(learned, truth)
        self.assertAlmostEqual(result["recall"], 0.0)
        self.assertAlmostEqual(result["f1"], 0.0)

    def test_both_empty_is_perfect(self):
        g1 = CausalGraph()
        g1.add_variable("X")
        g2 = CausalGraph()
        g2.add_variable("X")
        result = compare_graphs(g1, g2)
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["f1"], 1.0)

    def test_superset_learned_has_full_recall(self):
        """Learned graph has extra edges but all ground-truth edges."""
        truth = _make_simple_chain_graph()

        learned = CausalGraph()
        learned.add_variable("A")
        learned.add_variable("B")
        learned.add_variable("C")
        learned.add_mechanism("B", ["A"], lambda p: p["A"], label="A->B")
        learned.add_mechanism("C", ["B"], lambda p: p["B"], label="B->C")
        learned.add_mechanism("C", ["A"], lambda p: p["A"], label="A->C (extra)")

        result = compare_graphs(learned, truth)
        self.assertAlmostEqual(result["recall"], 1.0)
        # Precision < 1 due to extra edge.
        self.assertLess(result["precision"], 1.0)

    def test_subset_learned_has_full_precision(self):
        """Learned graph is a strict subset of truth: perfect precision, imperfect recall."""
        truth = _make_simple_chain_graph()

        learned = CausalGraph()
        learned.add_variable("A")
        learned.add_variable("B")
        learned.add_variable("C")
        learned.add_mechanism("B", ["A"], lambda p: p["A"], label="A->B")

        result = compare_graphs(learned, truth)
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 0.5)

    def test_multi_parent_edge_expansion(self):
        """An edge with two parents should create two (parent, effect) pairs."""
        g = CausalGraph()
        g.add_variable("X")
        g.add_variable("Y")
        g.add_variable("Z")
        g.add_mechanism("Z", ["X", "Y"], lambda p: p["X"] + p["Y"])

        result = compare_graphs(g, g)
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
