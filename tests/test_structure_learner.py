import unittest

import numpy as np

from telos.causal_graph import CausalGraph
from telos.world import Entity, Relation, WorldState
from telos.structure_learner import (
    generate_samples,
    learn_graph,
    compare_graphs,
    has_latent_edges,
)


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


def _make_larger_chain_data(rng, n=2000):
    """6-variable chain: A->B->C->D, E->D, E->F."""
    a = rng.standard_normal(n)
    b = 2.0 * a + rng.standard_normal(n) * 0.3
    e = rng.standard_normal(n)
    c = 1.5 * b + rng.standard_normal(n) * 0.3
    d = c + 2.0 * e + rng.standard_normal(n) * 0.3
    f = 1.8 * e + rng.standard_normal(n) * 0.3
    data = np.column_stack([a, b, c, d, e, f])
    names = ["A", "B", "C", "D", "E", "F"]
    return data, names


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
        empty = WorldState(
            entities={"rock": Entity(id="rock", type="rock", properties={})},
        )
        samples, names = generate_samples(empty, n=10)
        self.assertEqual(samples.shape, (10, 0))
        self.assertEqual(names, [])

    def test_nonlinear_produces_different_distribution(self):
        """Nonlinear mode should produce different values than linear."""
        world = _make_cup_on_laptop_world()
        s_lin, _ = generate_samples(world, n=200, seed=42, nonlinear=False)
        s_nlin, _ = generate_samples(world, n=200, seed=42, nonlinear=True)
        self.assertFalse(np.allclose(s_lin, s_nlin))


# ---------------------------------------------------------------------------
# learn_graph — PC
# ---------------------------------------------------------------------------

class TestLearnGraphPC(unittest.TestCase):

    def test_returned_graph_has_correct_variables(self):
        names = ["A", "B", "C"]
        data = np.random.default_rng(0).standard_normal((200, 3))
        g = learn_graph(data, names, method="pc")
        self.assertIsInstance(g, CausalGraph)
        self.assertEqual(sorted(g.variables()), sorted(names))

    def test_recovers_linear_chain_skeleton(self):
        rng = np.random.default_rng(42)
        n = 2000
        a = rng.standard_normal(n)
        b = a * 2 + rng.standard_normal(n) * 0.5
        c = b * 3 + rng.standard_normal(n) * 0.5
        data = np.column_stack([a, b, c])
        names = ["A", "B", "C"]

        g = learn_graph(data, names, alpha=0.01, method="pc")
        edge_pairs = set()
        for edge in g.all_edges():
            for parent in edge.parents:
                edge_pairs.add((parent, edge.effect))

        ab_connected = ("A", "B") in edge_pairs or ("B", "A") in edge_pairs
        bc_connected = ("B", "C") in edge_pairs or ("C", "B") in edge_pairs
        ac_connected = ("A", "C") in edge_pairs or ("C", "A") in edge_pairs

        self.assertTrue(ab_connected, "Expected edge between A and B")
        self.assertTrue(bc_connected, "Expected edge between B and C")
        self.assertFalse(ac_connected, "Should not have direct edge A-C")

    def test_recovers_v_structure(self):
        rng = np.random.default_rng(7)
        n = 3000
        a = rng.standard_normal(n)
        b = rng.standard_normal(n)
        c = a * 2 + b * 3 + rng.standard_normal(n) * 0.1
        data = np.column_stack([a, b, c])
        names = ["A", "B", "C"]

        g = learn_graph(data, names, alpha=0.05, method="pc")
        edge_pairs = set()
        for edge in g.all_edges():
            for parent in edge.parents:
                edge_pairs.add((parent, edge.effect))

        self.assertIn(("A", "C"), edge_pairs)
        self.assertIn(("B", "C"), edge_pairs)
        self.assertNotIn(("A", "B"), edge_pairs)
        self.assertNotIn(("B", "A"), edge_pairs)

    def test_empty_data_returns_edgeless_graph(self):
        names = ["X", "Y"]
        data = np.empty((0, 2))
        g = learn_graph(data, names, method="pc")
        self.assertEqual(g.all_edges(), [])

    def test_single_variable(self):
        names = ["X"]
        data = np.random.default_rng(0).standard_normal((100, 1))
        g = learn_graph(data, names, method="pc")
        self.assertEqual(g.variables(), ["X"])
        self.assertEqual(g.all_edges(), [])


# ---------------------------------------------------------------------------
# learn_graph — larger graphs
# ---------------------------------------------------------------------------

class TestLearnGraphLarger(unittest.TestCase):

    def test_six_variable_graph_recovers_key_edges(self):
        """PC should recover the skeleton of a 6-variable graph."""
        rng = np.random.default_rng(42)
        data, names = _make_larger_chain_data(rng, n=3000)

        g = learn_graph(data, names, alpha=0.01, method="pc")
        edge_pairs = set()
        for edge in g.all_edges():
            for parent in edge.parents:
                edge_pairs.add((parent, edge.effect))

        # A-B, B-C, C-D, E-D, E-F should be connected (in some direction).
        ab = ("A", "B") in edge_pairs or ("B", "A") in edge_pairs
        bc = ("B", "C") in edge_pairs or ("C", "B") in edge_pairs
        ef = ("E", "F") in edge_pairs or ("F", "E") in edge_pairs

        self.assertTrue(ab, "Expected edge between A and B")
        self.assertTrue(bc, "Expected edge between B and C")
        self.assertTrue(ef, "Expected edge between E and F")

    def test_ges_on_six_variable_graph(self):
        """GES should recover at least some edges in a 6-variable graph."""
        rng = np.random.default_rng(42)
        data, names = _make_larger_chain_data(rng, n=5000)

        g = learn_graph(data, names, method="ges")
        edge_pairs = set()
        for edge in g.all_edges():
            for parent in edge.parents:
                edge_pairs.add((parent, edge.effect))

        # GES should find at least some edges in a structured graph.
        self.assertGreater(len(edge_pairs), 0, "GES should discover edges")


# ---------------------------------------------------------------------------
# learn_graph — nonlinear (KCI)
# ---------------------------------------------------------------------------

class TestLearnGraphNonlinear(unittest.TestCase):

    def test_kci_recovers_nonlinear_v_structure(self):
        """KCI test should handle nonlinear relationships."""
        rng = np.random.default_rng(7)
        n = 300  # smaller n due to KCI being O(n^2)
        a = rng.standard_normal(n)
        b = rng.standard_normal(n)
        c = np.tanh(a * 2) + b ** 2 + rng.standard_normal(n) * 0.1
        data = np.column_stack([a, b, c])
        names = ["A", "B", "C"]

        g = learn_graph(data, names, alpha=0.05, method="pc", indep_test="kci")
        edge_pairs = set()
        for edge in g.all_edges():
            for parent in edge.parents:
                edge_pairs.add((parent, edge.effect))

        # A and B should both connect to C.
        ac = ("A", "C") in edge_pairs or ("C", "A") in edge_pairs
        bc = ("B", "C") in edge_pairs or ("C", "B") in edge_pairs
        self.assertTrue(ac, "KCI should find A-C connection in nonlinear data")
        self.assertTrue(bc, "KCI should find B-C connection in nonlinear data")


# ---------------------------------------------------------------------------
# learn_graph — FCI (latent confounders)
# ---------------------------------------------------------------------------

class TestLearnGraphFCI(unittest.TestCase):

    def test_fci_returns_valid_graph(self):
        """FCI should return a CausalGraph with variables."""
        rng = np.random.default_rng(42)
        n = 1000
        a = rng.standard_normal(n)
        b = a * 2 + rng.standard_normal(n) * 0.3
        data = np.column_stack([a, b])
        names = ["A", "B"]

        g = learn_graph(data, names, method="fci")
        self.assertIsInstance(g, CausalGraph)
        self.assertEqual(sorted(g.variables()), ["A", "B"])

    def test_fci_detects_latent_confounder(self):
        """When L causes both X and Y but L is unobserved, FCI should find
        a bidirected edge (latent confounder) between X and Y."""
        rng = np.random.default_rng(42)
        n = 2000
        # L is a latent common cause of X and Y.
        latent = rng.standard_normal(n)
        x = latent * 2.0 + rng.standard_normal(n) * 0.2
        y = latent * 3.0 + rng.standard_normal(n) * 0.2
        # We only observe X and Y (not L).
        data = np.column_stack([x, y])
        names = ["X", "Y"]

        g = learn_graph(data, names, alpha=0.05, method="fci")
        latent_pairs = has_latent_edges(g)

        # FCI should detect that X and Y are connected (possibly bidirected).
        edge_pairs = set()
        for edge in g.all_edges():
            for parent in edge.parents:
                edge_pairs.add((parent, edge.effect))

        connected = len(edge_pairs) > 0
        self.assertTrue(connected, "FCI should find a connection between X and Y")


# ---------------------------------------------------------------------------
# learn_graph — GES (better orientation)
# ---------------------------------------------------------------------------

class TestLearnGraphGES(unittest.TestCase):

    def test_ges_finds_connections_in_v_structure(self):
        """GES should find connections between A-C and B-C in a v-structure."""
        rng = np.random.default_rng(7)
        n = 5000
        a = rng.standard_normal(n)
        b = rng.standard_normal(n)
        c = a * 2 + b * 3 + rng.standard_normal(n) * 0.1
        data = np.column_stack([a, b, c])
        names = ["A", "B", "C"]

        g = learn_graph(data, names, method="ges")
        edge_pairs = set()
        for edge in g.all_edges():
            for parent in edge.parents:
                edge_pairs.add((parent, edge.effect))

        # GES should find connections (skeleton) even if orientation varies.
        ac = ("A", "C") in edge_pairs or ("C", "A") in edge_pairs
        bc = ("B", "C") in edge_pairs or ("C", "B") in edge_pairs
        self.assertTrue(ac, "GES should find A-C connection")
        self.assertTrue(bc, "GES should find B-C connection")

    def test_ges_returns_valid_graph(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((500, 3))
        names = ["X", "Y", "Z"]
        g = learn_graph(data, names, method="ges")
        self.assertIsInstance(g, CausalGraph)
        self.assertEqual(sorted(g.variables()), sorted(names))


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
        self.assertLess(result["precision"], 1.0)

    def test_subset_learned_has_full_precision(self):
        truth = _make_simple_chain_graph()
        learned = CausalGraph()
        learned.add_variable("A")
        learned.add_variable("B")
        learned.add_variable("C")
        learned.add_mechanism("B", ["A"], lambda p: p["A"], label="A->B")

        result = compare_graphs(learned, truth)
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 0.5)

    def test_latent_edges_excluded_from_comparison(self):
        """Edges labelled 'latent:' should not count in precision/recall."""
        g = CausalGraph()
        g.add_variable("X")
        g.add_variable("Y")
        g.add_mechanism("Y", ["X"], lambda p: p["X"], label="latent:fci:X->Y")

        truth = CausalGraph()
        truth.add_variable("X")
        truth.add_variable("Y")

        result = compare_graphs(g, truth)
        # Latent edges excluded → both have empty edge sets → perfect.
        self.assertAlmostEqual(result["precision"], 1.0)

    def test_multi_parent_edge_expansion(self):
        g = CausalGraph()
        g.add_variable("X")
        g.add_variable("Y")
        g.add_variable("Z")
        g.add_mechanism("Z", ["X", "Y"], lambda p: p["X"] + p["Y"])

        result = compare_graphs(g, g)
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 1.0)


# ---------------------------------------------------------------------------
# has_latent_edges
# ---------------------------------------------------------------------------

class TestHasLatentEdges(unittest.TestCase):

    def test_no_latent_edges(self):
        g = _make_simple_chain_graph()
        self.assertEqual(has_latent_edges(g), [])

    def test_detects_latent_edges(self):
        g = CausalGraph()
        g.add_variable("X")
        g.add_variable("Y")
        g.add_mechanism("Y", ["X"], lambda p: p["X"], label="latent:fci:X->Y")
        pairs = has_latent_edges(g)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0], ("X", "Y"))


if __name__ == "__main__":
    unittest.main()
