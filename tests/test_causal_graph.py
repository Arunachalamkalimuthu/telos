import unittest
from telos.causal_graph import CausalGraph, CausalEdge


class TestCausalGraphConstruction(unittest.TestCase):
    def test_add_variable_and_get_value(self):
        g = CausalGraph()
        g.add_variable("a", initial=1)
        self.assertEqual(g.get("a"), 1)

    def test_add_mechanism_links_parents_to_effect(self):
        g = CausalGraph()
        g.add_variable("a", initial=2)
        g.add_variable("b")
        g.add_mechanism("b", parents=["a"], mechanism=lambda p: p["a"] * 2, label="double")
        edges = g.edges_into("b")
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].parents, ("a",))
        self.assertEqual(edges[0].effect, "b")
        self.assertEqual(edges[0].label, "double")


class TestCausalGraphPropagate(unittest.TestCase):
    def test_propagate_simple_chain(self):
        g = CausalGraph()
        g.add_variable("a", initial=3)
        g.add_variable("b")
        g.add_variable("c")
        g.add_mechanism("b", ["a"], lambda p: p["a"] + 1)
        g.add_mechanism("c", ["b"], lambda p: p["b"] * 10)
        state = g.propagate()
        self.assertEqual(state["a"], 3)
        self.assertEqual(state["b"], 4)
        self.assertEqual(state["c"], 40)

    def test_propagate_multiple_parents(self):
        g = CausalGraph()
        g.add_variable("x", initial=2)
        g.add_variable("y", initial=5)
        g.add_variable("z")
        g.add_mechanism("z", ["x", "y"], lambda p: p["x"] + p["y"])
        self.assertEqual(g.propagate()["z"], 7)

    def test_propagate_raises_on_cycle(self):
        g = CausalGraph()
        g.add_variable("a")
        g.add_variable("b")
        g.add_mechanism("a", ["b"], lambda p: p["b"])
        g.add_mechanism("b", ["a"], lambda p: p["a"])
        with self.assertRaises(ValueError):
            g.propagate()


class TestCausalGraphIntervention(unittest.TestCase):
    def test_do_pins_variable_and_severs_incoming(self):
        g = CausalGraph()
        g.add_variable("a", initial=3)
        g.add_variable("b")
        g.add_mechanism("b", ["a"], lambda p: p["a"] + 1)
        g2 = g.do("b", 100)
        # Original graph untouched.
        self.assertEqual(g.propagate()["b"], 4)
        # Intervened graph: b is pinned regardless of a.
        self.assertEqual(g2.propagate()["b"], 100)
        # And b has no incoming edges anymore.
        self.assertEqual(g2.edges_into("b"), [])

    def test_counterfactual_multiple_interventions(self):
        g = CausalGraph()
        g.add_variable("gravity", initial=True)
        g.add_variable("sealed", initial=False)
        g.add_variable("spill")
        g.add_mechanism(
            "spill",
            ["gravity", "sealed"],
            lambda p: p["gravity"] and not p["sealed"],
        )
        self.assertTrue(g.propagate()["spill"])
        # Counterfactual 1: sealed=True → no spill.
        self.assertFalse(g.counterfactual({"sealed": True})["spill"])
        # Counterfactual 2: gravity=False → no spill.
        self.assertFalse(g.counterfactual({"gravity": False})["spill"])


class TestCausalGraphExplain(unittest.TestCase):
    def test_explain_path_returns_chain_of_edges(self):
        g = CausalGraph()
        g.add_variable("orientation", initial="inverted")
        g.add_variable("sealed", initial=False)
        g.add_variable("contents_escape")
        g.add_variable("liquid_falls")
        g.add_variable("laptop_damaged")
        g.add_mechanism(
            "contents_escape",
            ["orientation", "sealed"],
            lambda p: p["orientation"] == "inverted" and not p["sealed"],
            label="containment_breach",
        )
        g.add_mechanism(
            "liquid_falls",
            ["contents_escape"],
            lambda p: p["contents_escape"],
            label="gravity",
        )
        g.add_mechanism(
            "laptop_damaged",
            ["liquid_falls"],
            lambda p: p["liquid_falls"],
            label="impact",
        )
        chain = g.explain_path("laptop_damaged")
        labels = [e.label for e in chain]
        self.assertEqual(labels, ["containment_breach", "gravity", "impact"])


if __name__ == "__main__":
    unittest.main()
