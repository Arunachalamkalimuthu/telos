import unittest
from cawa.causal_graph import CausalGraph, CausalEdge


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


if __name__ == "__main__":
    unittest.main()
