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


if __name__ == "__main__":
    unittest.main()
