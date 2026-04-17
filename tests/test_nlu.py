"""Tests for telos.nlu — scene parsing, query parsing, property mapping, executable queries."""

import unittest

from telos.causal_graph import CausalGraph
from telos.nlu import (
    execute_query,
    map_properties,
    parse_query,
    parse_scene,
)


# ── map_properties ─────────────────────────────────────────────────────────


class TestMapProperties(unittest.TestCase):

    def test_ceramic_sets_fragile_and_material(self):
        props = map_properties("cup", ["ceramic"])
        self.assertTrue(props["fragile"])
        self.assertEqual(props["material"], "ceramic")

    def test_noun_type_provides_base_properties(self):
        props = map_properties("laptop", [])
        self.assertTrue(props["electronic"])
        self.assertIn("mass", props)

    def test_adjective_overrides_noun(self):
        """Adjective 'sturdy' should override noun default fragile=True."""
        props = map_properties("glass", ["sturdy"])
        self.assertFalse(props["fragile"])

    def test_unknown_type_and_adj(self):
        props = map_properties("frambulator", ["glorbic"])
        self.assertEqual(props, {})

    def test_multiple_adjectives(self):
        props = map_properties("cup", ["heavy", "ceramic", "sealed"])
        self.assertTrue(props["fragile"])
        self.assertEqual(props["mass_hint"], "heavy")
        self.assertTrue(props["sealed"])

    def test_liquid_nouns(self):
        props = map_properties("coffee", [])
        self.assertTrue(props["conductive"])
        self.assertEqual(props["type_hint"], "liquid")


# ── parse_scene (basic) ───────────────────────────────────────────────────


class TestParseSceneBasic(unittest.TestCase):

    def test_cup_on_table(self):
        ws = parse_scene("A cup is on a table")
        self.assertIn("cup", ws.entities)
        self.assertIn("table", ws.entities)
        on_rels = ws.relations_of("ON")
        self.assertEqual(len(on_rels), 1)
        self.assertEqual(on_rels[0].src, "cup")
        self.assertEqual(on_rels[0].dst, "table")

    def test_laptop_near_cup(self):
        ws = parse_scene("A laptop is near a cup")
        near_rels = ws.relations_of("NEAR")
        self.assertEqual(len(near_rels), 1)

    def test_heavy_cup_on_wooden_table(self):
        ws = parse_scene("A heavy cup is on a wooden table")
        cup = ws.entities["cup"]
        self.assertIn("heavy", cup.properties.get("attributes", []))
        table = ws.entities["table"]
        self.assertIn("wooden", table.properties.get("attributes", []))

    def test_cup_on_table_near_laptop(self):
        ws = parse_scene("A cup is on a table near a laptop")
        types = {e.type for e in ws.entities.values()}
        self.assertIn("cup", types)
        self.assertIn("table", types)
        self.assertIn("laptop", types)


# ── parse_scene (physics property enrichment) ─────────────────────────────


class TestParseSceneProperties(unittest.TestCase):

    def test_ceramic_cup_has_fragile(self):
        ws = parse_scene("A ceramic cup is on a table")
        cup = ws.entities["cup"]
        self.assertTrue(cup.get("fragile"))
        self.assertEqual(cup.get("material"), "ceramic")

    def test_wooden_table_has_material(self):
        ws = parse_scene("A cup is on a wooden table")
        table = ws.entities["table"]
        self.assertEqual(table.get("material"), "wood")

    def test_laptop_gets_electronic(self):
        ws = parse_scene("A laptop is near a cup")
        laptop = ws.entities["laptop"]
        self.assertTrue(laptop.get("electronic"))

    def test_sealed_cup_via_predicate(self):
        ws = parse_scene("The cup is sealed")
        cup = ws.entities["cup"]
        self.assertTrue(cup.get("sealed"))

    def test_enrich_disabled(self):
        ws = parse_scene("A ceramic cup is on a table", enrich_properties=False)
        cup = ws.entities["cup"]
        self.assertNotIn("fragile", dict(cup.properties))
        self.assertNotIn("material", dict(cup.properties))


# ── parse_scene (negation) ────────────────────────────────────────────────


class TestParseSceneNegation(unittest.TestCase):

    def test_not_sealed(self):
        """'The cup is not sealed' → sealed=False."""
        ws = parse_scene("The cup is not sealed")
        cup = ws.entities["cup"]
        self.assertFalse(cup.get("sealed"))

    def test_not_fragile_attribute(self):
        """Negated adjective modifier is prefixed with not_."""
        ws = parse_scene("A not fragile cup is on the table")
        cup = ws.entities["cup"]
        attrs = cup.properties.get("attributes", [])
        self.assertIn("not_fragile", attrs)


# ── parse_query (basic) ──────────────────────────────────────────────────


class TestParseQueryBasic(unittest.TestCase):

    def test_counterfactual_what_happens(self):
        result = parse_query("What happens if the cup falls?")
        self.assertEqual(result["type"], "counterfactual")
        self.assertEqual(result["subject"], "cup")

    def test_prediction_will(self):
        result = parse_query("Will the laptop get damaged?")
        self.assertEqual(result["type"], "prediction")
        self.assertEqual(result["subject"], "laptop")

    def test_counterfactual_what_would(self):
        result = parse_query("What would happen if the cup were sealed?")
        self.assertEqual(result["type"], "counterfactual")


# ── parse_query (executable interventions) ────────────────────────────────


class TestParseQueryInterventions(unittest.TestCase):

    def test_counterfactual_sealed_produces_intervention(self):
        result = parse_query("What would happen if the cup were sealed?")
        self.assertIn("intervention", result)
        self.assertIn("cup.sealed", result["intervention"])
        self.assertTrue(result["intervention"]["cup.sealed"])

    def test_counterfactual_falls_produces_intervention(self):
        result = parse_query("What happens if the cup falls?")
        self.assertIn("intervention", result)
        self.assertIn("cup.falls", result["intervention"])
        self.assertTrue(result["intervention"]["cup.falls"])

    def test_prediction_damaged_produces_target(self):
        result = parse_query("Will the laptop get damaged?")
        self.assertIn("target", result)
        self.assertEqual(result["target"], "laptop.damaged")

    def test_prediction_break_produces_target(self):
        result = parse_query("Will the cup break?")
        self.assertIn("target", result)
        self.assertEqual(result["target"], "cup.breaks")

    def test_counterfactual_spill(self):
        result = parse_query("What if the coffee spills?")
        self.assertIn("intervention", result)
        self.assertIn("coffee.contents_escape", result["intervention"])


# ── execute_query ────────────────────────────────────────────────────────


class TestExecuteQuery(unittest.TestCase):

    def _make_graph(self) -> CausalGraph:
        """Simple causal graph: cup.contents_escape → laptop.damaged."""
        g = CausalGraph()
        g.add_variable("cup.contents_escape", initial=True)
        g.add_variable("laptop.damaged", initial=False)
        g.add_mechanism(
            "laptop.damaged",
            ["cup.contents_escape"],
            lambda p: bool(p["cup.contents_escape"]),
            label="liquid_damage",
        )
        return g

    def test_counterfactual_execution(self):
        g = self._make_graph()
        query = parse_query("What would happen if the cup were sealed?")
        # Manually set intervention to match graph variables.
        query["intervention"] = {"cup.contents_escape": False}
        result = execute_query(query, g)
        self.assertEqual(result["type"], "counterfactual")
        self.assertFalse(result["result"]["laptop.damaged"])

    def test_prediction_execution(self):
        g = self._make_graph()
        query = {
            "type": "prediction",
            "subject": "laptop",
            "action": "damage",
            "target": "laptop.damaged",
        }
        result = execute_query(query, g)
        self.assertEqual(result["type"], "prediction")
        self.assertTrue(result["result"])  # laptop is damaged in default state

    def test_counterfactual_with_nonexistent_variable(self):
        """Interventions on non-existent variables are silently filtered."""
        g = self._make_graph()
        query = {
            "type": "counterfactual",
            "intervention": {"nonexistent.var": True},
        }
        result = execute_query(query, g)
        self.assertEqual(result["intervention"], {})
        # Falls back to propagate().
        self.assertTrue(result["result"]["laptop.damaged"])

    def test_prediction_with_nonexistent_target(self):
        g = self._make_graph()
        query = {
            "type": "prediction",
            "target": "nonexistent.var",
        }
        result = execute_query(query, g)
        # Returns full state when target not found.
        self.assertIn("laptop.damaged", result["result"])


if __name__ == "__main__":
    unittest.main()
