"""Tests for telos.nlu — scene and query parsing."""

from telos.nlu import parse_scene, parse_query


# ── parse_scene ──────────────────────────────────────────────────────────


class TestParseScene:
    def test_cup_on_table(self):
        ws = parse_scene("A cup is on a table")
        # Entities
        assert "cup" in ws.entities
        assert "table" in ws.entities
        assert ws.entities["cup"].type == "cup"
        assert ws.entities["table"].type == "table"
        # Relation
        on_rels = ws.relations_of("ON")
        assert len(on_rels) == 1
        assert on_rels[0].src == "cup"
        assert on_rels[0].dst == "table"

    def test_laptop_near_cup(self):
        ws = parse_scene("A laptop is near a cup")
        assert "laptop" in ws.entities
        assert "cup" in ws.entities
        near_rels = ws.relations_of("NEAR")
        assert len(near_rels) == 1
        assert near_rels[0].src == "laptop"
        assert near_rels[0].dst == "cup"

    def test_heavy_cup_on_wooden_table(self):
        ws = parse_scene("A heavy cup is on a wooden table")
        cup = ws.entities["cup"]
        assert "heavy" in cup.properties.get("attributes", [])
        table = ws.entities["table"]
        assert "wooden" in table.properties.get("attributes", [])
        assert len(ws.relations_of("ON")) == 1

    def test_cup_on_table_near_laptop(self):
        ws = parse_scene("A cup is on a table near a laptop")
        entity_types = {e.type for e in ws.entities.values()}
        assert len(entity_types) >= 2
        # Should have at least cup, table, laptop
        assert "cup" in entity_types
        assert "table" in entity_types
        assert "laptop" in entity_types


# ── parse_query ──────────────────────────────────────────────────────────


class TestParseQuery:
    def test_what_happens_if_cup_falls(self):
        result = parse_query("What happens if the cup falls?")
        assert result["type"] == "counterfactual"
        assert result["subject"] == "cup"

    def test_will_laptop_get_damaged(self):
        result = parse_query("Will the laptop get damaged?")
        assert result["type"] == "prediction"
        assert result["subject"] == "laptop"

    def test_what_would_happen_if_sealed(self):
        result = parse_query("What would happen if the cup were sealed?")
        assert result["type"] == "counterfactual"
