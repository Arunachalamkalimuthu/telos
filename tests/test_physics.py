import unittest
from cawa.world import Entity, Relation, WorldState
from cawa.physics import gravity, containment


class TestGravity(unittest.TestCase):
    def test_gravity_emits_fall_edge_for_unsupported_massed_object(self):
        cup = Entity(id="cup", type="cup", properties={"mass": 0.2})
        ws = WorldState(entities={"cup": cup}, relations=())
        edges = gravity(ws)
        effects = [e.effect for e in edges]
        self.assertIn("cup.falls", effects)

    def test_gravity_does_not_emit_for_supported_object(self):
        cup = Entity(id="cup", type="cup", properties={"mass": 0.2})
        table = Entity(id="table", type="table", properties={})
        ws = WorldState(
            entities={"cup": cup, "table": table},
            relations=(Relation("ON", "cup", "table"),),
        )
        edges = gravity(ws)
        effects = [e.effect for e in edges]
        self.assertNotIn("cup.falls", effects)

    def test_gravity_ignores_massless_entity(self):
        ghost = Entity(id="ghost", type="concept", properties={})
        ws = WorldState(entities={"ghost": ghost}, relations=())
        self.assertEqual(gravity(ws), [])


class TestContainment(unittest.TestCase):
    def test_inverted_unsealed_container_with_contents_emits_escape(self):
        cup = Entity(
            id="cup",
            type="cup",
            properties={"orientation": "inverted", "sealed": False, "contains": "coffee"},
        )
        ws = WorldState(entities={"cup": cup}, relations=())
        edges = containment(ws)
        effects = [e.effect for e in edges]
        self.assertIn("cup.contents_escape", effects)

    def test_sealed_container_does_not_emit_escape(self):
        cup = Entity(
            id="cup",
            type="cup",
            properties={"orientation": "inverted", "sealed": True, "contains": "coffee"},
        )
        ws = WorldState(entities={"cup": cup}, relations=())
        edges = containment(ws)
        effects = [e.effect for e in edges]
        self.assertNotIn("cup.contents_escape", effects)

    def test_upright_container_does_not_emit_escape(self):
        cup = Entity(
            id="cup",
            type="cup",
            properties={"orientation": "upright", "sealed": False, "contains": "coffee"},
        )
        ws = WorldState(entities={"cup": cup}, relations=())
        self.assertEqual(containment(ws), [])

    def test_empty_container_does_not_emit_escape(self):
        cup = Entity(
            id="cup",
            type="cup",
            properties={"orientation": "inverted", "sealed": False},
        )
        ws = WorldState(entities={"cup": cup}, relations=())
        self.assertEqual(containment(ws), [])


if __name__ == "__main__":
    unittest.main()
