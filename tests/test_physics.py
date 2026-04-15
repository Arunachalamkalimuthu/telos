import unittest
from cawa.world import Entity, Relation, WorldState
from cawa.physics import gravity


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


if __name__ == "__main__":
    unittest.main()
