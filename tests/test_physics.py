import unittest
from cawa.world import Entity, Relation, WorldState
from cawa.physics import gravity, containment, impact


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


class TestImpact(unittest.TestCase):
    def test_fragile_object_falling_onto_hard_surface_emits_break(self):
        glass = Entity(
            id="glass",
            type="glass",
            properties={"mass": 0.3, "fragile": True, "impact_threshold": 1.0},
        )
        floor = Entity(id="floor", type="floor", properties={"hardness": "hard"})
        ws = WorldState(
            entities={"glass": glass, "floor": floor},
            relations=(Relation("WILL_HIT", "glass", "floor", attributes={"velocity": 3.0}),),
        )
        edges = impact(ws)
        effects = [e.effect for e in edges]
        self.assertIn("glass.breaks", effects)

    def test_non_fragile_object_does_not_break(self):
        ball = Entity(
            id="ball",
            type="ball",
            properties={"mass": 0.1, "fragile": False, "impact_threshold": 1.0},
        )
        floor = Entity(id="floor", type="floor", properties={"hardness": "hard"})
        ws = WorldState(
            entities={"ball": ball, "floor": floor},
            relations=(Relation("WILL_HIT", "ball", "floor", attributes={"velocity": 10.0}),),
        )
        edges = impact(ws)
        self.assertEqual(edges, [])

    def test_low_velocity_impact_does_not_break(self):
        glass = Entity(
            id="glass",
            type="glass",
            properties={"mass": 0.3, "fragile": True, "impact_threshold": 5.0},
        )
        floor = Entity(id="floor", type="floor", properties={"hardness": "hard"})
        ws = WorldState(
            entities={"glass": glass, "floor": floor},
            relations=(Relation("WILL_HIT", "glass", "floor", attributes={"velocity": 1.0}),),
        )
        edges = impact(ws)
        self.assertEqual(edges, [])


if __name__ == "__main__":
    unittest.main()
