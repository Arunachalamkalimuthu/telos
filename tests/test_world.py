import unittest
from cawa.world import UNKNOWN, Entity, Relation, WorldState


class TestEntity(unittest.TestCase):
    def test_entity_has_id_type_properties(self):
        e = Entity(id="cup_1", type="cup", properties={"mass": 0.2, "sealed": False})
        self.assertEqual(e.id, "cup_1")
        self.assertEqual(e.type, "cup")
        self.assertEqual(e.get("mass"), 0.2)
        self.assertFalse(e.get("sealed"))

    def test_entity_unknown_property_returns_UNKNOWN(self):
        e = Entity(id="x", type="frambulator", properties={})
        self.assertIs(e.get("glorbic_index"), UNKNOWN)

    def test_entity_is_hashable_and_frozen(self):
        e = Entity(id="a", type="t", properties={"k": 1})
        self.assertEqual(hash(e), hash(e))
        with self.assertRaises(Exception):
            e.id = "changed"


class TestRelation(unittest.TestCase):
    def test_relation_has_name_src_dst(self):
        r = Relation(name="ON", src="cup_1", dst="table_1")
        self.assertEqual(r.name, "ON")
        self.assertEqual(r.src, "cup_1")
        self.assertEqual(r.dst, "table_1")


class TestWorldState(unittest.TestCase):
    def test_worldstate_construction(self):
        cup = Entity(id="cup_1", type="cup", properties={"mass": 0.2})
        ws = WorldState(entities={"cup_1": cup}, relations=())
        self.assertIs(ws.get_entity("cup_1"), cup)

    def test_with_entity_returns_new_state(self):
        ws1 = WorldState(entities={}, relations=())
        cup = Entity(id="cup_1", type="cup", properties={})
        ws2 = ws1.with_entity(cup)
        self.assertNotIn("cup_1", ws1.entities)
        self.assertIn("cup_1", ws2.entities)
        self.assertIsNot(ws1, ws2)

    def test_with_relation_returns_new_state(self):
        ws1 = WorldState(entities={}, relations=())
        r = Relation(name="ON", src="a", dst="b")
        ws2 = ws1.with_relation(r)
        self.assertEqual(len(ws1.relations), 0)
        self.assertEqual(len(ws2.relations), 1)
        self.assertEqual(ws2.relations[0], r)


if __name__ == "__main__":
    unittest.main()
