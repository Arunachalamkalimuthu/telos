import unittest
from cawa.world import UNKNOWN, Entity


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


if __name__ == "__main__":
    unittest.main()
