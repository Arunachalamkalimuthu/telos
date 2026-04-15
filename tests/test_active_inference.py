import unittest
from cawa.active_inference import Action, pragmatic_value, epistemic_value


class TestAction(unittest.TestCase):
    def test_action_stores_name_effects_description(self):
        a = Action(name="seal_cup", effects={"sealed": True}, description="put a lid on the cup")
        self.assertEqual(a.name, "seal_cup")
        self.assertEqual(a.effects["sealed"], True)


class TestPragmaticValue(unittest.TestCase):
    def test_pragmatic_value_is_zero_when_state_matches_goal(self):
        state = {"laptop_damaged": False, "cup_sealed": True}
        goal = {"laptop_damaged": False}
        self.assertEqual(pragmatic_value(state, goal), 0.0)

    def test_pragmatic_value_is_negative_when_state_violates_goal(self):
        state = {"laptop_damaged": True}
        goal = {"laptop_damaged": False}
        self.assertLess(pragmatic_value(state, goal), 0.0)

    def test_pragmatic_value_ignores_variables_not_in_goal(self):
        state = {"laptop_damaged": False, "random_var": "anything"}
        goal = {"laptop_damaged": False}
        self.assertEqual(pragmatic_value(state, goal), 0.0)


class TestEpistemicValue(unittest.TestCase):
    def test_epistemic_value_is_higher_when_more_uncertainty_resolved(self):
        state_before = {"a": "UNKNOWN", "b": "UNKNOWN", "c": 1}
        state_after_more_info = {"a": 10, "b": 20, "c": 1}
        state_after_less_info = {"a": 10, "b": "UNKNOWN", "c": 1}
        self.assertGreater(
            epistemic_value(state_before, state_after_more_info),
            epistemic_value(state_before, state_after_less_info),
        )


if __name__ == "__main__":
    unittest.main()
