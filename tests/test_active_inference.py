import unittest
from telos.active_inference import Action, pragmatic_value, epistemic_value, Plan, select_action
from telos.causal_graph import CausalGraph


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


class TestSelectAction(unittest.TestCase):
    def _build_graph(self):
        # Simple graph: sealed → spill → damage.
        g = CausalGraph()
        g.add_variable("sealed", initial=False)
        g.add_variable("spill")
        g.add_variable("damage")
        g.add_mechanism("spill", ["sealed"], lambda p: not p["sealed"], label="containment")
        g.add_mechanism("damage", ["spill"], lambda p: p["spill"], label="liquid_damage")
        return g

    def test_select_action_prefers_lower_efe(self):
        g = self._build_graph()
        actions = [
            Action(name="do_nothing", effects={}, description="leave as is"),
            Action(name="seal_cup", effects={"sealed": True}, description="put a lid on"),
        ]
        goal = {"damage": False}
        plan = select_action(g, actions, goal)
        self.assertIsInstance(plan, Plan)
        self.assertEqual(plan.action.name, "seal_cup")

    def test_plan_contains_causal_chain_and_counterfactuals(self):
        g = self._build_graph()
        actions = [
            Action(name="do_nothing", effects={}),
            Action(name="seal_cup", effects={"sealed": True}),
        ]
        goal = {"damage": False}
        plan = select_action(g, actions, goal)
        self.assertTrue(any(e.label for e in plan.causal_chain))
        self.assertIn("seal_cup", plan.counterfactuals)
        self.assertIn("do_nothing", plan.counterfactuals)


if __name__ == "__main__":
    unittest.main()
