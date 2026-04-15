import unittest
from cawa.world import Entity, WorldState
from cawa.theory_of_mind import AgentMind, predict_action


class TestAgentMind(unittest.TestCase):
    def test_agent_mind_stores_beliefs_goals_capabilities_actions(self):
        beliefs = WorldState(entities={}, relations=())
        m = AgentMind(
            id="child",
            beliefs=beliefs,
            goals=[{"type": "reach", "target": "parent"}],
            capabilities=frozenset({"visual"}),
            actions=["run", "stop", "wait"],
        )
        self.assertEqual(m.id, "child")
        self.assertIn("visual", m.capabilities)
        self.assertIn("run", m.actions)


class TestPredictAction(unittest.TestCase):
    def test_prediction_uses_agent_beliefs_not_ground_truth(self):
        # Ground truth: road is dangerous. Agent's belief: road is safe (it's a child).
        agent_belief_world = WorldState(
            entities={"road": Entity(id="road", type="road", properties={"danger": False})},
            relations=(),
        )
        mind = AgentMind(
            id="child",
            beliefs=agent_belief_world,
            goals=[{"type": "reach", "target": "parent"}],
            capabilities=frozenset({"visual"}),
            actions=["run_toward_parent", "stop"],
        )
        ground_truth = WorldState(
            entities={"road": Entity(id="road", type="road", properties={"danger": True})},
            relations=(),
        )
        action = predict_action(mind, ground_truth)
        self.assertEqual(action, "run_toward_parent")

    def test_prediction_defaults_to_first_action_when_no_goal_match(self):
        mind = AgentMind(
            id="x",
            beliefs=WorldState(entities={}, relations=()),
            goals=[],
            capabilities=frozenset(),
            actions=["wait"],
        )
        self.assertEqual(predict_action(mind, WorldState(entities={}, relations=())), "wait")


if __name__ == "__main__":
    unittest.main()
