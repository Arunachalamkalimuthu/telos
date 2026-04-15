import unittest
from telos.world import Entity, WorldState
from telos.theory_of_mind import AgentMind, predict_action, intervention_effect, Intervention


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


class TestInterventionEffect(unittest.TestCase):
    def _mind(self, capabilities):
        return AgentMind(
            id="child",
            beliefs=WorldState(entities={}, relations=()),
            goals=[],
            capabilities=frozenset(capabilities),
            actions=["run"],
        )

    def test_verbal_signal_reaches_hearing_agent(self):
        mind = self._mind({"visual", "auditory"})
        self.assertTrue(intervention_effect(Intervention(kind="verbal", content="stop"), mind))

    def test_verbal_signal_does_not_reach_deaf_agent(self):
        mind = self._mind({"visual"})
        self.assertFalse(intervention_effect(Intervention(kind="verbal", content="stop"), mind))

    def test_visual_signal_does_not_reach_blind_agent(self):
        mind = self._mind({"auditory"})
        self.assertFalse(intervention_effect(Intervention(kind="visual", content="wave"), mind))

    def test_physical_intervention_always_effective(self):
        mind = self._mind(set())
        self.assertTrue(intervention_effect(Intervention(kind="physical", content="intercept"), mind))


if __name__ == "__main__":
    unittest.main()
