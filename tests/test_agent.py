import unittest
from cawa.world import Entity, Relation, WorldState
from cawa.active_inference import Action
from cawa.agent import CAWAAgent


class TestCAWAAgent(unittest.TestCase):
    def test_perceive_and_build_graph_applies_physics(self):
        cup = Entity(
            id="cup",
            type="cup",
            properties={
                "mass": 0.2,
                "orientation": "inverted",
                "sealed": False,
                "contains": "coffee",
            },
        )
        ws = WorldState(entities={"cup": cup}, relations=())
        agent = CAWAAgent()
        agent.perceive(ws)
        graph = agent.build_causal_graph()
        self.assertIn("cup.contents_escape", graph.variables())

    def test_plan_returns_plan_with_explanation(self):
        cup = Entity(
            id="cup",
            type="cup",
            properties={
                "mass": 0.2,
                "orientation": "inverted",
                "sealed": False,
                "contains": "coffee",
            },
        )
        ws = WorldState(entities={"cup": cup}, relations=())
        agent = CAWAAgent()
        agent.perceive(ws)
        graph = agent.build_causal_graph()
        actions = [
            Action(name="do_nothing", effects={}),
            Action(name="seal_cup", effects={"cup.contents_escape": False}),
        ]
        plan = agent.plan(graph, goal={"cup.contents_escape": False}, actions=actions)
        self.assertEqual(plan.action.name, "seal_cup")
        explanation = agent.explain(plan)
        self.assertIn("seal_cup", explanation)
        self.assertIn("containment", explanation)


if __name__ == "__main__":
    unittest.main()
