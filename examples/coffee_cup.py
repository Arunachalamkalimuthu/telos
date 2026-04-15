"""Coffee cup on laptop: physics + counterfactuals."""

from cawa import (
    Action,
    CAWAAgent,
    Entity,
    Relation,
    WorldState,
)


def scene() -> WorldState:
    cup = Entity(
        id="cup",
        type="cup",
        properties={
            "mass": 0.25,
            "orientation": "inverted",
            "sealed": False,
            "contains": "coffee",
            "material": "ceramic",
        },
    )
    coffee = Entity(
        id="coffee",
        type="liquid",
        properties={"conductive": True},
    )
    laptop = Entity(
        id="laptop",
        type="laptop",
        properties={"electronic": True, "mass": 1.8},
    )
    return WorldState(
        entities={"cup": cup, "coffee": coffee, "laptop": laptop},
        relations=(
            Relation("ON", "cup", "laptop"),
            Relation("WILL_CONTACT", "coffee", "laptop"),
        ),
    )


def run() -> None:
    print("=== Example: inverted coffee cup on a laptop ===")
    world = scene()
    agent = CAWAAgent()
    agent.perceive(world)
    graph = agent.build_causal_graph()

    # Goal: the laptop is not damaged.
    goal = {"laptop.damaged": False}

    actions = [
        Action(name="do_nothing", effects={}, description="leave the scene alone"),
        Action(
            name="seal_cup",
            effects={"cup.contents_escape": False},
            description="put a lid on the cup",
        ),
        Action(
            name="turn_cup_upright",
            effects={"cup.contents_escape": False},
            description="rotate the cup so the opening faces up",
        ),
    ]

    plan = agent.plan(graph, goal=goal, actions=actions)
    print(agent.explain(plan))
    print()
    print("Counterfactual reasoning:")
    print("  If the cup were sealed:          no spill → no damage.")
    print("  If gravity did not apply:        no spill → no damage.")
    print("  If cup material were absorbent:  no spill → no damage.")


if __name__ == "__main__":
    run()
