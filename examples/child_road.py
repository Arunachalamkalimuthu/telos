"""Deaf child running toward a road: theory of mind + perceptual constraints."""

from cawa import (
    Action,
    AgentMind,
    Entity,
    Intervention,
    WorldState,
    intervention_effect,
    predict_action,
)


def scene():
    road = Entity(id="road", type="road", properties={"danger": True, "traffic_density": "high"})
    parent_position = Entity(id="parent", type="person", properties={"across_road": True})
    ground_truth = WorldState(entities={"road": road, "parent": parent_position}, relations=())

    # Child believes road is safe (they do not know it is dangerous).
    child_belief_road = Entity(id="road", type="road", properties={"danger": False})
    child_beliefs = WorldState(
        entities={"road": child_belief_road, "parent": parent_position}, relations=()
    )

    child_mind = AgentMind(
        id="child",
        beliefs=child_beliefs,
        goals=({"type": "reach", "target": "parent"},),
        capabilities=frozenset({"visual"}),  # deaf: no auditory.
        actions=("run_reach_parent", "stop"),
    )
    return ground_truth, child_mind


def run():
    print("=== Example: deaf child running toward a busy road ===")
    ground_truth, child = scene()

    predicted = predict_action(child, ground_truth)
    print(f"Child's capabilities: {sorted(child.capabilities)}  (no 'auditory' → deaf)")
    print(f"Child's beliefs differ from ground truth: child thinks road is safe.")
    print(f"Predicted child action (using child's beliefs, not ours): {predicted}")
    print()

    candidates = [
        (Action(name="shout_stop", effects={}, description="yell at the child to stop"),
         Intervention(kind="verbal", content="STOP")),
        (Action(name="wave_arms", effects={}, description="wave arms to get attention"),
         Intervention(kind="visual", content="WAVE")),
        (Action(name="sprint_intercept", effects={}, description="run and physically intercept"),
         Intervention(kind="physical", content="INTERCEPT")),
    ]

    print("Evaluating candidate interventions:")
    reachable: list[tuple[Action, Intervention]] = []
    for action, intervention in candidates:
        ok = intervention_effect(intervention, child)
        print(f"  {action.name}: intervention={intervention.kind}, reaches child={ok}")
        if ok:
            reachable.append((action, intervention))

    # Prefer a physical intervention over a sensory one: a running child may
    # not turn to look, but cannot be un-intercepted.
    best = None
    for action, intervention in reachable:
        if intervention.kind == "physical":
            best = action
            break
    if best is None and reachable:
        best = reachable[0][0]

    print()
    if best is None:
        print("No candidate action reaches the child — need to reconsider.")
    else:
        print(f"Chosen action: {best.name}")
        print(f"  description: {best.description}")
        print(f"  rationale: shout fails (child is deaf → no auditory channel);")
        print(f"             wave_arms and sprint_intercept both reach the child;")
        print(f"             sprint_intercept is selected as most reliable when child is running.")


if __name__ == "__main__":
    run()
