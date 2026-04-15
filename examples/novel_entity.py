"""Unknown entity falls off unknown surface: physics applies; unknowns flagged honestly."""

from cawa import UNKNOWN, CAWAAgent, Entity, Relation, WorldState


def scene():
    frambulator = Entity(
        id="frambulator",
        type="frambulator",
        properties={
            "mass": 0.5,                # enough to engage gravity
            "glorbic_index": "high",    # unknown semantics
            # orientation, sealed, contains, fragile, impact_threshold — all UNKNOWN.
        },
    )
    zibbly = Entity(
        id="zibbly",
        type="zibbly",
        properties={},  # we only know it could be fallen off, not what it is.
    )
    return WorldState(
        entities={"frambulator": frambulator, "zibbly": zibbly},
        relations=(),  # no ON relation → frambulator is unsupported.
    )


def run():
    print("=== Example: 'A frambulator with high glorbic_index falls off a zibbly' ===")
    world = scene()
    agent = CAWAAgent()
    agent.perceive(world)
    graph = agent.build_causal_graph()

    print("Variables in constructed causal graph:")
    for var in sorted(graph.variables()):
        print(f"  {var}")

    fall_state = graph.propagate()
    print()
    print("Physics applies regardless of entity identity:")
    print(f"  frambulator.falls = {fall_state.get('frambulator.falls')}")

    print()
    print("Honest uncertainty flags:")
    framb = world.get_entity("frambulator")
    for prop in ("orientation", "sealed", "contains", "fragile", "impact_threshold"):
        value = framb.get(prop)
        flag = "UNKNOWN" if value is UNKNOWN else repr(value)
        print(f"  frambulator.{prop} = {flag}")
    glorbic = framb.get("glorbic_index")
    print(f"  frambulator.glorbic_index = {glorbic!r}  — semantics UNKNOWN")
    print()
    print("Conclusion: the frambulator falls (gravity applies).")
    print("            Impact consequences are not predictable because fragility,")
    print("            impact_threshold, and the meaning of glorbic_index are unknown.")
    print("            CAWA refuses to hallucinate; it flags the gap.")


if __name__ == "__main__":
    run()
