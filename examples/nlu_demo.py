"""Natural-language understanding demo: text descriptions → WorldState + structured queries."""

from telos.nlu import parse_scene, parse_query


def _print_world(world) -> None:
    """Pretty-print entities and relations from a WorldState."""
    print("  Entities:")
    if world.entities:
        for eid, entity in world.entities.items():
            attrs = entity.properties.get("attributes", [])
            attrs_str = f"  attributes={attrs}" if attrs else ""
            print(f"    [{eid}]  type={entity.type}{attrs_str}")
    else:
        print("    (none)")

    print("  Relations:")
    if world.relations:
        for rel in world.relations:
            print(f"    {rel.name}(src={rel.src}, dst={rel.dst})")
    else:
        print("    (none)")


def _print_query(result: dict) -> None:
    """Pretty-print a structured query dict."""
    print(f"    type    : {result['type']}")
    print(f"    subject : {result['subject']}")
    print(f"    action  : {result['action']}")


def run() -> None:
    print("=== Example: NLU — natural language → scene + query ===")
    print()

    # ------------------------------------------------------------------
    # 1. Parse a simple scene
    # ------------------------------------------------------------------
    scene_text = "A coffee cup is on a laptop"
    print(f"Scene: \"{scene_text}\"")
    world = parse_scene(scene_text)
    _print_world(world)
    print()

    # ------------------------------------------------------------------
    # 2. Counterfactual query
    # ------------------------------------------------------------------
    cf_text = "What happens if the cup falls?"
    print(f"Query: \"{cf_text}\"")
    cf_result = parse_query(cf_text)
    _print_query(cf_result)
    print()

    # ------------------------------------------------------------------
    # 3. Prediction query
    # ------------------------------------------------------------------
    pred_text = "Will the laptop get damaged?"
    print(f"Query: \"{pred_text}\"")
    pred_result = parse_query(pred_text)
    _print_query(pred_result)
    print()

    # ------------------------------------------------------------------
    # 4. Full pipeline demo — richer scene
    # ------------------------------------------------------------------
    rich_scene = "A heavy ceramic cup is on the edge of a wooden table near an open laptop"
    print(f"Rich scene: \"{rich_scene}\"")
    rich_world = parse_scene(rich_scene)
    _print_world(rich_world)
    print()


if __name__ == "__main__":
    run()
