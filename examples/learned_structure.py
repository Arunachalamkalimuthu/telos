"""Coffee cup scene: compare PC-learned causal graph against hand-built graph."""

from telos import Entity, Relation, WorldState, CAWAAgent
from telos.physics import ALL_PRIMITIVES
from telos.structure_learner import generate_samples, learn_graph, compare_graphs


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
    print("=== Example: structure learner vs hand-built (coffee cup scene) ===")

    world = scene()

    # --- Ground-truth graph from the hand-built agent ---
    agent = CAWAAgent()
    agent.perceive(world)
    ground_truth = agent.build_causal_graph()

    # --- Data-driven graph via PC algorithm ---
    samples, variable_names = generate_samples(world, primitives=ALL_PRIMITIVES, n=1000, seed=42)

    print(f"\nSamples generated: {samples.shape[0]} observations × {samples.shape[1]} variables")
    print("Variables:", ", ".join(variable_names))

    learned = learn_graph(samples, variable_names, alpha=0.05)

    # --- Learned edges ---
    learned_edges = [(e.parents, e.effect, e.label) for e in learned.all_edges()]
    print(f"\nLearned edges ({len(learned_edges)}):")
    if learned_edges:
        for parents, effect, label in learned_edges:
            parent_str = ", ".join(parents) if parents else "(root)"
            print(f"  {parent_str} → {effect}   [{label}]")
    else:
        print("  (none)")

    # --- Ground-truth edges for reference ---
    truth_edges = [(e.parents, e.effect, e.label) for e in ground_truth.all_edges()]
    print(f"\nHand-built edges ({len(truth_edges)}):")
    for parents, effect, label in truth_edges:
        parent_str = ", ".join(parents) if parents else "(root)"
        print(f"  {parent_str} → {effect}   [{label}]")

    # --- Comparison metrics ---
    metrics = compare_graphs(learned, ground_truth)
    print("\nComparison metrics (learned vs hand-built):")
    print(f"  Precision : {metrics['precision']:.3f}")
    print(f"  Recall    : {metrics['recall']:.3f}")
    print(f"  F1        : {metrics['f1']:.3f}")


if __name__ == "__main__":
    run()
