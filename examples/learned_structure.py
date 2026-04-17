"""Learned causal structure: compare PC, FCI, and GES against hand-built graph."""

import numpy as np

from telos import Entity, Relation, WorldState, CAWAAgent
from telos.physics import ALL_PRIMITIVES
from telos.structure_learner import (
    generate_samples,
    learn_graph,
    compare_graphs,
    has_latent_edges,
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


def _print_edges(graph, label):
    edges = [(e.parents, e.effect, e.label) for e in graph.all_edges()]
    print(f"\n{label} ({len(edges)} edges):")
    if edges:
        for parents, effect, lbl in edges:
            parent_str = ", ".join(parents) if parents else "(root)"
            print(f"  {parent_str} -> {effect}   [{lbl}]")
    else:
        print("  (none)")


def run() -> None:
    print("=== Example: causal structure learning ===")

    world = scene()

    # Ground-truth graph.
    agent = CAWAAgent()
    agent.perceive(world)
    ground_truth = agent.build_causal_graph()
    _print_edges(ground_truth, "Hand-built (ground truth)")

    # --- Linear samples + PC algorithm ---
    print("\n--- PC algorithm (linear data, Fisher-Z test) ---")
    samples, names = generate_samples(world, primitives=ALL_PRIMITIVES, n=1000, seed=42)
    print(f"Samples: {samples.shape[0]} x {samples.shape[1]}, variables: {', '.join(names)}")

    pc_graph = learn_graph(samples, names, alpha=0.05, method="pc")
    _print_edges(pc_graph, "PC learned")
    metrics = compare_graphs(pc_graph, ground_truth)
    print(f"  Precision={metrics['precision']:.2f}  Recall={metrics['recall']:.2f}  F1={metrics['f1']:.2f}")

    # --- Nonlinear samples + KCI test ---
    print("\n--- PC algorithm (nonlinear data, KCI test) ---")
    nl_samples, nl_names = generate_samples(
        world, primitives=ALL_PRIMITIVES, n=300, seed=42, nonlinear=True,
    )
    kci_graph = learn_graph(nl_samples, nl_names, alpha=0.05, method="pc", indep_test="kci")
    _print_edges(kci_graph, "PC+KCI learned (nonlinear)")
    metrics = compare_graphs(kci_graph, ground_truth)
    print(f"  Precision={metrics['precision']:.2f}  Recall={metrics['recall']:.2f}  F1={metrics['f1']:.2f}")

    # --- FCI (latent confounder detection) ---
    print("\n--- FCI algorithm (latent confounder detection) ---")
    # Simulate a latent confounder: L causes both X and Y.
    rng = np.random.default_rng(42)
    n = 2000
    latent = rng.standard_normal(n)
    x = latent * 2.0 + rng.standard_normal(n) * 0.2
    y = latent * 3.0 + rng.standard_normal(n) * 0.2
    z = x * 1.5 + rng.standard_normal(n) * 0.2
    fci_data = np.column_stack([x, y, z])
    fci_names = ["X", "Y", "Z"]

    fci_graph = learn_graph(fci_data, fci_names, alpha=0.05, method="fci")
    _print_edges(fci_graph, "FCI learned")
    latent_pairs = has_latent_edges(fci_graph)
    if latent_pairs:
        print(f"  Latent confounders detected: {latent_pairs}")
    else:
        print("  No latent confounders detected (may appear as bidirected edges)")

    # --- GES (score-based, better orientation) ---
    print("\n--- GES algorithm (score-based) ---")
    rng2 = np.random.default_rng(7)
    n2 = 3000
    a = rng2.standard_normal(n2)
    b = rng2.standard_normal(n2)
    c = a * 2 + b * 3 + rng2.standard_normal(n2) * 0.1
    ges_data = np.column_stack([a, b, c])
    ges_names = ["A", "B", "C"]

    ges_graph = learn_graph(ges_data, ges_names, method="ges")
    _print_edges(ges_graph, "GES learned (v-structure A->C<-B)")
    metrics = compare_graphs(ges_graph, ges_graph)  # self-comparison
    print(f"  Edges found: {len(ges_graph.all_edges())}")


if __name__ == "__main__":
    run()
