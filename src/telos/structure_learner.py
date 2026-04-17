"""Learn causal structure from data via the PC algorithm (causal-learn)."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .causal_graph import CausalEdge, CausalGraph
from .physics import apply_all, ALL_PRIMITIVES
from .world import WorldState


def generate_samples(
    world: WorldState,
    primitives: list[Callable[[WorldState], list[CausalEdge]]] | None = None,
    n: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """Perturb root variables randomly, propagate through physics edges.

    Root variables (those whose physics edges have no parents) are sampled
    from a standard normal distribution.  Non-root variables are computed
    from a linear combination of their parents plus small Gaussian noise,
    preserving the causal structure while giving the PC algorithm enough
    signal to recover edges.

    Returns ``(samples, variable_names)`` where *samples* has shape
    ``(n, num_vars)`` and *variable_names* lists each column's name.
    """
    primitives = primitives if primitives is not None else ALL_PRIMITIVES

    edges = apply_all(world, primitives)
    if not edges:
        return np.empty((n, 0), dtype=float), []

    # Collect all variable names from the edges.
    all_vars: set[str] = set()
    for edge in edges:
        all_vars.add(edge.effect)
        for p in edge.parents:
            all_vars.add(p)

    variable_names = sorted(all_vars)
    num_vars = len(variable_names)
    var_idx = {name: i for i, name in enumerate(variable_names)}

    # Separate root variables (edges with no parents) from non-root.
    root_vars: set[str] = set()
    nonroot_edges: list[CausalEdge] = []
    for edge in edges:
        if not edge.parents:
            root_vars.add(edge.effect)
        else:
            nonroot_edges.append(edge)

    # Build a lightweight CausalGraph for topological ordering of non-roots.
    topo_graph = CausalGraph()
    for var in variable_names:
        topo_graph.add_variable(var, initial=0.0)
    for edge in nonroot_edges:
        topo_graph.add_mechanism(
            edge.effect, list(edge.parents), edge.mechanism, edge.label,
        )
    topo_order = topo_graph._topological_order()

    # Parent map: for each non-root var, which parents does it depend on?
    parent_map: dict[str, list[str]] = {}
    for edge in nonroot_edges:
        parent_map[edge.effect] = list(edge.parents)

    rng = np.random.default_rng(seed)
    samples = np.empty((n, num_vars), dtype=float)

    for row in range(n):
        vals: dict[str, float] = {}
        # Sample root variables from standard normal.
        for var in variable_names:
            if var in root_vars:
                vals[var] = float(rng.standard_normal())

        # Propagate non-root variables in topological order.
        for var in topo_order:
            if var in root_vars:
                continue
            parents = parent_map.get(var)
            if parents is None:
                # Variable appears in the graph but has no edge — treat as root.
                vals[var] = float(rng.standard_normal())
                continue
            # Linear combination of parent values plus noise.
            val = sum(vals.get(p, 0.0) for p in parents) + rng.standard_normal() * 0.1
            vals[var] = val

        for name in variable_names:
            samples[row, var_idx[name]] = vals.get(name, 0.0)

    return samples, variable_names


def learn_graph(
    samples: np.ndarray,
    variable_names: list[str],
    alpha: float = 0.05,
) -> CausalGraph:
    """Run the PC algorithm and convert the result to a telos CausalGraph.

    Uses Fisher-Z conditional independence test from *causal-learn*.
    Directed edges are preserved as-is.  Undirected edges (where PC cannot
    determine orientation) are added in both directions.
    """
    from causallearn.search.ConstraintBased.PC import pc

    graph = CausalGraph()
    for name in variable_names:
        graph.add_variable(name)

    n_vars = len(variable_names)
    if n_vars < 2 or samples.shape[0] < 3:
        return graph

    result = pc(
        samples,
        alpha=alpha,
        indep_test="fisherz",
        node_names=variable_names,
        show_progress=False,
    )

    adj = result.G.graph  # shape (n_vars, n_vars)

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            val_ij = adj[i, j]
            val_ji = adj[j, i]

            if val_ij == -1 and val_ji == 1:
                # Directed edge i -> j
                graph.add_mechanism(
                    variable_names[j],
                    [variable_names[i]],
                    mechanism=lambda p, _p=variable_names[i]: p[_p],
                    label=f"learned:{variable_names[i]}->{variable_names[j]}",
                )
            elif val_ij == 1 and val_ji == -1:
                # Directed edge j -> i
                graph.add_mechanism(
                    variable_names[i],
                    [variable_names[j]],
                    mechanism=lambda p, _p=variable_names[j]: p[_p],
                    label=f"learned:{variable_names[j]}->{variable_names[i]}",
                )
            elif val_ij == -1 and val_ji == -1:
                # Undirected edge i -- j: add in both directions.
                graph.add_mechanism(
                    variable_names[j],
                    [variable_names[i]],
                    mechanism=lambda p, _p=variable_names[i]: p[_p],
                    label=f"learned:{variable_names[i]}--{variable_names[j]}",
                )
                graph.add_mechanism(
                    variable_names[i],
                    [variable_names[j]],
                    mechanism=lambda p, _p=variable_names[j]: p[_p],
                    label=f"learned:{variable_names[j]}--{variable_names[i]}",
                )

    return graph


def compare_graphs(
    learned: CausalGraph,
    ground_truth: CausalGraph,
) -> dict[str, float]:
    """Compare learned and ground-truth graphs by directed edge sets.

    Returns ``{"precision": ..., "recall": ..., "f1": ...}``.

    Each edge is represented as a ``(parent, effect)`` pair.  Multi-parent
    edges are expanded into one pair per parent.
    """

    def _edge_set(g: CausalGraph) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for edge in g.all_edges():
            for parent in edge.parents:
                pairs.add((parent, edge.effect))
        return pairs

    learned_edges = _edge_set(learned)
    truth_edges = _edge_set(ground_truth)

    if not learned_edges and not truth_edges:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    tp = len(learned_edges & truth_edges)

    precision = tp / len(learned_edges) if learned_edges else 0.0
    recall = tp / len(truth_edges) if truth_edges else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {"precision": precision, "recall": recall, "f1": f1}
