"""Learn causal structure from data via constraint-based and score-based algorithms.

Supports three discovery algorithms from causal-learn:
- **PC** (default): constraint-based, recovers skeleton + partial orientation
- **FCI**: constraint-based, handles latent confounders (outputs PAG)
- **GES**: score-based, better edge orientation than PC on observational data

Supports two independence tests:
- **fisherz** (default): fast, assumes linear relationships
- **kci**: kernel-based, handles nonlinear relationships (slower)
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

from .causal_graph import CausalEdge, CausalGraph
from .physics import apply_all, ALL_PRIMITIVES
from .world import WorldState


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

def generate_samples(
    world: WorldState,
    primitives: list[Callable[[WorldState], list[CausalEdge]]] | None = None,
    n: int = 500,
    seed: int = 42,
    nonlinear: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Generate observational data by perturbing a world state and running physics.

    Root variables (those whose physics edges have no parents) are sampled
    from a standard normal distribution.  Non-root variables are computed
    from their parents plus noise.

    Args:
        world: The scene to sample from.
        primitives: Physics primitives to apply. Defaults to ALL_PRIMITIVES.
        n: Number of samples.
        seed: Random seed for reproducibility.
        nonlinear: If True, use nonlinear transformations (tanh, quadratic)
            between parent and child variables. If False, use linear combinations.

    Returns:
        (samples, variable_names) where samples has shape (n, num_vars).
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

    # Generate stable coefficients per edge for consistency across samples.
    edge_coeffs: dict[str, list[float]] = {}
    for var, parents in parent_map.items():
        edge_coeffs[var] = [rng.uniform(1.0, 3.0) for _ in parents]

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
                vals[var] = float(rng.standard_normal())
                continue

            coeffs = edge_coeffs[var]
            noise = rng.standard_normal() * 0.1

            if nonlinear:
                # Nonlinear: tanh of weighted sum + quadratic interaction.
                weighted = sum(c * vals.get(p, 0.0) for c, p in zip(coeffs, parents))
                val = np.tanh(weighted) + 0.3 * weighted ** 2 + noise
            else:
                # Linear combination of parent values.
                val = sum(c * vals.get(p, 0.0) for c, p in zip(coeffs, parents)) + noise

            vals[var] = val

        for name in variable_names:
            samples[row, var_idx[name]] = vals.get(name, 0.0)

    return samples, variable_names


# ---------------------------------------------------------------------------
# Graph learning
# ---------------------------------------------------------------------------

def learn_graph(
    samples: np.ndarray,
    variable_names: list[str],
    alpha: float = 0.05,
    method: Literal["pc", "fci", "ges"] = "pc",
    indep_test: Literal["fisherz", "kci"] = "fisherz",
) -> CausalGraph:
    """Discover causal structure from observational data.

    Args:
        samples: (n_samples, n_variables) array of observations.
        variable_names: Name for each column.
        alpha: Significance level for conditional independence tests (PC/FCI).
        method: Discovery algorithm.
            - "pc": Peter-Clark algorithm. Fast, assumes no latent confounders.
            - "fci": Fast Causal Inference. Handles latent confounders, outputs PAG.
            - "ges": Greedy Equivalence Search. Score-based, better orientation.
        indep_test: Independence test for PC/FCI.
            - "fisherz": assumes linear Gaussian. Fast.
            - "kci": kernel-based, handles nonlinear. Slower.

    Returns:
        A telos CausalGraph. Directed edges become mechanisms.
        Undirected edges (PC) are added in both directions.
        Bidirected edges (FCI, indicating latent confounders) are stored
        with a "latent:" label prefix.
    """
    graph = CausalGraph()
    for name in variable_names:
        graph.add_variable(name)

    n_vars = len(variable_names)
    if n_vars < 2 or samples.shape[0] < 3:
        return graph

    if method == "pc":
        return _learn_pc(samples, variable_names, alpha, indep_test, graph)
    elif method == "fci":
        return _learn_fci(samples, variable_names, alpha, indep_test, graph)
    elif method == "ges":
        return _learn_ges(samples, variable_names, graph)
    else:
        raise ValueError(f"unknown method: {method!r}")


def _learn_pc(
    samples: np.ndarray,
    variable_names: list[str],
    alpha: float,
    indep_test: str,
    graph: CausalGraph,
) -> CausalGraph:
    """PC algorithm: constraint-based, no latent confounders."""
    from causallearn.search.ConstraintBased.PC import pc

    result = pc(
        samples,
        alpha=alpha,
        indep_test=indep_test,
        node_names=variable_names,
        show_progress=False,
    )
    adj = result.G.graph
    n_vars = len(variable_names)

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            val_ij = adj[i, j]
            val_ji = adj[j, i]

            if val_ij == -1 and val_ji == 1:
                # Directed: i -> j
                _add_learned_edge(graph, variable_names[i], variable_names[j], "pc")
            elif val_ij == 1 and val_ji == -1:
                # Directed: j -> i
                _add_learned_edge(graph, variable_names[j], variable_names[i], "pc")
            elif val_ij == -1 and val_ji == -1:
                # Undirected: add both directions.
                _add_learned_edge(graph, variable_names[i], variable_names[j], "pc:undirected")
                _add_learned_edge(graph, variable_names[j], variable_names[i], "pc:undirected")

    return graph


def _learn_fci(
    samples: np.ndarray,
    variable_names: list[str],
    alpha: float,
    indep_test: str,
    graph: CausalGraph,
) -> CausalGraph:
    """FCI algorithm: handles latent confounders, outputs PAG.

    PAG edge types in causal-learn adjacency matrix:
    - adj[i,j]=-1, adj[j,i]=1  → i -> j  (directed)
    - adj[i,j]=1, adj[j,i]=-1  → j -> i  (directed)
    - adj[i,j]=1, adj[j,i]=1   → i <-> j (bidirected, latent confounder)
    - adj[i,j]=2, adj[j,i]=1   → i o-> j (partially oriented)
    - adj[i,j]=2, adj[j,i]=2   → i o-o j (unoriented)
    """
    from causallearn.search.ConstraintBased.FCI import fci

    g, edges = fci(
        samples,
        independence_test_method=indep_test,
        alpha=alpha,
        show_progress=False,
    )
    adj = g.graph
    n_vars = len(variable_names)

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            val_ij = adj[i, j]
            val_ji = adj[j, i]

            if val_ij == 0 and val_ji == 0:
                continue

            if val_ij == -1 and val_ji == 1:
                _add_learned_edge(graph, variable_names[i], variable_names[j], "fci")
            elif val_ij == 1 and val_ji == -1:
                _add_learned_edge(graph, variable_names[j], variable_names[i], "fci")
            elif val_ij == 1 and val_ji == 1:
                # Bidirected: latent confounder between i and j.
                _add_learned_edge(graph, variable_names[i], variable_names[j], "latent:fci")
                _add_learned_edge(graph, variable_names[j], variable_names[i], "latent:fci")
            elif val_ij == 2 and val_ji == 1:
                # Partially oriented: o-> (i toward j).
                _add_learned_edge(graph, variable_names[i], variable_names[j], "fci:partial")
            elif val_ij == 1 and val_ji == 2:
                _add_learned_edge(graph, variable_names[j], variable_names[i], "fci:partial")
            elif val_ij == 2 and val_ji == 2:
                # Unoriented circle-circle.
                _add_learned_edge(graph, variable_names[i], variable_names[j], "fci:unoriented")
                _add_learned_edge(graph, variable_names[j], variable_names[i], "fci:unoriented")

    return graph


def _learn_ges(
    samples: np.ndarray,
    variable_names: list[str],
    graph: CausalGraph,
) -> CausalGraph:
    """GES algorithm: score-based, often orients more edges than PC."""
    from causallearn.search.ScoreBased.GES import ges

    result = ges(samples, score_func="local_score_CV_general")
    adj = result["G"].graph
    n_vars = len(variable_names)

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            val_ij = adj[i, j]
            val_ji = adj[j, i]

            if val_ij == -1 and val_ji == 1:
                _add_learned_edge(graph, variable_names[i], variable_names[j], "ges")
            elif val_ij == 1 and val_ji == -1:
                _add_learned_edge(graph, variable_names[j], variable_names[i], "ges")
            elif val_ij == -1 and val_ji == -1:
                _add_learned_edge(graph, variable_names[i], variable_names[j], "ges:undirected")
                _add_learned_edge(graph, variable_names[j], variable_names[i], "ges:undirected")

    return graph


def _add_learned_edge(
    graph: CausalGraph,
    parent: str,
    child: str,
    label_prefix: str,
) -> None:
    """Add a directed mechanism edge from parent to child."""
    graph.add_mechanism(
        child,
        [parent],
        mechanism=lambda p, _p=parent: p[_p],
        label=f"{label_prefix}:{parent}->{child}",
    )


# ---------------------------------------------------------------------------
# Graph comparison
# ---------------------------------------------------------------------------

def compare_graphs(
    learned: CausalGraph,
    ground_truth: CausalGraph,
) -> dict[str, float]:
    """Compare learned and ground-truth graphs by directed edge sets.

    Returns ``{"precision": ..., "recall": ..., "f1": ...}``.

    Each edge is represented as a ``(parent, effect)`` pair.  Multi-parent
    edges are expanded into one pair per parent.  Edges with "latent:" labels
    are excluded from comparison (they represent hidden confounders, not
    direct causal links).
    """

    def _edge_set(g: CausalGraph, exclude_latent: bool = True) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for edge in g.all_edges():
            if exclude_latent and edge.label.startswith("latent:"):
                continue
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


def has_latent_edges(graph: CausalGraph) -> list[tuple[str, str]]:
    """Return pairs of variables connected by latent confounder edges."""
    latent_pairs: list[tuple[str, str]] = []
    for edge in graph.all_edges():
        if edge.label.startswith("latent:"):
            for parent in edge.parents:
                latent_pairs.append((parent, edge.effect))
    return latent_pairs
