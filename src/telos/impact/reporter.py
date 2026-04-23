"""Rich terminal reporter for impact analysis results."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.tree import Tree


def _risk_color(risk: float) -> str:
    """Return a Rich color name based on risk severity."""
    if risk >= 0.9:
        return "red"
    if risk >= 0.6:
        return "yellow"
    return "green"


def format_impact(result: dict) -> str:
    """Render an impact-analysis result as a Rich tree view.

    Groups affected nodes by *via* field to build a hierarchical tree.
    Risk scores are coloured red (>=0.9), yellow (>=0.6), or green.
    """
    console = Console(record=True, width=100)

    target = result["target"]
    affected = result.get("affected", [])

    tree = Tree(f"[bold]{target}[/bold]")

    # Group affected entries by their "via" field.
    children_of: dict[str, list[dict]] = {}
    for entry in affected:
        via = entry.get("via", target) or target
        children_of.setdefault(via, []).append(entry)

    # Recursive helper to build tree branches.
    added: set[str] = set()

    def _add_children(parent_branch: Tree, parent_id: str) -> None:
        for entry in children_of.get(parent_id, []):
            node_id = entry["node_id"]
            if node_id in added:
                continue
            added.add(node_id)
            color = _risk_color(entry["risk"])
            label = (
                f"[{color}]{node_id}[/{color}] "
                f"risk=[{color}]{entry['risk']:.2f}[/{color}] "
                f"depth={entry['depth']} "
                f"({entry['edge_kind']})"
            )
            branch = parent_branch.add(label)
            _add_children(branch, node_id)

    _add_children(tree, target)

    # Add any remaining nodes not reachable from the tree root (edge cases).
    for entry in affected:
        if entry["node_id"] not in added:
            color = _risk_color(entry["risk"])
            label = (
                f"[{color}]{entry['node_id']}[/{color}] "
                f"risk=[{color}]{entry['risk']:.2f}[/{color}] "
                f"depth={entry['depth']} "
                f"({entry['edge_kind']})"
            )
            tree.add(label)

    console.print(tree)

    # Hottest path
    hottest = result.get("hottest_path", [])
    if hottest:
        console.print()
        console.print(
            "[bold]Hottest path:[/bold] " + " -> ".join(hottest)
        )

    # Files affected
    files = result.get("files_affected", [])
    if files:
        console.print()
        console.print(f"[bold]Files affected ({len(files)}):[/bold]")
        for f in files:
            console.print(f"  {f}")

    return console.export_text()


def format_hotspots(hotspots: list[dict]) -> str:
    """Render a hotspots list as a Rich table."""
    console = Console(record=True, width=100)

    table = Table(title="Hotspots")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Function", style="bold")
    table.add_column("File")
    table.add_column("Dependents", justify="right")
    table.add_column("Risk", justify="right")

    for i, entry in enumerate(hotspots, 1):
        dep_count = entry["dependent_count"]
        # Simple risk heuristic: more dependents = higher risk.
        if dep_count >= 10:
            risk_label = "[red]high[/red]"
        elif dep_count >= 5:
            risk_label = "[yellow]medium[/yellow]"
        else:
            risk_label = "[green]low[/green]"

        table.add_row(
            str(i),
            entry["name"],
            entry["file_path"],
            str(dep_count),
            risk_label,
        )

    console.print(table)
    return console.export_text()


def format_counterfactual(result: dict) -> str:
    """Render a counterfactual comparison."""
    console = Console(record=True, width=100)

    target = result["target"]
    intervention = result["intervention_at"]
    without_count = result["without_count"]
    with_count = result["with_count"]
    reduction = result["reduction"]

    if without_count > 0:
        pct = (reduction / without_count) * 100
    else:
        pct = 0.0

    console.print(f"[bold]Counterfactual Analysis[/bold]")
    console.print(f"  Target:       {target}")
    console.print(f"  Intervention: {intervention}")
    console.print()
    console.print(f"  Without fix:  [red]{without_count}[/red] affected nodes")
    console.print(f"  With fix:     [green]{with_count}[/green] affected nodes")
    console.print(
        f"  Reduction:    [bold]{reduction}[/bold] nodes saved ({pct:.0f}%)"
    )

    return console.export_text()


def format_info(info: dict) -> str:
    """Render graph metadata as a simple key-value display."""
    console = Console(record=True, width=100)

    console.print("[bold]Graph Info[/bold]")
    for key, value in info.items():
        console.print(f"  {key}: {value}")

    return console.export_text()
