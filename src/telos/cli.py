"""telos CLI — causal impact analyzer for codebases."""

from __future__ import annotations

import os
import datetime

import typer
from rich.console import Console

from telos.code_parser.store import GraphStore
from telos.code_parser.graph_builder import GraphBuilder
from telos.impact.analyzer import ImpactAnalyzer
from telos.impact.counterfactual import CounterfactualAnalyzer
from telos.impact.reporter import (
    format_counterfactual,
    format_hotspots,
    format_impact,
    format_info,
)

app = typer.Typer(help="telos — causal impact analyzer for codebases")
console = Console()

_DB_REL = os.path.join(".telos", "graph.db")


def _db_path(path: str) -> str:
    return os.path.join(os.path.abspath(path), _DB_REL)


def _require_db(path: str) -> GraphStore:
    """Open an existing graph store or exit with an error."""
    db = _db_path(path)
    if not os.path.exists(db):
        console.print(
            f"[red]No telos graph found at {db}. "
            "Run [bold]telos init[/bold] first.[/red]"
        )
        raise typer.Exit(code=1)
    return GraphStore(db)


@app.command()
def init(
    path: str = typer.Argument(".", help="Root directory to scan"),
    force: bool = typer.Option(False, "--force", "-f", help="Clear existing graph first"),
) -> None:
    """Scan a codebase and build the dependency graph."""
    abs_path = os.path.abspath(path)
    db = _db_path(path)

    console.print(f"[bold]Scanning[/bold] {abs_path} ...")

    store = GraphStore(db)
    if force:
        store.clear()

    builder = GraphBuilder(store)
    summary = builder.scan_directory(abs_path)

    # Persist metadata.
    store.set_meta("last_scan", datetime.datetime.now().isoformat())
    store.set_meta("repo_root", abs_path)

    # Print summary.
    console.print(f"  Files scanned: {summary['files_scanned']}")
    console.print(f"  Nodes:         {summary['nodes']}")
    console.print(f"  Edges:         {summary['edges']}")

    # Top-5 hotspots.
    analyzer = ImpactAnalyzer(store)
    top5 = analyzer.hotspots(top_n=5)
    if top5:
        console.print()
        console.print(format_hotspots(top5))

    console.print()
    console.print("[green]Ready.[/green] Run: [bold]telos impact <file:function>[/bold]")

    store.close()


@app.command()
def impact(
    target: str = typer.Argument(..., help="Node id (e.g. src/app.py:main)"),
    path: str = typer.Option(".", "--path", "-p", help="Project root"),
    fix: str = typer.Option(None, "--fix", help="Intervention node for counterfactual"),
    depth: int = typer.Option(None, "--depth", "-d", help="Max traversal depth"),
    min_risk: float = typer.Option(0.0, "--min-risk", "-r", help="Minimum risk threshold"),
) -> None:
    """Analyze the impact of changing a node."""
    store = _require_db(path)

    if fix:
        analyzer = ImpactAnalyzer(store)
        cf = CounterfactualAnalyzer(store, analyzer)
        result = cf.analyze(target, intervention_at=fix)
        console.print(format_counterfactual(result))
    else:
        analyzer = ImpactAnalyzer(store)
        kwargs: dict = {"min_risk": min_risk}
        if depth is not None:
            kwargs["max_depth"] = depth
        result = analyzer.analyze(target, **kwargs)
        console.print(format_impact(result))

    store.close()


@app.command()
def hotspots(
    path: str = typer.Option(".", "--path", "-p", help="Project root"),
    top: int = typer.Option(10, "--top", "-n", help="Number of hotspots"),
) -> None:
    """Show the most depended-on nodes in the graph."""
    store = _require_db(path)
    analyzer = ImpactAnalyzer(store)
    result = analyzer.hotspots(top_n=top)
    console.print(format_hotspots(result))
    store.close()


@app.command()
def graph(
    target: str = typer.Argument(..., help="Node id to explore"),
    path: str = typer.Option(".", "--path", "-p", help="Project root"),
    depth: int = typer.Option(3, "--depth", "-d", help="Max traversal depth"),
) -> None:
    """Show impact tree for a node."""
    store = _require_db(path)
    analyzer = ImpactAnalyzer(store)
    result = analyzer.analyze(target, max_depth=depth)
    console.print(format_impact(result))
    store.close()


@app.command()
def info(
    path: str = typer.Option(".", "--path", "-p", help="Project root"),
) -> None:
    """Show graph metadata and statistics."""
    store = _require_db(path)
    stats = store.get_stats()
    meta: dict = dict(stats)
    # Add stored metadata.
    for key in ("last_scan", "repo_root"):
        val = store.get_meta(key)
        if val is not None:
            meta[key] = val
    console.print(format_info(meta))
    store.close()


if __name__ == "__main__":
    app()
