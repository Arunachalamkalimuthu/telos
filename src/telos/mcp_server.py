"""Telos MCP Server — expose causal impact analysis as tools for any LLM.

Run with: python -m telos.mcp_server
Or configure in Claude Code settings as an MCP server.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime

from mcp.server.fastmcp import FastMCP

from telos.code_parser.store import GraphStore
from telos.code_parser.graph_builder import GraphBuilder
from telos.impact.analyzer import ImpactAnalyzer
from telos.impact.counterfactual import CounterfactualAnalyzer
from telos.memory.project_memory import ProjectMemory
from telos.memory.cross_session_learner import CrossSessionLearner

mcp = FastMCP(
    "telos",
    instructions=(
        "Telos is a causal impact analyzer for codebases with persistent memory. "
        "It builds dependency graphs from source code, traces the full transitive "
        "impact of any change, runs counterfactual reasoning, and remembers WHY "
        "decisions were made across sessions — not as text, but as causal chains."
    ),
)


def _get_db_path(repo_path: str) -> str:
    return os.path.join(repo_path, ".telos", "graph.db")


def _require_store(repo_path: str) -> GraphStore:
    db_path = _get_db_path(repo_path)
    if not os.path.exists(db_path):
        raise ValueError(
            f"Telos not initialized at {repo_path}. "
            "Call telos_init first."
        )
    return GraphStore(db_path)


@mcp.tool()
def telos_init(repo_path: str = ".", force: bool = False) -> str:
    """Scan a codebase and build the causal dependency graph.

    Call this before using any other telos tool. Scans all supported
    source files (Python, JavaScript, TypeScript, Go, Java, Rust) and
    builds a graph of function calls, imports, data flows, and
    inheritance relationships.

    Args:
        repo_path: Path to the repository root. Defaults to current directory.
        force: If True, rebuild the graph from scratch.

    Returns:
        Summary of scan results including file count, node count, edge count,
        and top dependency hotspots.
    """
    repo_path = os.path.abspath(repo_path)
    db_path = _get_db_path(repo_path)

    store = GraphStore(db_path)
    if force:
        store.clear()

    builder = GraphBuilder(store)
    start = time.time()
    stats = builder.scan_directory(repo_path)
    elapsed = time.time() - start

    store.set_meta("last_scan", datetime.now().isoformat())
    store.set_meta("repo_root", repo_path)

    analyzer = ImpactAnalyzer(store)
    hotspots = analyzer.hotspots(top_n=5)

    result = {
        "status": "success",
        "repo_path": repo_path,
        "files_scanned": stats["files_scanned"],
        "nodes": stats["nodes"],
        "edges": stats["edges"],
        "scan_time_seconds": round(elapsed, 2),
        "top_hotspots": [
            {
                "node": h["node_id"],
                "dependents": h["dependent_count"],
            }
            for h in hotspots
        ],
    }

    store.close()
    return json.dumps(result, indent=2)


@mcp.tool()
def telos_impact(
    target: str,
    repo_path: str = ".",
    max_depth: int | None = None,
    min_risk: float = 0.0,
) -> str:
    """Trace the full causal impact of changing a code node.

    Shows every function, class, and module that would be affected if
    the target is modified, with risk scores based on edge weights.

    Args:
        target: The node to analyze (e.g. "src/auth.py:validate_token").
        repo_path: Path to the repository root. Must have been initialized.
        max_depth: Maximum traversal depth. None for unlimited.
        min_risk: Minimum risk score to include (0.0 to 1.0).

    Returns:
        JSON with affected nodes, risk scores, hottest path, and files affected.
    """
    repo_path = os.path.abspath(repo_path)
    store = _require_store(repo_path)
    analyzer = ImpactAnalyzer(store)

    result = analyzer.analyze(target, max_depth=max_depth, min_risk=min_risk)

    output = {
        "target": result["target"],
        "affected_count": len(result["affected"]),
        "affected": [
            {
                "node": a["node_id"],
                "risk": round(a["risk"], 3),
                "depth": a["depth"],
                "edge_kind": a["edge_kind"],
                "file": a["file_path"],
                "via": a["via"],
            }
            for a in result["affected"]
        ],
        "hottest_path": result["hottest_path"],
        "files_affected": result["files_affected"],
    }

    store.close()
    return json.dumps(output, indent=2)


@mcp.tool()
def telos_counterfactual(
    target: str,
    intervention_at: str,
    repo_path: str = ".",
) -> str:
    """Evaluate a counterfactual: what if we add a fallback at a specific node?

    Compares the blast radius of changing the target with and without
    an intervention (fallback/fix) at the specified node. Uses Pearl's
    do-operator to sever the causal chain at the intervention point.

    Args:
        target: The node being changed (e.g. "src/auth.py:validate_token").
        intervention_at: Where to add the fallback (e.g. "src/api/middleware.py:require_auth").
        repo_path: Path to the repository root.

    Returns:
        JSON comparing without-fix vs with-fix blast radius, reduction count and percentage.
    """
    repo_path = os.path.abspath(repo_path)
    store = _require_store(repo_path)
    analyzer = ImpactAnalyzer(store)
    cf = CounterfactualAnalyzer(store, analyzer)

    result = cf.analyze(target, intervention_at=intervention_at)

    pct = 0.0
    if result["without_count"] > 0:
        pct = result["reduction"] / result["without_count"] * 100

    output = {
        "target": result["target"],
        "intervention_at": result["intervention_at"],
        "without_fix": {
            "affected_count": result["without_count"],
            "files": result["without_fix"]["files_affected"],
        },
        "with_fix": {
            "affected_count": result["with_count"],
            "files": result["with_fix"]["files_affected"],
        },
        "reduction": result["reduction"],
        "reduction_percentage": round(pct, 1),
    }

    store.close()
    return json.dumps(output, indent=2)


@mcp.tool()
def telos_hotspots(
    repo_path: str = ".",
    top_n: int = 10,
) -> str:
    """Show the most depended-on code nodes in the codebase.

    These are the nodes with the most incoming edges — changes to these
    have the widest blast radius and highest risk.

    Args:
        repo_path: Path to the repository root.
        top_n: Number of hotspots to return.

    Returns:
        JSON list of hotspots with node ID, name, file, and dependent count.
    """
    repo_path = os.path.abspath(repo_path)
    store = _require_store(repo_path)
    analyzer = ImpactAnalyzer(store)

    hotspots = analyzer.hotspots(top_n=top_n)

    output = [
        {
            "rank": i + 1,
            "node": h["node_id"],
            "name": h["name"],
            "file": h["file_path"],
            "dependents": h["dependent_count"],
        }
        for i, h in enumerate(hotspots)
    ]

    store.close()
    return json.dumps(output, indent=2)


@mcp.tool()
def telos_info(repo_path: str = ".") -> str:
    """Show graph statistics and metadata.

    Args:
        repo_path: Path to the repository root.

    Returns:
        JSON with node count, edge count, last scan time, and repo root.
    """
    repo_path = os.path.abspath(repo_path)
    store = _require_store(repo_path)
    stats = store.get_stats()

    db_path = _get_db_path(repo_path)
    db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0

    output = {
        "repo_root": store.get_meta("repo_root") or repo_path,
        "last_scan": store.get_meta("last_scan") or "never",
        "node_count": stats["node_count"],
        "edge_count": stats["edge_count"],
        "db_size_bytes": db_size,
    }

    store.close()
    return json.dumps(output, indent=2)


# ---------------------------------------------------------------------------
# Memory tools
# ---------------------------------------------------------------------------

def _get_memory_path(repo_path: str) -> str:
    return os.path.join(repo_path, ".telos", "memory.db")


def _get_memory(repo_path: str) -> ProjectMemory:
    return ProjectMemory(_get_memory_path(repo_path))


@mcp.tool()
def telos_memory_start_session(
    description: str = "",
    repo_path: str = ".",
) -> str:
    """Start a new memory session for tracking decisions and changes.

    Call this at the beginning of a work session. All subsequent
    record_* calls will be grouped into this session.

    Args:
        description: What this session is about.
        repo_path: Repository root path.

    Returns:
        JSON with session_id.
    """
    repo_path = os.path.abspath(repo_path)
    memory = _get_memory(repo_path)
    session_id = memory.start_session(description)
    memory.close()
    return json.dumps({"session_id": session_id, "description": description})


@mcp.tool()
def telos_memory_record_decision(
    summary: str,
    reasoning: str = "",
    file_path: str = "",
    node_id: str = "",
    repo_path: str = ".",
) -> str:
    """Record a decision made during the current session.

    Captures WHY a decision was made, not just WHAT was decided.
    Automatically links to the previous event in the session chain.

    Args:
        summary: What was decided (e.g., "Add retry logic to API client").
        reasoning: Why this decision was made (e.g., "Upstream service has intermittent failures").
        file_path: Related file (if any).
        node_id: Related code graph node (if any).
        repo_path: Repository root path.

    Returns:
        JSON with event_id.
    """
    repo_path = os.path.abspath(repo_path)
    memory = _get_memory(repo_path)
    event_id = memory.record_decision(summary, reasoning, file_path, node_id)
    memory.close()
    return json.dumps({"event_id": event_id, "kind": "decision", "summary": summary})


@mcp.tool()
def telos_memory_record_change(
    summary: str,
    file_path: str,
    node_id: str = "",
    diff: str = "",
    repo_path: str = ".",
) -> str:
    """Record a code change made during the current session.

    Args:
        summary: What changed (e.g., "Added retry with max_retries=2 and exponential backoff").
        file_path: The file that was changed.
        node_id: The specific function/class changed (if applicable).
        diff: The diff or description of changes.
        repo_path: Repository root path.

    Returns:
        JSON with event_id.
    """
    repo_path = os.path.abspath(repo_path)
    memory = _get_memory(repo_path)
    event_id = memory.record_change(summary, file_path, node_id, diff)
    memory.close()
    return json.dumps({"event_id": event_id, "kind": "change", "file_path": file_path})


@mcp.tool()
def telos_memory_record_outcome(
    summary: str,
    success: bool,
    file_path: str = "",
    node_id: str = "",
    repo_path: str = ".",
) -> str:
    """Record the outcome of a change or decision.

    If success is False, the outcome is automatically linked to the most
    recent change event as a causal relationship.

    Args:
        summary: What happened (e.g., "Tests pass" or "Payment endpoint returns 500").
        success: Whether the outcome was positive.
        file_path: Related file (if any).
        node_id: Related code graph node (if any).
        repo_path: Repository root path.

    Returns:
        JSON with event_id.
    """
    repo_path = os.path.abspath(repo_path)
    memory = _get_memory(repo_path)
    event_id = memory.record_outcome(summary, success, file_path, node_id)
    memory.close()
    return json.dumps({"event_id": event_id, "kind": "outcome", "success": success})


@mcp.tool()
def telos_memory_why(
    event_id: str,
    repo_path: str = ".",
) -> str:
    """Trace the causal chain leading to an event — root cause analysis.

    Walks backwards through event links to find what caused this event,
    what caused that, all the way to the root decision.

    Args:
        event_id: The event to trace back from.
        repo_path: Repository root path.

    Returns:
        JSON list of events in causal order (root cause first).
    """
    repo_path = os.path.abspath(repo_path)
    memory = _get_memory(repo_path)
    chain = memory.why(event_id)
    memory.close()
    return json.dumps([
        {"id": e["id"], "kind": e["kind"], "summary": e["summary"], "timestamp": e["timestamp"]}
        for e in chain
    ], indent=2)


@mcp.tool()
def telos_memory_what_happened(
    file_path: str = "",
    node_id: str = "",
    repo_path: str = ".",
) -> str:
    """Recall everything that happened to a file or code node across all sessions.

    Returns all events (decisions, changes, outcomes) related to the
    specified file or node, most recent first.

    Args:
        file_path: File to query (e.g., "src/auth.py").
        node_id: Code graph node to query (e.g., "src/auth.py:validate_token").
        repo_path: Repository root path.

    Returns:
        JSON list of events.
    """
    repo_path = os.path.abspath(repo_path)
    memory = _get_memory(repo_path)
    events = memory.what_happened(file_path=file_path, node_id=node_id)
    memory.close()
    return json.dumps([
        {"id": e["id"], "kind": e["kind"], "summary": e["summary"],
         "timestamp": e["timestamp"], "file_path": e.get("file_path", "")}
        for e in events
    ], indent=2)


@mcp.tool()
def telos_memory_patterns(repo_path: str = ".") -> str:
    """Analyze patterns across all sessions — what changes often, what breaks.

    Returns:
    - Most frequently changed files
    - Failure-prone files (high failure rate after changes)
    - Files frequently changed together
    - Total session and event counts

    Args:
        repo_path: Repository root path.

    Returns:
        JSON summary of cross-session patterns.
    """
    repo_path = os.path.abspath(repo_path)
    memory = _get_memory(repo_path)
    learner = CrossSessionLearner(memory._graph)
    result = learner.patterns()
    memory.close()
    return json.dumps(result, indent=2)


@mcp.tool()
def telos_memory_search(
    query: str,
    repo_path: str = ".",
) -> str:
    """Search across all memory events by keyword.

    Args:
        query: Search term to match against event summaries.
        repo_path: Repository root path.

    Returns:
        JSON list of matching events.
    """
    repo_path = os.path.abspath(repo_path)
    memory = _get_memory(repo_path)
    events = memory.search(query)
    memory.close()
    return json.dumps([
        {"id": e["id"], "kind": e["kind"], "summary": e["summary"],
         "timestamp": e["timestamp"]}
        for e in events
    ], indent=2)


@mcp.tool()
def telos_memory_recent(
    limit: int = 20,
    repo_path: str = ".",
) -> str:
    """Show the most recent memory events across all sessions.

    Args:
        limit: Maximum number of events to return.
        repo_path: Repository root path.

    Returns:
        JSON list of recent events.
    """
    repo_path = os.path.abspath(repo_path)
    memory = _get_memory(repo_path)
    events = memory.recent(limit=limit)
    memory.close()
    return json.dumps([
        {"id": e["id"], "kind": e["kind"], "summary": e["summary"],
         "timestamp": e["timestamp"], "session_id": e.get("session_id", "")}
        for e in events
    ], indent=2)


if __name__ == "__main__":
    mcp.run()
