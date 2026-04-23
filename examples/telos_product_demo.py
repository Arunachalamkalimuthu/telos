"""End-to-end demo: all 4 telos phases working together on a real codebase.

Demonstrates:
  Phase 1 — Code parser + impact analysis + counterfactual reasoning
  Phase 2 — (not shown — requires MCP client; see README for setup)
  Phase 3 — Memory layer: sessions, decisions, causal chains
  Phase 4 — Git history: co-changes, bug-prone files, developer expertise

Run with: PYTHONPATH=src python3 -m examples.telos_product_demo

This demo runs against the telos repo itself. It requires:
  - telos installed: `make install`
  - telos graph initialized: `telos init`
"""

from __future__ import annotations

import os
import tempfile

from telos.code_parser.graph_builder import GraphBuilder
from telos.code_parser.store import GraphStore
from telos.impact.analyzer import ImpactAnalyzer
from telos.impact.counterfactual import CounterfactualAnalyzer
from telos.memory.project_memory import ProjectMemory
from telos.history.git_learner import GitLearner
from telos.history.developer_model import DeveloperModel


REPO_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def run() -> None:
    print("=== telos end-to-end product demo ===")
    print(f"Repo: {REPO_PATH}")

    # ------------------------------------------------------------------
    # Phase 1: build code graph, trace impact, counterfactual analysis
    # ------------------------------------------------------------------
    section("Phase 1: Code Graph + Impact Analysis")

    # Use a temp dir so the demo doesn't touch the real .telos/ db.
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "graph.db")
        store = GraphStore(db_path)
        builder = GraphBuilder(store)

        print(f"Scanning {REPO_PATH} ...")
        stats = builder.scan_directory(REPO_PATH)
        print(f"  Files scanned: {stats['files_scanned']}")
        print(f"  Nodes:         {stats['nodes']}")
        print(f"  Edges:         {stats['edges']}")

        analyzer = ImpactAnalyzer(store)

        # Find a target with real outgoing impact: pick a node that has CALLS
        # edges leaving it (an orchestrator, not a leaf helper).
        candidate_target = ""
        for node in store.get_all_nodes():
            if "src/telos/" not in node["file_path"]:
                continue
            outgoing = store.get_edges_from(node["id"])
            if sum(1 for e in outgoing if e["kind"] == "CALLS") >= 3:
                candidate_target = node["id"]
                break
        target = candidate_target or "src/telos/cli.py:init"
        impact = analyzer.analyze(target, max_depth=3)

        print(f"\nImpact of changing {target}:")
        print(f"  Affected nodes: {len(impact['affected'])}")
        print(f"  Files affected: {len(impact['files_affected'])}")
        for entry in impact["affected"][:5]:
            print(
                f"    {entry['edge_kind']:10} → {entry['node_id']:50}"
                f" risk={entry['risk']:.2f}"
            )

        # Hotspots
        hotspots = analyzer.hotspots(top_n=3)
        print(f"\nTop 3 hotspots (most depended-on):")
        for i, h in enumerate(hotspots, 1):
            print(f"  {i}. {h['node_id']} → {h['dependent_count']} dependents")

        # Counterfactual — "what if we intervene at the first affected node?"
        if impact["affected"]:
            intervention = impact["affected"][0]["node_id"]
            cf = CounterfactualAnalyzer(store, analyzer)
            cf_result = cf.analyze(target, intervention_at=intervention)

            print(f"\nCounterfactual (intervention at {intervention}):")
            print(f"  Without fix: {cf_result['without_count']} nodes affected")
            print(f"  With fix:    {cf_result['with_count']} nodes affected")
            print(f"  Reduction:   {cf_result['reduction']} nodes")

        store.close()

    # ------------------------------------------------------------------
    # Phase 3: Memory — sessions, decisions, causal chains
    # ------------------------------------------------------------------
    section("Phase 3: Memory Layer")

    with tempfile.TemporaryDirectory() as tmp:
        memory_path = os.path.join(tmp, "memory.db")
        memory = ProjectMemory(memory_path)

        session_id = memory.start_session(
            "Demo session — investigating auth module"
        )
        print(f"Started session: {session_id}")

        decision = memory.record_decision(
            summary="Add retry logic to API client",
            reasoning="Upstream service has intermittent timeouts",
            file_path="src/api/client.py",
        )
        print(f"  Recorded decision: {decision}")

        change = memory.record_change(
            summary="Added retry with max_retries=5 and exponential backoff",
            file_path="src/api/client.py",
            node_id="src/api/client.py:call",
        )
        print(f"  Recorded change: {change}")

        outcome_1 = memory.record_outcome(
            summary="Connection pool exhausted in 12 seconds under load",
            success=False,
            file_path="src/api/client.py",
        )
        print(f"  Recorded outcome (failure): {outcome_1}")

        decision_2 = memory.record_decision(
            summary="Cap retries at 2, not 5",
            reasoning="Pool exhaustion — unbounded retries overwhelm the pool",
            file_path="src/api/client.py",
        )
        print(f"  Recorded follow-up decision: {decision_2}")

        # Root cause trace
        print("\nTracing causal chain backward from follow-up decision:")
        chain = memory.why(decision_2)
        for event in chain:
            print(f"  [{event['kind']}] {event['summary']}")

        # Query by file
        print("\nEverything that happened to src/api/client.py:")
        history = memory.what_happened(file_path="src/api/client.py")
        for event in history:
            print(f"  [{event['kind']}] {event['summary']}")

        memory.close()

    # ------------------------------------------------------------------
    # Phase 4: Git history learning (real data from telos repo)
    # ------------------------------------------------------------------
    section("Phase 4: Git History + Developer Expertise")

    try:
        learner = GitLearner(REPO_PATH)
        commits = learner.get_commits(max_count=100)

        stats = learner.get_stats()
        print(f"Repo stats:")
        print(f"  Total commits analyzed: {stats['total_commits']}")
        print(f"  Unique authors:         {stats['unique_authors']}")
        print(f"  Date range:             {stats['date_range']}")

        # Co-change patterns
        print("\nMost frequently co-changed file pairs:")
        co_changes = sorted(
            learner.co_change_matrix(commits).items(),
            key=lambda kv: -kv[1],
        )[:3]
        for (a, b), count in co_changes:
            print(f"  {count:3d}x  {a}  ↔  {b}")

        # Bug-prone files
        print("\nMost bug-prone files (by commit-message keywords):")
        bug_prone = learner.bug_prone_files(commits, top_n=3)
        for entry in bug_prone:
            print(
                f"  {entry['file_path']:60} "
                f"{entry['bug_fix_count']} bug fixes "
                f"({entry['bug_rate']:.1%} of total)"
            )

        # Developer model
        print("\nDeveloper expertise:")
        dm = DeveloperModel(learner)
        profiles = dm.build_profiles(max_commits=100)
        for name, profile in list(profiles.items())[:3]:
            areas = ", ".join(profile.expertise_areas[:3]) or "(none)"
            print(
                f"  {name}: {profile.commit_count} commits, "
                f"expert in: {areas}"
            )

    except ValueError as e:
        print(f"Git history analysis skipped: {e}")

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("  Demo complete.")
    print("=" * 72)
    print()
    print("To use telos with an LLM (Phase 2 — MCP server):")
    print("  See README.md for Claude Code configuration.")


if __name__ == "__main__":
    run()
