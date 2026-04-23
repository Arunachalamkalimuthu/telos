"""Tests for the ImpactAnalyzer."""

import tempfile, os, pytest

from telos.code_parser.store import GraphStore
from telos.impact.analyzer import ImpactAnalyzer


@pytest.fixture()
def store_and_analyzer():
    """Build a small test graph and return (store, analyzer).

    Graph::

        a ──CALLS(1.0)──▸ b ──CALLS(1.0)──▸ c ──DATA_FLOW(0.8)──▸ d
                                              ▴
                          e ──CALLS(1.0)──────┘

    All edges are CALLS with weight 1.0 except c→d which is DATA_FLOW 0.8.
    """
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = GraphStore(db_path)

    store.add_node("a", "a.py", "a", "function", "python", 1, 10)
    store.add_node("b", "b.py", "b", "function", "python", 1, 10)
    store.add_node("c", "c.py", "c", "function", "python", 1, 10)
    store.add_node("d", "d.py", "d", "function", "python", 1, 10)
    store.add_node("e", "e.py", "e", "function", "python", 1, 10)

    store.add_edge("a", "b", "CALLS", weight=1.0, file_path="a.py", line=5)
    store.add_edge("b", "c", "CALLS", weight=1.0, file_path="b.py", line=3)
    store.add_edge("c", "d", "DATA_FLOW", weight=0.8, file_path="c.py", line=7)
    store.add_edge("e", "c", "CALLS", weight=1.0, file_path="e.py", line=2)

    analyzer = ImpactAnalyzer(store)
    yield store, analyzer

    store.close()
    os.unlink(db_path)


# ── Direct impact ───────────────────────────────────────────────────

def test_direct_impact(store_and_analyzer):
    _, analyzer = store_and_analyzer
    result = analyzer.analyze("a")
    ids = [e["node_id"] for e in result["affected"]]
    assert "b" in ids


# ── Transitive impact ──────────────────────────────────────────────

def test_transitive_impact(store_and_analyzer):
    _, analyzer = store_and_analyzer
    result = analyzer.analyze("a")
    ids = {e["node_id"] for e in result["affected"]}
    assert ids == {"b", "c", "d"}


# ── Risk scores decrease along the chain ───────────────────────────

def test_risk_scores_decrease(store_and_analyzer):
    _, analyzer = store_and_analyzer
    result = analyzer.analyze("a")
    by_id = {e["node_id"]: e for e in result["affected"]}
    assert by_id["b"]["risk"] >= by_id["d"]["risk"]


# ── Depth limit ────────────────────────────────────────────────────

def test_depth_limit(store_and_analyzer):
    _, analyzer = store_and_analyzer
    result = analyzer.analyze("a", max_depth=1)
    ids = {e["node_id"] for e in result["affected"]}
    assert "b" in ids
    assert "d" not in ids


# ── min_risk filter ────────────────────────────────────────────────

def test_min_risk_filter(store_and_analyzer):
    _, analyzer = store_and_analyzer
    result = analyzer.analyze("a", min_risk=0.9)
    ids = {e["node_id"] for e in result["affected"]}
    assert "d" not in ids  # risk = 1.0 * 1.0 * 0.8 = 0.8 < 0.9


# ── Hottest path ───────────────────────────────────────────────────

def test_hottest_path(store_and_analyzer):
    _, analyzer = store_and_analyzer
    result = analyzer.analyze("a")
    path = result["hottest_path"]
    assert path[0] == "a"
    assert path[-1] == "d"


# ── Affected files ─────────────────────────────────────────────────

def test_affected_files(store_and_analyzer):
    _, analyzer = store_and_analyzer
    result = analyzer.analyze("a")
    assert "b.py" in result["files_affected"]


# ── Non-existent node ─────────────────────────────────────────────

def test_nonexistent_node(store_and_analyzer):
    _, analyzer = store_and_analyzer
    result = analyzer.analyze("zzz_does_not_exist")
    assert result["affected"] == []
