"""Tests for the CounterfactualAnalyzer."""

import tempfile, os, pytest

from telos.code_parser.store import GraphStore
from telos.impact.analyzer import ImpactAnalyzer
from telos.impact.counterfactual import CounterfactualAnalyzer


@pytest.fixture()
def counterfactual():
    """Same graph as the analyzer tests; return a CounterfactualAnalyzer.

    Graph::

        a ──CALLS(1.0)──▸ b ──CALLS(1.0)──▸ c ──DATA_FLOW(0.8)──▸ d
                                              ▴
                          e ──CALLS(1.0)──────┘
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
    cf = CounterfactualAnalyzer(store, analyzer)
    yield cf

    store.close()
    os.unlink(db_path)


# ── Intervention reduces blast radius ─────────────────────────────

def test_intervention_reduces_blast_radius(counterfactual):
    result = counterfactual.analyze("a", intervention_at="b")
    assert result["with_count"] < result["without_count"]


# ── Reduction field is positive ────────────────────────────────────

def test_intervention_report_has_reduction(counterfactual):
    result = counterfactual.analyze("a", intervention_at="b")
    assert result["reduction"] > 0


# ── Intervention at a leaf changes nothing ─────────────────────────

def test_intervention_at_leaf_no_change(counterfactual):
    result = counterfactual.analyze("a", intervention_at="d")
    assert result["with_count"] == result["without_count"]
    assert result["reduction"] == 0
