"""Tests for the FixEvaluator."""

from __future__ import annotations

import os
import tempfile

import pytest

from telos.code_parser.store import GraphStore
from telos.history.fix_evaluator import FixCandidate, FixEvaluator, FixScore
from telos.impact.analyzer import ImpactAnalyzer
from telos.impact.counterfactual import CounterfactualAnalyzer


@pytest.fixture()
def evaluator():
    """Build a toy graph and wrap it in a FixEvaluator.

    Graph::

        a ──CALLS──▸ b ──CALLS──▸ c ──DATA_FLOW──▸ d
                                      ▴
                      e ──CALLS───────┘
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
    yield FixEvaluator(counterfactual_analyzer=cf)

    store.close()
    os.unlink(db_path)


def test_fix_candidate_and_fix_score_dataclasses() -> None:
    c = FixCandidate(
        description="Add fallback at b",
        intervention_at="b",
        change_file="b.py",
    )
    assert c.description == "Add fallback at b"
    assert c.intervention_at == "b"
    assert c.change_file == "b.py"

    s = FixScore(candidate=c)
    assert s.candidate is c
    assert s.blast_radius_reduction == 0
    assert s.historical_risk == 0.0
    assert s.author_expertise_risk == 0.0
    assert s.reasoning == []


def test_evaluate_returns_list_of_scores(evaluator: FixEvaluator) -> None:
    candidates = [
        FixCandidate("fallback at b", "b", "b.py"),
        FixCandidate("fallback at c", "c", "c.py"),
        FixCandidate("fallback at d", "d", "d.py"),
    ]
    scores = evaluator.evaluate("a", candidates)
    assert isinstance(scores, list)
    assert len(scores) == 3
    assert all(isinstance(s, FixScore) for s in scores)
    assert all(isinstance(s.reasoning, list) for s in scores)


def test_higher_reduction_scores_higher(evaluator: FixEvaluator) -> None:
    candidates = [
        FixCandidate("fallback at b", "b", "b.py"),  # saves most
        FixCandidate("fallback at d", "d", "d.py"),  # saves none (leaf)
    ]
    scores = evaluator.evaluate("a", candidates)
    # Scores are sorted descending. The b-fallback should come first.
    assert scores[0].candidate.intervention_at == "b"
    assert scores[0].blast_radius_reduction > scores[-1].blast_radius_reduction
    assert scores[0].total_score >= scores[-1].total_score


def test_without_git_learner_still_works(evaluator: FixEvaluator) -> None:
    # The fixture builds an evaluator with no git_learner / developer_model.
    candidates = [FixCandidate("fallback at b", "b", "b.py")]
    scores = evaluator.evaluate("a", candidates)
    assert len(scores) == 1
    # No historical data -> epistemic should be 0.5
    assert scores[0].epistemic_score == 0.5
    assert scores[0].historical_risk == 0.0
    assert scores[0].author_expertise_risk == 0.0


def test_rank_and_explain_returns_dict_with_recommendation(
    evaluator: FixEvaluator,
) -> None:
    candidates = [
        FixCandidate("fallback at b", "b", "b.py"),
        FixCandidate("fallback at c", "c", "c.py"),
    ]
    result = evaluator.rank_and_explain("a", candidates)
    assert isinstance(result, dict)
    assert result["target"] == "a"
    assert "ranked_candidates" in result
    assert "recommendation" in result
    assert isinstance(result["ranked_candidates"], list)
    assert len(result["ranked_candidates"]) == 2
    first = result["ranked_candidates"][0]
    assert first["rank"] == 1
    assert "description" in first
    assert "total_score" in first
    assert "reasoning" in first
    assert isinstance(first["reasoning"], list)


def test_rank_and_explain_empty_candidates(evaluator: FixEvaluator) -> None:
    result = evaluator.rank_and_explain("a", [])
    assert result["ranked_candidates"] == []
    assert "No candidates" in result["recommendation"]
