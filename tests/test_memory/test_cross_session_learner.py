"""Tests for the CrossSessionLearner pattern detection module."""

import os
import tempfile

import pytest

from telos.memory.project_memory import ProjectMemory
from telos.memory.cross_session_learner import CrossSessionLearner


@pytest.fixture
def setup():
    """Create a ProjectMemory + CrossSessionLearner backed by a temp db."""
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "learner_test.db")
    mem = ProjectMemory(db_path)
    learner = CrossSessionLearner(mem._graph)
    yield mem, learner
    mem.close()


# ------------------------------------------------------------------
# Helper to populate realistic event history
# ------------------------------------------------------------------


def _run_session(mem: ProjectMemory, label: str, actions: list[dict]) -> str:
    """Run a mini-session and return its session_id.

    *actions* is a list of dicts like:
        {"kind": "decision", "summary": ..., "file_path": ..., ...}
        {"kind": "change",   "summary": ..., "file_path": ..., "node_id": ...}
        {"kind": "outcome",  "summary": ..., "success": True/False}
    """
    sid = mem.start_session(label)
    for act in actions:
        kind = act["kind"]
        if kind == "decision":
            mem.record_decision(
                summary=act.get("summary", ""),
                reasoning=act.get("reasoning", ""),
                file_path=act.get("file_path", ""),
                node_id=act.get("node_id", ""),
            )
        elif kind == "change":
            mem.record_change(
                summary=act.get("summary", ""),
                file_path=act.get("file_path", ""),
                node_id=act.get("node_id", ""),
                diff=act.get("diff", ""),
            )
        elif kind == "outcome":
            mem.record_outcome(
                summary=act.get("summary", ""),
                success=act["success"],
                file_path=act.get("file_path", ""),
                node_id=act.get("node_id", ""),
            )
    mem.end_session(f"{label} done")
    return sid


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_most_changed_files(setup):
    mem, learner = setup

    _run_session(mem, "s1", [
        {"kind": "change", "summary": "edit a", "file_path": "a.py"},
        {"kind": "change", "summary": "edit a again", "file_path": "a.py"},
        {"kind": "change", "summary": "edit a third time", "file_path": "a.py"},
        {"kind": "change", "summary": "edit b", "file_path": "b.py"},
    ])

    result = learner.most_changed_files()
    assert len(result) == 2
    assert result[0]["file_path"] == "a.py"
    assert result[0]["change_count"] == 3
    assert result[1]["file_path"] == "b.py"
    assert result[1]["change_count"] == 1


def test_most_changed_nodes(setup):
    mem, learner = setup

    _run_session(mem, "s1", [
        {"kind": "change", "summary": "edit node X", "file_path": "a.py", "node_id": "X"},
        {"kind": "change", "summary": "edit node X", "file_path": "a.py", "node_id": "X"},
        {"kind": "change", "summary": "edit node Y", "file_path": "b.py", "node_id": "Y"},
        {"kind": "change", "summary": "no node", "file_path": "c.py"},
    ])

    result = learner.most_changed_nodes()
    assert len(result) == 2
    assert result[0]["node_id"] == "X"
    assert result[0]["change_count"] == 2
    assert result[1]["node_id"] == "Y"
    assert result[1]["change_count"] == 1


def test_failure_prone_files(setup):
    mem, learner = setup

    # Session with a change to parser.py followed by failure.
    _run_session(mem, "s1", [
        {"kind": "change", "summary": "break parser", "file_path": "parser.py"},
        {"kind": "outcome", "summary": "tests fail", "success": False},
    ])
    # Another session — parser.py changed, succeeds this time.
    _run_session(mem, "s2", [
        {"kind": "change", "summary": "fix parser", "file_path": "parser.py"},
        {"kind": "outcome", "summary": "tests pass", "success": True},
    ])
    # A clean file with no failures.
    _run_session(mem, "s3", [
        {"kind": "change", "summary": "edit utils", "file_path": "utils.py"},
        {"kind": "outcome", "summary": "all good", "success": True},
    ])

    result = learner.failure_prone_files()
    assert len(result) == 1
    entry = result[0]
    assert entry["file_path"] == "parser.py"
    assert entry["failure_count"] == 1
    assert entry["change_count"] == 2
    assert entry["failure_rate"] == 0.5


def test_change_pairs(setup):
    mem, learner = setup

    # Two sessions where a.py and b.py are changed together.
    _run_session(mem, "s1", [
        {"kind": "change", "summary": "edit a", "file_path": "a.py"},
        {"kind": "change", "summary": "edit b", "file_path": "b.py"},
    ])
    _run_session(mem, "s2", [
        {"kind": "change", "summary": "edit a", "file_path": "a.py"},
        {"kind": "change", "summary": "edit b", "file_path": "b.py"},
        {"kind": "change", "summary": "edit c", "file_path": "c.py"},
    ])

    result = learner.change_pairs()
    # a.py + b.py should appear with count=2
    ab = [r for r in result if set([r["file_a"], r["file_b"]]) == {"a.py", "b.py"}]
    assert len(ab) == 1
    assert ab[0]["co_change_count"] == 2

    # a.py + c.py and b.py + c.py should appear with count=1
    for pair_set in [{"a.py", "c.py"}, {"b.py", "c.py"}]:
        matches = [r for r in result if set([r["file_a"], r["file_b"]]) == pair_set]
        assert len(matches) == 1
        assert matches[0]["co_change_count"] == 1


def test_decision_history(setup):
    mem, learner = setup

    _run_session(mem, "s1", [
        {"kind": "decision", "summary": "Use caching", "reasoning": "Perf matters", "file_path": "cache.py"},
        {"kind": "change", "summary": "Add cache", "file_path": "cache.py"},
        {"kind": "outcome", "summary": "Latency improved", "success": True},
    ])

    result = learner.decision_history(file_path="cache.py")
    assert len(result) == 1
    entry = result[0]
    assert entry["decision"] == "Use caching"
    assert entry["reasoning"] == "Perf matters"
    assert entry["outcome"] == "Latency improved"
    assert entry["success"] is True
    assert entry["timestamp"]  # non-empty


def test_session_summary(setup):
    mem, learner = setup

    sid = _run_session(mem, "full session", [
        {"kind": "decision", "summary": "Plan refactor"},
        {"kind": "change", "summary": "Refactor a", "file_path": "a.py"},
        {"kind": "change", "summary": "Refactor b", "file_path": "b.py"},
        {"kind": "outcome", "summary": "Tests pass", "success": True},
        {"kind": "outcome", "summary": "Lint fails", "success": False},
    ])

    result = learner.session_summary(sid)
    assert result["session_id"] == sid
    assert result["decisions"] == 1
    assert result["changes"] == 2
    assert result["outcomes"] == 2
    assert result["successes"] == 1
    assert result["failures"] == 1
    assert sorted(result["files_touched"]) == ["a.py", "b.py"]


def test_patterns_returns_summary(setup):
    mem, learner = setup

    _run_session(mem, "s1", [
        {"kind": "change", "summary": "edit a", "file_path": "a.py"},
        {"kind": "outcome", "summary": "ok", "success": True},
    ])

    result = learner.patterns()
    assert "most_changed" in result
    assert "failure_prone" in result
    assert "co_changes" in result
    assert "total_sessions" in result
    assert "total_events" in result
    assert result["total_sessions"] >= 1
    assert result["total_events"] >= 1


def test_empty_history(setup):
    _, learner = setup

    assert learner.most_changed_files() == []
    assert learner.most_changed_nodes() == []
    assert learner.failure_prone_files() == []
    assert learner.change_pairs() == []
    assert learner.decision_history() == []

    summary = learner.session_summary("nonexistent")
    assert summary["decisions"] == 0
    assert summary["changes"] == 0
    assert summary["files_touched"] == []

    p = learner.patterns()
    assert p["total_events"] == 0
    assert p["total_sessions"] == 0
