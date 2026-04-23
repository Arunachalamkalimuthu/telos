"""Tests for the ProjectMemory session management and causal recording layer."""

import os
import tempfile

import pytest

from telos.memory.project_memory import ProjectMemory


@pytest.fixture
def mem():
    """Yield a ProjectMemory backed by a temp SQLite db, then clean up."""
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "project_memory.db")
    pm = ProjectMemory(db_path)
    yield pm
    pm.close()


# ------------------------------------------------------------------
# Session management
# ------------------------------------------------------------------


def test_start_and_end_session(mem: ProjectMemory):
    sid = mem.start_session("Initial exploration")
    assert sid.startswith("sess_")
    assert mem.current_session == sid

    mem.end_session("Wrapped up exploration")
    assert mem.current_session is None

    # Session history should contain start and end events.
    history = mem.get_session_history(sid)
    assert len(history) == 2
    assert history[0]["kind"] == "session_start"
    assert history[1]["kind"] == "session_end"


# ------------------------------------------------------------------
# Recording
# ------------------------------------------------------------------


def test_record_decision(mem: ProjectMemory):
    mem.start_session()
    eid = mem.record_decision(
        summary="Use event graph for memory",
        reasoning="It tracks causality",
    )
    event = mem._graph.get_event(eid)
    assert event["kind"] == "decision"
    assert event["data"]["reasoning"] == "It tracks causality"


def test_record_change_with_file(mem: ProjectMemory):
    mem.start_session()
    eid = mem.record_change(
        summary="Add caching layer",
        file_path="src/cache.py",
        diff="+ class Cache: ...",
    )
    event = mem._graph.get_event(eid)
    assert event["kind"] == "change"
    assert event["file_path"] == "src/cache.py"
    assert event["data"]["diff"] == "+ class Cache: ..."


def test_record_outcome_success(mem: ProjectMemory):
    mem.start_session()
    mem.record_change(summary="Fix parser", file_path="parser.py")
    eid = mem.record_outcome(summary="All tests pass", success=True)
    event = mem._graph.get_event(eid)
    assert event["kind"] == "outcome"
    assert event["data"]["success"] is True


def test_record_outcome_failure_links_to_change(mem: ProjectMemory):
    mem.start_session()
    change_id = mem.record_change(summary="Break parser", file_path="parser.py")
    outcome_id = mem.record_outcome(summary="Tests fail", success=False)

    # The causal chain of the outcome should include the change via "caused".
    chain = mem.why(outcome_id)
    chain_ids = [e["id"] for e in chain]
    assert change_id in chain_ids


# ------------------------------------------------------------------
# Session chain (auto-linking)
# ------------------------------------------------------------------


def test_session_chain_auto_links(mem: ProjectMemory):
    sid = mem.start_session("Chain test")
    d = mem.record_decision(summary="Plan refactor")
    c = mem.record_change(summary="Refactor code", file_path="main.py")
    o = mem.record_outcome(summary="Refactor OK", success=True)
    mem.end_session("Done")

    # Walking back from outcome should traverse change -> decision -> start.
    chain = mem.why(o)
    chain_ids = [e["id"] for e in chain]
    assert d in chain_ids
    assert c in chain_ids


# ------------------------------------------------------------------
# Querying
# ------------------------------------------------------------------


def test_why_returns_causal_chain(mem: ProjectMemory):
    mem.start_session()
    d = mem.record_decision(summary="Pick approach A")
    c = mem.record_change(summary="Implement A", file_path="a.py")
    o = mem.record_outcome(summary="A works", success=True)

    chain = mem.why(o)
    # Chain should go from root cause to the event before o.
    assert len(chain) >= 2
    assert chain[-1]["id"] == c
    assert chain[-2]["id"] == d


def test_what_happened_for_file(mem: ProjectMemory):
    mem.start_session()
    mem.record_change(summary="Create config", file_path="config.py")
    mem.record_change(summary="Update config", file_path="config.py")
    mem.record_change(summary="Unrelated file", file_path="other.py")

    events = mem.what_happened(file_path="config.py")
    assert len(events) == 2
    # Most recent first.
    assert events[0]["summary"] == "Update config"
    assert events[1]["summary"] == "Create config"


def test_last_time_returns_most_recent(mem: ProjectMemory):
    mem.start_session()
    mem.record_change(summary="First edit", file_path="app.py")
    mem.record_change(summary="Second edit", file_path="app.py")

    last = mem.last_time(file_path="app.py")
    assert last is not None
    assert last["summary"] == "Second edit"


def test_search_finds_events(mem: ProjectMemory):
    mem.start_session()
    mem.record_decision(summary="Adopt microservices architecture")
    mem.record_change(summary="Split monolith", file_path="main.py")
    mem.record_outcome(summary="Microservices deployed", success=True)

    results = mem.search("microservice")
    assert len(results) == 2  # decision + outcome
    summaries = {r["summary"] for r in results}
    assert "Adopt microservices architecture" in summaries
    assert "Microservices deployed" in summaries
