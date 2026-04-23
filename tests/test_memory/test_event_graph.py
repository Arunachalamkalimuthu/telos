"""Tests for the EventGraph causal memory module."""

import os
import tempfile

import pytest

from telos.memory.event_graph import EventGraph


@pytest.fixture
def graph():
    """Yield an EventGraph backed by a temp SQLite db, then clean up."""
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "memory.db")
    g = EventGraph(db_path)
    yield g
    g.close()


def test_add_and_get_event(graph: EventGraph):
    eid = graph.add_event(
        kind="decision",
        summary="Use SQLite for the memory layer",
        session_id="s1",
        data={"reason": "lightweight"},
        file_path="src/memory.py",
        node_id="node_42",
    )
    assert isinstance(eid, str)
    assert len(eid) == 12

    event = graph.get_event(eid)
    assert event is not None
    assert event["kind"] == "decision"
    assert event["summary"] == "Use SQLite for the memory layer"
    assert event["session_id"] == "s1"
    assert event["data"] == {"reason": "lightweight"}
    assert event["file_path"] == "src/memory.py"
    assert event["node_id"] == "node_42"
    assert event["timestamp"]  # non-empty ISO string


def test_link_events(graph: EventGraph):
    e1 = graph.add_event(kind="decision", summary="Choose approach A")
    e2 = graph.add_event(kind="change", summary="Implement approach A")
    graph.link_events(e1, e2, kind="caused", weight=0.9)

    # Verify the link by checking causal chain of e2
    chain = graph.get_causal_chain(e2)
    assert len(chain) == 1
    assert chain[0]["id"] == e1


def test_get_session_events(graph: EventGraph):
    e1 = graph.add_event(kind="session_start", summary="Begin session", session_id="sess_a")
    e2 = graph.add_event(kind="decision", summary="Pick strategy", session_id="sess_a")
    e3 = graph.add_event(kind="session_end", summary="End session", session_id="sess_a")
    # Different session — should not appear
    graph.add_event(kind="decision", summary="Other", session_id="sess_b")

    events = graph.get_session_events("sess_a")
    assert len(events) == 3
    ids = [e["id"] for e in events]
    assert ids == [e1, e2, e3]


def test_get_events_for_node(graph: EventGraph):
    graph.add_event(kind="change", summary="Refactor parser", node_id="parser.parse")
    graph.add_event(kind="outcome", summary="Parser tests pass", node_id="parser.parse")
    graph.add_event(kind="change", summary="Fix CLI", node_id="cli.run")

    events = graph.get_events_for_node("parser.parse")
    assert len(events) == 2
    assert all(e["node_id"] == "parser.parse" for e in events)


def test_get_events_for_file(graph: EventGraph):
    graph.add_event(kind="change", summary="Edit config", file_path="config.py")
    graph.add_event(kind="outcome", summary="Config ok", file_path="config.py")
    graph.add_event(kind="change", summary="Edit main", file_path="main.py")

    events = graph.get_events_for_file("config.py")
    assert len(events) == 2
    assert all(e["file_path"] == "config.py" for e in events)


def test_get_causal_chain(graph: EventGraph):
    e1 = graph.add_event(kind="decision", summary="Root cause")
    e2 = graph.add_event(kind="change", summary="Intermediate step")
    e3 = graph.add_event(kind="outcome", summary="Final outcome")
    graph.link_events(e1, e2, kind="caused")
    graph.link_events(e2, e3, kind="led_to")

    chain = graph.get_causal_chain(e3)
    assert len(chain) == 2
    assert chain[0]["id"] == e1
    assert chain[1]["id"] == e2


def test_get_consequences(graph: EventGraph):
    root = graph.add_event(kind="decision", summary="Root decision")
    c1 = graph.add_event(kind="change", summary="Consequence 1")
    c2 = graph.add_event(kind="change", summary="Consequence 2")
    graph.link_events(root, c1, kind="caused")
    graph.link_events(root, c2, kind="caused")

    results = graph.get_consequences(root)
    result_ids = {e["id"] for e in results}
    assert c1 in result_ids
    assert c2 in result_ids
    assert len(results) == 2


def test_get_recent_events(graph: EventGraph):
    for i in range(5):
        graph.add_event(kind="decision", summary=f"Event {i}")

    recent = graph.get_recent_events(limit=3)
    assert len(recent) == 3
    # Most recent first
    assert recent[0]["summary"] == "Event 4"
    assert recent[1]["summary"] == "Event 3"
    assert recent[2]["summary"] == "Event 2"


def test_search_events(graph: EventGraph):
    graph.add_event(kind="decision", summary="Refactor the parser module")
    graph.add_event(kind="change", summary="Fix CLI argument parsing")
    graph.add_event(kind="outcome", summary="Deploy to production")

    results = graph.search_events("parser")
    assert len(results) == 1
    assert results[0]["summary"] == "Refactor the parser module"

    results = graph.search_events("pars")
    assert len(results) == 2


def test_get_stats(graph: EventGraph):
    e1 = graph.add_event(kind="decision", summary="A", session_id="s1")
    e2 = graph.add_event(kind="change", summary="B", session_id="s1")
    e3 = graph.add_event(kind="outcome", summary="C", session_id="s2")
    graph.link_events(e1, e2)
    graph.link_events(e2, e3)

    stats = graph.get_stats()
    assert stats["event_count"] == 3
    assert stats["link_count"] == 2
    assert stats["session_count"] == 2
