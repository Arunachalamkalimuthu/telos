"""Tests for the Rich terminal reporter."""

from telos.impact.reporter import (
    format_counterfactual,
    format_hotspots,
    format_impact,
    format_info,
)


def test_format_impact_produces_string():
    result = {
        "target": "app.py:main",
        "affected": [
            {
                "node_id": "db.py:connect",
                "risk": 0.95,
                "depth": 1,
                "edge_kind": "CALLS",
                "file_path": "db.py",
                "via": "app.py:main",
            },
        ],
        "hottest_path": ["app.py:main", "db.py:connect"],
        "files_affected": ["db.py"],
    }
    output = format_impact(result)
    assert isinstance(output, str)
    assert "app.py:main" in output
    assert "db.py:connect" in output


def test_format_hotspots_produces_string():
    hotspots = [
        {
            "node_id": "utils.py:helper",
            "name": "helper",
            "file_path": "utils.py",
            "dependent_count": 7,
        },
    ]
    output = format_hotspots(hotspots)
    assert isinstance(output, str)
    assert "helper" in output
    assert "7" in output


def test_format_counterfactual_produces_string():
    result = {
        "target": "app.py:main",
        "intervention_at": "cache.py:get",
        "without_fix": {"affected": []},
        "with_fix": {"affected": []},
        "reduction": 3,
        "without_count": 10,
        "with_count": 7,
    }
    output = format_counterfactual(result)
    assert isinstance(output, str)
    assert "10" in output
    assert "7" in output


def test_format_info_produces_string():
    info = {"node_count": 42, "edge_count": 100}
    output = format_info(info)
    assert isinstance(output, str)
    assert "42" in output
