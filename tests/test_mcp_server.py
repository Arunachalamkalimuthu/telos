"""Tests for telos MCP server tools."""

import json
import os
import tempfile
import unittest

from telos.mcp_server import (
    telos_init,
    telos_impact,
    telos_counterfactual,
    telos_hotspots,
    telos_info,
    telos_memory_start_session,
    telos_memory_record_decision,
    telos_memory_record_change,
    telos_memory_record_outcome,
    telos_memory_why,
    telos_memory_what_happened,
    telos_memory_patterns,
    telos_memory_search,
    telos_memory_recent,
)


class TestMCPTools(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create test Python files with dependencies.
        with open(os.path.join(self.tmpdir, "auth.py"), "w") as f:
            f.write(
                "def validate_token(token):\n"
                "    return token is not None\n"
            )
        with open(os.path.join(self.tmpdir, "api.py"), "w") as f:
            f.write(
                "from auth import validate_token\n\n"
                "def handler(request):\n"
                "    validate_token(request.token)\n"
                "    return process(request)\n\n"
                "def process(request):\n"
                "    return request.data\n"
            )
        with open(os.path.join(self.tmpdir, "payment.py"), "w") as f:
            f.write(
                "def charge(amount):\n"
                "    return amount > 0\n"
            )
        # Initialize the graph.
        telos_init(repo_path=self.tmpdir)

    def test_telos_init_returns_json(self):
        result = json.loads(telos_init(repo_path=self.tmpdir, force=True))
        self.assertEqual(result["status"], "success")
        self.assertGreater(result["files_scanned"], 0)
        self.assertGreater(result["nodes"], 0)

    def test_telos_info_returns_stats(self):
        result = json.loads(telos_info(repo_path=self.tmpdir))
        self.assertGreater(result["node_count"], 0)
        self.assertGreater(result["edge_count"], 0)
        self.assertIn("last_scan", result)

    def test_telos_hotspots_returns_list(self):
        result = json.loads(telos_hotspots(repo_path=self.tmpdir))
        self.assertIsInstance(result, list)

    def test_telos_impact_returns_affected(self):
        result = json.loads(telos_impact(
            target="auth.py:validate_token",
            repo_path=self.tmpdir,
        ))
        self.assertEqual(result["target"], "auth.py:validate_token")
        self.assertIsInstance(result["affected"], list)

    def test_telos_impact_nonexistent_node(self):
        result = json.loads(telos_impact(
            target="nonexistent:function",
            repo_path=self.tmpdir,
        ))
        self.assertEqual(result["affected_count"], 0)

    def test_telos_counterfactual_returns_comparison(self):
        # First check if there's an impact path
        impact = json.loads(telos_impact(
            target="auth.py:validate_token",
            repo_path=self.tmpdir,
        ))
        if impact["affected_count"] > 0:
            first_affected = impact["affected"][0]["node"]
            result = json.loads(telos_counterfactual(
                target="auth.py:validate_token",
                intervention_at=first_affected,
                repo_path=self.tmpdir,
            ))
            self.assertIn("without_fix", result)
            self.assertIn("with_fix", result)
            self.assertIn("reduction", result)

    def test_telos_init_not_initialized_raises(self):
        empty_dir = tempfile.mkdtemp()
        with self.assertRaises(ValueError):
            telos_impact(target="x", repo_path=empty_dir)

    def test_telos_impact_with_depth_limit(self):
        result = json.loads(telos_impact(
            target="auth.py:validate_token",
            repo_path=self.tmpdir,
            max_depth=1,
        ))
        for affected in result["affected"]:
            self.assertLessEqual(affected["depth"], 1)

    def test_telos_hotspots_with_top_n(self):
        result = json.loads(telos_hotspots(repo_path=self.tmpdir, top_n=2))
        self.assertLessEqual(len(result), 2)


class TestMCPMemoryTools(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_start_session(self):
        result = json.loads(telos_memory_start_session(
            description="Fix auth bug",
            repo_path=self.tmpdir,
        ))
        self.assertIn("session_id", result)
        self.assertEqual(result["description"], "Fix auth bug")

    def test_record_decision(self):
        telos_memory_start_session(description="test", repo_path=self.tmpdir)
        result = json.loads(telos_memory_record_decision(
            summary="Add retry logic",
            reasoning="Upstream is flaky",
            file_path="api.py",
            repo_path=self.tmpdir,
        ))
        self.assertEqual(result["kind"], "decision")

    def test_record_change(self):
        telos_memory_start_session(description="test", repo_path=self.tmpdir)
        result = json.loads(telos_memory_record_change(
            summary="Added retry with backoff",
            file_path="api.py",
            repo_path=self.tmpdir,
        ))
        self.assertEqual(result["kind"], "change")
        self.assertEqual(result["file_path"], "api.py")

    def test_record_outcome(self):
        telos_memory_start_session(description="test", repo_path=self.tmpdir)
        result = json.loads(telos_memory_record_outcome(
            summary="Tests pass",
            success=True,
            repo_path=self.tmpdir,
        ))
        self.assertEqual(result["kind"], "outcome")
        self.assertTrue(result["success"])

    def test_what_happened_for_file(self):
        telos_memory_start_session(description="test", repo_path=self.tmpdir)
        telos_memory_record_change(summary="Changed auth", file_path="auth.py", repo_path=self.tmpdir)
        result = json.loads(telos_memory_what_happened(file_path="auth.py", repo_path=self.tmpdir))
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0]["file_path"], "auth.py")

    def test_search(self):
        telos_memory_start_session(description="test", repo_path=self.tmpdir)
        telos_memory_record_decision(summary="Add retry logic to payment", repo_path=self.tmpdir)
        result = json.loads(telos_memory_search(query="retry", repo_path=self.tmpdir))
        self.assertGreater(len(result), 0)

    def test_recent(self):
        telos_memory_start_session(description="test", repo_path=self.tmpdir)
        telos_memory_record_decision(summary="decision 1", repo_path=self.tmpdir)
        telos_memory_record_change(summary="change 1", file_path="x.py", repo_path=self.tmpdir)
        result = json.loads(telos_memory_recent(limit=5, repo_path=self.tmpdir))
        self.assertGreater(len(result), 0)

    def test_patterns(self):
        telos_memory_start_session(description="test", repo_path=self.tmpdir)
        telos_memory_record_change(summary="fix", file_path="a.py", repo_path=self.tmpdir)
        result = json.loads(telos_memory_patterns(repo_path=self.tmpdir))
        self.assertIn("most_changed", result)
        self.assertIn("total_events", result)


if __name__ == "__main__":
    unittest.main()
