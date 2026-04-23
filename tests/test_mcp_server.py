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


if __name__ == "__main__":
    unittest.main()
