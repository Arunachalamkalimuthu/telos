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
    telos_history_patterns,
    telos_history_bug_prone,
    telos_developer_profile,
    telos_developer_risk,
    telos_suggest_reviewers,
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


class TestMCPHistoryTools(unittest.TestCase):
    """Phase 4 history/developer tools — run against the telos repo itself."""

    REPO = "."

    def _unwrap(self, raw):
        """Parse tool output and skip if we got an error (e.g., not a git repo)."""
        result = json.loads(raw)
        if isinstance(result, dict) and "error" in result:
            self.skipTest(f"tool returned error: {result['error']}")
        return result

    def test_history_patterns_returns_json(self):
        result = self._unwrap(telos_history_patterns(repo_path=self.REPO))
        self.assertIn("co_change_top", result)
        self.assertIn("bug_prone_top", result)
        self.assertIn("recent_hotspots", result)
        self.assertIn("stats", result)
        self.assertIsInstance(result["co_change_top"], list)
        self.assertIsInstance(result["bug_prone_top"], list)
        self.assertIsInstance(result["recent_hotspots"], list)
        self.assertIn("total_commits", result["stats"])

    def test_history_bug_prone(self):
        result = self._unwrap(
            telos_history_bug_prone(repo_path=self.REPO, top_n=5)
        )
        self.assertIsInstance(result, list)
        self.assertLessEqual(len(result), 5)
        for entry in result:
            self.assertIn("file_path", entry)
            self.assertIn("bug_rate", entry)
            self.assertGreaterEqual(entry["bug_rate"], 0.0)
            self.assertLessEqual(entry["bug_rate"], 1.0)

    def _pick_author(self):
        """Return some author present in the repo, or skip if none."""
        from telos.history.git_learner import GitLearner

        try:
            learner = GitLearner(self.REPO)
        except ValueError:
            self.skipTest("not a git repo")
        commits = learner.get_commits(max_count=50)
        for c in commits:
            if c.get("author"):
                return c["author"]
        self.skipTest("no authors in history")

    def test_developer_profile(self):
        author = self._pick_author()
        result = self._unwrap(
            telos_developer_profile(author=author, repo_path=self.REPO)
        )
        self.assertIn("name", result)
        self.assertEqual(result["name"], author)
        self.assertIn("expertise_files", result)
        self.assertIn("expertise_areas", result)
        self.assertIn("commit_count", result)
        self.assertIn("recent_activity_days", result)
        self.assertLessEqual(len(result["expertise_files"]), 10)

    def test_developer_risk(self):
        author = self._pick_author()
        result = self._unwrap(
            telos_developer_risk(
                author=author,
                file_path="src/telos/mcp_server.py",
                repo_path=self.REPO,
            )
        )
        self.assertIn("risk", result)
        self.assertIn("reasoning", result)
        self.assertGreaterEqual(result["risk"], 0.0)
        self.assertLessEqual(result["risk"], 1.0)

    def test_suggest_reviewers_returns_list(self):
        # Pick a file that definitely exists in history.
        from telos.history.git_learner import GitLearner

        try:
            learner = GitLearner(self.REPO)
        except ValueError:
            self.skipTest("not a git repo")
        churn = learner.file_churn(learner.get_commits(max_count=200))
        if not churn:
            self.skipTest("no file churn data")
        target_file = max(churn, key=churn.get)

        result = self._unwrap(
            telos_suggest_reviewers(
                file_path=target_file,
                repo_path=self.REPO,
                top_n=3,
            )
        )
        self.assertIsInstance(result, list)
        self.assertLessEqual(len(result), 3)
        # For a frequently-touched file we expect at least one reviewer.
        self.assertGreater(len(result), 0)

    def test_history_patterns_non_git_returns_error(self):
        tmp = tempfile.mkdtemp()
        raw = telos_history_patterns(repo_path=tmp)
        result = json.loads(raw)
        self.assertIn("error", result)

    def test_developer_profile_unknown_author_returns_error(self):
        raw = telos_developer_profile(
            author="__definitely_not_an_author__",
            repo_path=self.REPO,
        )
        result = json.loads(raw)
        # Either "not a git repo" error or "no commits found" error — both OK.
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
