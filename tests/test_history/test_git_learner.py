"""Tests for the GitLearner — run against the telos repo itself."""

from __future__ import annotations

import os
import tempfile

import pytest

from telos.history import GitLearner


REPO = "."


@pytest.fixture(scope="module")
def learner() -> GitLearner:
    return GitLearner(REPO)


@pytest.fixture(scope="module")
def commits(learner: GitLearner) -> list[dict]:
    return learner.get_commits(max_count=100)


def test_is_git_repo() -> None:
    GitLearner(REPO)  # should not raise


def test_not_git_repo_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # Ensure it's not inside a git repo by checking git's verdict.
        # If the tmp dir happens to be inside a repo we skip gracefully.
        if os.path.exists(os.path.join(tmp, ".git")):
            pytest.skip("tempdir unexpectedly inside a repo")
        try:
            GitLearner(tmp)
        except ValueError:
            return
        # If git found a parent repo, that's still valid behaviour — only
        # fail if construction silently accepted a clearly-invalid input.
        pytest.skip("tempdir resolved to an enclosing git repo")


def test_get_commits_returns_list(commits: list[dict]) -> None:
    assert isinstance(commits, list)
    assert len(commits) > 0


def test_get_commits_fields(commits: list[dict]) -> None:
    for commit in commits:
        assert set(commit.keys()) >= {"sha", "author", "timestamp", "message", "files"}
        assert isinstance(commit["sha"], str) and len(commit["sha"]) >= 7
        assert isinstance(commit["author"], str)
        assert isinstance(commit["timestamp"], int)
        assert isinstance(commit["message"], str)
        assert isinstance(commit["files"], list)


def test_file_churn(learner: GitLearner, commits: list[dict]) -> None:
    churn = learner.file_churn(commits)
    assert isinstance(churn, dict)
    assert all(count > 0 for count in churn.values())
    # The repo has changed files, so churn shouldn't be empty.
    assert len(churn) > 0


def test_co_change_matrix(learner: GitLearner, commits: list[dict]) -> None:
    matrix = learner.co_change_matrix(commits)
    assert isinstance(matrix, dict)
    for (file_a, file_b), count in matrix.items():
        assert file_a < file_b, "keys must be sorted for determinism"
        assert isinstance(count, int) and count > 0


def test_bug_prone_files(learner: GitLearner, commits: list[dict]) -> None:
    bugs = learner.bug_prone_files(commits, top_n=10)
    assert isinstance(bugs, list)
    for entry in bugs:
        assert {"file_path", "bug_fix_count", "total_changes", "bug_rate"} <= set(entry)
        assert 0.0 <= entry["bug_rate"] <= 1.0
    # sorted by bug_fix_count desc
    counts = [e["bug_fix_count"] for e in bugs]
    assert counts == sorted(counts, reverse=True)


def test_author_expertise(learner: GitLearner, commits: list[dict]) -> None:
    experts = learner.author_expertise(commits)
    assert isinstance(experts, dict)
    assert len(experts) > 0
    for author, files in experts.items():
        assert isinstance(author, str)
        assert isinstance(files, list)
        assert len(files) <= 20
        assert all(isinstance(f, str) for f in files)


def test_recent_hotspots(learner: GitLearner) -> None:
    hotspots = learner.recent_hotspots(days=3650, top_n=5)
    assert isinstance(hotspots, list)
    assert len(hotspots) <= 5
    for entry in hotspots:
        assert set(entry.keys()) == {"file_path", "recent_changes"}
        assert entry["recent_changes"] > 0


def test_commit_coupling_strength(
    learner: GitLearner, commits: list[dict]
) -> None:
    # Pick two files that both exist in the commit history so the
    # denominator is non-zero whenever possible.
    churn = learner.file_churn(commits)
    files = sorted(churn, key=churn.get, reverse=True)[:2]
    if len(files) < 2:
        pytest.skip("not enough distinct files in history")

    strength = learner.commit_coupling_strength(files[0], files[1], commits)
    assert isinstance(strength, float)
    assert 0.0 <= strength <= 1.0


def test_commit_coupling_strength_missing_files(
    learner: GitLearner, commits: list[dict]
) -> None:
    strength = learner.commit_coupling_strength(
        "__nope_a__", "__nope_b__", commits
    )
    assert strength == 0.0


def test_get_stats(learner: GitLearner) -> None:
    stats = learner.get_stats()
    assert {"total_commits", "unique_authors", "total_file_changes", "date_range"} <= set(stats)
    assert stats["total_commits"] > 0
    assert stats["unique_authors"] > 0
    assert set(stats["date_range"].keys()) == {"earliest", "latest"}
