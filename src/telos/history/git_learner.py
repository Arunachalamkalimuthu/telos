"""Learn patterns from git history.

Shells out to `git log` and derives signals that complement the static
code graph: co-change coupling, churn, bug-prone files, author expertise
and recent hotspots. These signals can be used to up-weight edges in the
causal graph based on real historical evidence.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations


_BUG_RE = re.compile(r"\b(fix|bug|issue|error|crash|regression|broken)\b")


class GitLearner:
    """Learn patterns from git history using subprocess calls to git."""

    def __init__(self, repo_path: str) -> None:
        self.repo_path = os.path.abspath(repo_path)
        if not os.path.isdir(self.repo_path):
            raise ValueError(f"Not a directory: {self.repo_path}")
        try:
            subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise ValueError(f"Not a git repository: {self.repo_path}") from exc

    # ------------------------------------------------------------------
    # Primitive: fetch commits
    # ------------------------------------------------------------------
    def get_commits(
        self, since: str | None = None, max_count: int = 1000
    ) -> list[dict]:
        """Run ``git log`` and parse commits into dicts."""
        cmd = [
            "git",
            "log",
            "--name-only",
            "--pretty=format:%H|%an|%at|%s",
            f"-n{max_count}",
        ]
        if since:
            cmd.append(f"--since={since}")

        result = subprocess.run(
            cmd,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return self._parse_log(result.stdout)

    @staticmethod
    def _parse_log(output: str) -> list[dict]:
        commits: list[dict] = []
        current: dict | None = None

        for raw_line in output.splitlines():
            line = raw_line.rstrip()

            if not line:
                if current is not None:
                    commits.append(current)
                    current = None
                continue

            # Header lines contain 3 pipe separators. File paths might
            # contain pipes in exotic cases, so anchor on a 40-char SHA.
            if current is None and _looks_like_header(line):
                sha, author, ts, message = line.split("|", 3)
                try:
                    timestamp = int(ts)
                except ValueError:
                    timestamp = 0
                current = {
                    "sha": sha,
                    "author": author,
                    "timestamp": timestamp,
                    "message": message,
                    "files": [],
                }
            elif current is not None:
                current["files"].append(line)

        if current is not None:
            commits.append(current)

        return commits

    def _commits(self, commits: list[dict] | None) -> list[dict]:
        return commits if commits is not None else self.get_commits()

    # ------------------------------------------------------------------
    # Co-change / churn
    # ------------------------------------------------------------------
    def co_change_matrix(
        self, commits: list[dict] | None = None
    ) -> dict[tuple[str, str], int]:
        """Return ``{(file_a, file_b): count}`` where ``file_a < file_b``."""
        commits = self._commits(commits)
        pairs: Counter[tuple[str, str]] = Counter()
        for commit in commits:
            files = sorted(set(commit.get("files", [])))
            if len(files) < 2:
                continue
            for a, b in combinations(files, 2):
                pairs[(a, b)] += 1
        return dict(pairs)

    def file_churn(self, commits: list[dict] | None = None) -> dict[str, int]:
        """Return ``{file: commit_count}`` across the supplied history."""
        commits = self._commits(commits)
        counts: Counter[str] = Counter()
        for commit in commits:
            for path in set(commit.get("files", [])):
                counts[path] += 1
        return dict(counts)

    # ------------------------------------------------------------------
    # Bug-prone files
    # ------------------------------------------------------------------
    def bug_prone_files(
        self, commits: list[dict] | None = None, top_n: int = 20
    ) -> list[dict]:
        """Rank files by how often they appear in bug-fix commits."""
        commits = self._commits(commits)

        total_changes: Counter[str] = Counter()
        bug_changes: Counter[str] = Counter()

        for commit in commits:
            files = set(commit.get("files", []))
            is_bug = bool(_BUG_RE.search(commit.get("message", "").lower()))
            for path in files:
                total_changes[path] += 1
                if is_bug:
                    bug_changes[path] += 1

        results: list[dict] = []
        for path, bug_count in bug_changes.items():
            total = total_changes[path]
            results.append(
                {
                    "file_path": path,
                    "bug_fix_count": bug_count,
                    "total_changes": total,
                    "bug_rate": bug_count / total if total else 0.0,
                }
            )

        results.sort(
            key=lambda r: (r["bug_fix_count"], r["bug_rate"]), reverse=True
        )
        return results[:top_n]

    # ------------------------------------------------------------------
    # Author expertise
    # ------------------------------------------------------------------
    def author_expertise(
        self, commits: list[dict] | None = None
    ) -> dict[str, list[str]]:
        """Return ``{author: [most-touched files, ...]}`` (top 20 each)."""
        commits = self._commits(commits)
        by_author: dict[str, Counter[str]] = defaultdict(Counter)

        for commit in commits:
            author = commit.get("author", "")
            for path in set(commit.get("files", [])):
                by_author[author][path] += 1

        return {
            author: [path for path, _ in counts.most_common(20)]
            for author, counts in by_author.items()
        }

    # ------------------------------------------------------------------
    # Recent hotspots
    # ------------------------------------------------------------------
    def recent_hotspots(self, days: int = 30, top_n: int = 10) -> list[dict]:
        """Files most changed in the last ``days`` days."""
        commits = self.get_commits(since=f"{days} days ago")
        churn = self.file_churn(commits)
        ranked = sorted(churn.items(), key=lambda kv: kv[1], reverse=True)
        return [
            {"file_path": path, "recent_changes": count}
            for path, count in ranked[:top_n]
        ]

    # ------------------------------------------------------------------
    # Coupling strength
    # ------------------------------------------------------------------
    def commit_coupling_strength(
        self,
        file_a: str,
        file_b: str,
        commits: list[dict] | None = None,
    ) -> float:
        """Fraction of min(changes to a, changes to b) that are co-changes."""
        commits = self._commits(commits)

        a_count = 0
        b_count = 0
        co_count = 0
        for commit in commits:
            files = set(commit.get("files", []))
            has_a = file_a in files
            has_b = file_b in files
            if has_a:
                a_count += 1
            if has_b:
                b_count += 1
            if has_a and has_b:
                co_count += 1

        denom = min(a_count, b_count)
        if denom == 0:
            return 0.0
        return co_count / denom

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        """Aggregate statistics over the default history window."""
        commits = self.get_commits()

        authors = {c.get("author", "") for c in commits}
        total_file_changes = sum(len(c.get("files", [])) for c in commits)

        timestamps = [c.get("timestamp", 0) for c in commits if c.get("timestamp")]
        if timestamps:
            earliest = datetime.fromtimestamp(min(timestamps)).isoformat()
            latest = datetime.fromtimestamp(max(timestamps)).isoformat()
        else:
            earliest = latest = ""

        return {
            "total_commits": len(commits),
            "unique_authors": len(authors),
            "total_file_changes": total_file_changes,
            "date_range": {"earliest": earliest, "latest": latest},
        }


_SHA_RE = re.compile(r"^[0-9a-f]{7,40}\|")


def _looks_like_header(line: str) -> bool:
    """Heuristic: real header lines start with a hex SHA followed by ``|``."""
    return bool(_SHA_RE.match(line)) and line.count("|") >= 3
