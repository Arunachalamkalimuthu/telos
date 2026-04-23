"""Developer Model: build AgentMind profiles from git history.

Each developer in the git log is modelled as an AgentMind whose
``capabilities`` are the top-level directories they have committed to.
This lets telos reason about *what a developer knows* vs *what they're
changing* — touching code outside one's expertise is a risk signal for
mistakes, and it surfaces sensible reviewer suggestions.
"""

from __future__ import annotations

import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import median

from ..theory_of_mind import AgentMind
from ..world import WorldState


_CO_AUTHOR_RE = re.compile(
    r"^\s*co-authored-by:\s*(.+?)\s*<[^>]*>\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _top_level_dir(path: str) -> str:
    """Return the top-level directory for ``path``.

    ``src/auth.py`` → ``src``; ``docs/guide.md`` → ``docs``;
    ``README.md`` → ``""`` (file lives at the repo root).
    """
    if not path:
        return ""
    # git log emits forward-slash separated paths regardless of platform.
    head, sep, _ = path.partition("/")
    if not sep:
        return ""
    return head


def _parse_co_authors(message: str) -> list[str]:
    if not message:
        return []
    return [m.group(1).strip() for m in _CO_AUTHOR_RE.finditer(message)]


@dataclass
class DeveloperProfile:
    name: str
    expertise_files: list[str] = field(default_factory=list)
    expertise_areas: list[str] = field(default_factory=list)
    commit_count: int = 0
    recent_activity_days: int = 0
    co_authors: list[str] = field(default_factory=list)
    typical_commit_size: int = 0

    def knows_file(self, file_path: str) -> bool:
        """True iff ``file_path`` is among the developer's top touched files."""
        return file_path in self.expertise_files

    def knows_area(self, file_path: str) -> bool:
        """True iff ``file_path``'s top-level directory is a known area."""
        area = _top_level_dir(file_path)
        if not area:
            return False
        return area in self.expertise_areas

    def to_agent_mind(self) -> AgentMind:
        """Convert this profile to a telos ``AgentMind``.

        Capabilities are the expertise areas (top-level dirs). Goals and
        beliefs are deliberately empty — we can't infer intent from a log
        of past commits alone.
        """
        return AgentMind(
            id=self.name,
            beliefs=WorldState(),
            goals=(),
            capabilities=frozenset(self.expertise_areas),
            actions=("commit", "review", "refactor"),
        )


class DeveloperModel:
    """Build ``DeveloperProfile`` instances from a ``GitLearner``."""

    def __init__(self, git_learner) -> None:
        self.git_learner = git_learner

    # ------------------------------------------------------------------
    # Profile construction
    # ------------------------------------------------------------------
    def build_profiles(self, max_commits: int = 500) -> dict[str, DeveloperProfile]:
        commits = self.git_learner.get_commits(max_count=max_commits)
        return self._profiles_from_commits(commits)

    def profile_for(
        self, author: str, max_commits: int = 500
    ) -> DeveloperProfile | None:
        profiles = self.build_profiles(max_commits=max_commits)
        return profiles.get(author)

    def _profiles_from_commits(
        self, commits: list[dict]
    ) -> dict[str, DeveloperProfile]:
        file_counts: dict[str, Counter[str]] = defaultdict(Counter)
        area_counts: dict[str, Counter[str]] = defaultdict(Counter)
        commit_counts: Counter[str] = Counter()
        latest_ts: dict[str, int] = {}
        co_authors: dict[str, Counter[str]] = defaultdict(Counter)
        commit_sizes: dict[str, list[int]] = defaultdict(list)

        for commit in commits:
            author = commit.get("author", "")
            if not author:
                continue
            files = list(set(commit.get("files", [])))
            commit_counts[author] += 1
            commit_sizes[author].append(len(files))

            ts = commit.get("timestamp", 0) or 0
            if ts > latest_ts.get(author, 0):
                latest_ts[author] = ts

            for path in files:
                file_counts[author][path] += 1
                area = _top_level_dir(path)
                if area:
                    area_counts[author][area] += 1

            for co in _parse_co_authors(commit.get("message", "")):
                if co and co != author:
                    co_authors[author][co] += 1

        now = int(time.time())
        profiles: dict[str, DeveloperProfile] = {}

        for author, count in commit_counts.items():
            areas_sorted = [a for a, _ in area_counts[author].most_common()]
            files_sorted = [p for p, _ in file_counts[author].most_common(20)]
            last_ts = latest_ts.get(author, 0)
            if last_ts:
                days = max(0, (now - last_ts) // 86400)
            else:
                days = 0
            sizes = commit_sizes.get(author) or [0]
            typical = int(median(sizes)) if sizes else 0
            co_sorted = [c for c, _ in co_authors[author].most_common()]

            profiles[author] = DeveloperProfile(
                name=author,
                expertise_files=files_sorted,
                expertise_areas=areas_sorted,
                commit_count=count,
                recent_activity_days=int(days),
                co_authors=co_sorted,
                typical_commit_size=typical,
            )

        return profiles

    # ------------------------------------------------------------------
    # Risk scoring
    # ------------------------------------------------------------------
    def risk_score_for_change(
        self,
        author: str,
        file_path: str,
        max_commits: int = 500,
    ) -> dict:
        profiles = self.build_profiles(max_commits=max_commits)
        profile = profiles.get(author)

        if profile is None:
            return {
                "author": author,
                "file_path": file_path,
                "knows_file": False,
                "knows_area": False,
                "risk": 1.0,
                "reasoning": (
                    f"{author!r} has no commits in the history window — "
                    "unknown developer, highest risk."
                ),
            }

        # Very few commits → treat similarly to unknown.
        if profile.commit_count < 2:
            return {
                "author": author,
                "file_path": file_path,
                "knows_file": profile.knows_file(file_path),
                "knows_area": profile.knows_area(file_path),
                "risk": 1.0,
                "reasoning": (
                    f"{author!r} has only {profile.commit_count} commit(s) — "
                    "too little history to trust."
                ),
            }

        if profile.knows_file(file_path):
            return {
                "author": author,
                "file_path": file_path,
                "knows_file": True,
                "knows_area": True,
                "risk": 0.0,
                "reasoning": (
                    f"{author!r} has touched {file_path!r} before — "
                    "file is in their expertise."
                ),
            }

        if profile.knows_area(file_path):
            area = _top_level_dir(file_path)
            return {
                "author": author,
                "file_path": file_path,
                "knows_file": False,
                "knows_area": True,
                "risk": 0.3,
                "reasoning": (
                    f"{author!r} works in {area!r} but has not touched "
                    f"{file_path!r} before — moderate familiarity."
                ),
            }

        return {
            "author": author,
            "file_path": file_path,
            "knows_file": False,
            "knows_area": False,
            "risk": 0.7,
            "reasoning": (
                f"{author!r} is an active contributor but has never touched "
                f"the {_top_level_dir(file_path) or 'root'!r} area — elevated risk."
            ),
        }

    # ------------------------------------------------------------------
    # Reviewer suggestions
    # ------------------------------------------------------------------
    def suggest_reviewers(
        self,
        file_path: str,
        exclude_author: str = "",
        top_n: int = 3,
        max_commits: int = 500,
    ) -> list[str]:
        """Rank authors by how many times they've touched ``file_path``."""
        commits = self.git_learner.get_commits(max_count=max_commits)
        touches: Counter[str] = Counter()

        area = _top_level_dir(file_path)

        for commit in commits:
            author = commit.get("author", "")
            if not author or author == exclude_author:
                continue
            files = set(commit.get("files", []))
            if file_path in files:
                touches[author] += 2  # direct file hit weighs more
            elif area and any(_top_level_dir(f) == area for f in files):
                touches[author] += 1  # same-area hit

        return [author for author, _ in touches.most_common(top_n)]
