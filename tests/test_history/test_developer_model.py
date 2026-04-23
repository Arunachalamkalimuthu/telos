"""Tests for the DeveloperModel — run against the telos repo itself."""

from __future__ import annotations

import pytest

from telos.history import DeveloperModel, DeveloperProfile, GitLearner
from telos.theory_of_mind import AgentMind


REPO = "."
MAX_COMMITS = 200


@pytest.fixture(scope="module")
def learner() -> GitLearner:
    return GitLearner(REPO)


@pytest.fixture(scope="module")
def model(learner: GitLearner) -> DeveloperModel:
    return DeveloperModel(learner)


@pytest.fixture(scope="module")
def profiles(model: DeveloperModel) -> dict[str, DeveloperProfile]:
    return model.build_profiles(max_commits=MAX_COMMITS)


@pytest.fixture(scope="module")
def top_author(profiles: dict[str, DeveloperProfile]) -> str:
    # Pick the author with the most commits — guaranteed to have expertise.
    return max(profiles.items(), key=lambda kv: kv[1].commit_count)[0]


def test_build_profiles_returns_dict(profiles: dict[str, DeveloperProfile]) -> None:
    assert isinstance(profiles, dict)
    assert len(profiles) > 0
    for name, profile in profiles.items():
        assert isinstance(name, str)
        assert isinstance(profile, DeveloperProfile)


def test_profile_has_expected_fields(
    profiles: dict[str, DeveloperProfile], top_author: str
) -> None:
    profile = profiles[top_author]
    assert profile.name == top_author
    assert isinstance(profile.expertise_files, list)
    assert isinstance(profile.expertise_areas, list)
    assert isinstance(profile.commit_count, int)
    assert profile.commit_count >= 1
    assert isinstance(profile.recent_activity_days, int)
    assert profile.recent_activity_days >= 0
    assert isinstance(profile.co_authors, list)
    assert isinstance(profile.typical_commit_size, int)
    assert profile.typical_commit_size >= 0


def test_profile_for_known_author(model: DeveloperModel, top_author: str) -> None:
    profile = model.profile_for(top_author, max_commits=MAX_COMMITS)
    assert profile is not None
    assert profile.name == top_author


def test_profile_for_unknown_author(model: DeveloperModel) -> None:
    profile = model.profile_for(
        "Definitely Not A Real Author 12345", max_commits=MAX_COMMITS
    )
    assert profile is None


def test_knows_file_for_touched_file(
    profiles: dict[str, DeveloperProfile], top_author: str
) -> None:
    profile = profiles[top_author]
    assert profile.expertise_files, "top author should have expertise files"
    touched = profile.expertise_files[0]
    assert profile.knows_file(touched) is True
    assert profile.knows_file("nonexistent/path/file.xyz") is False


def test_knows_area_for_touched_dir(
    profiles: dict[str, DeveloperProfile], top_author: str
) -> None:
    profile = profiles[top_author]
    assert profile.expertise_areas, "top author should have expertise areas"
    # Find the first file that sits inside a known top-level directory.
    # Root-level files (e.g. README.md) legitimately have no "area".
    nested = next(
        (f for f in profile.expertise_files if "/" in f),
        None,
    )
    assert nested is not None, "top author should have at least one nested file"
    assert profile.knows_area(nested) is True
    assert profile.knows_area("totally-unknown-area-xyz/thing.py") is False


def test_to_agent_mind_returns_agent_mind(
    profiles: dict[str, DeveloperProfile], top_author: str
) -> None:
    profile = profiles[top_author]
    mind = profile.to_agent_mind()
    assert isinstance(mind, AgentMind)
    assert mind.id == top_author
    assert mind.capabilities == frozenset(profile.expertise_areas)
    assert mind.actions == ("commit", "review", "refactor")
    assert mind.goals == ()


def test_risk_score_knows_file(
    model: DeveloperModel,
    profiles: dict[str, DeveloperProfile],
    top_author: str,
) -> None:
    touched = profiles[top_author].expertise_files[0]
    result = model.risk_score_for_change(
        top_author, touched, max_commits=MAX_COMMITS
    )
    assert result["author"] == top_author
    assert result["file_path"] == touched
    assert result["knows_file"] is True
    assert result["risk"] == 0.0
    assert "reasoning" in result and result["reasoning"]


def test_risk_score_unknown_file_known_area(
    model: DeveloperModel,
    profiles: dict[str, DeveloperProfile],
    top_author: str,
) -> None:
    profile = profiles[top_author]
    # Pick an area the author knows, but fabricate an untouched file path.
    assert profile.expertise_areas
    area = profile.expertise_areas[0]
    fabricated = f"{area}/__this_file_does_not_exist_in_history__.xyz"
    assert fabricated not in profile.expertise_files

    result = model.risk_score_for_change(
        top_author, fabricated, max_commits=MAX_COMMITS
    )
    assert result["knows_file"] is False
    assert result["knows_area"] is True
    assert result["risk"] == 0.3


def test_risk_score_unknown_author(model: DeveloperModel) -> None:
    result = model.risk_score_for_change(
        "Definitely Not A Real Author 12345",
        "src/telos/world.py",
        max_commits=MAX_COMMITS,
    )
    assert result["risk"] == 1.0
    assert result["knows_file"] is False
    assert result["knows_area"] is False


def test_suggest_reviewers_returns_list(
    model: DeveloperModel,
    profiles: dict[str, DeveloperProfile],
    top_author: str,
) -> None:
    touched = profiles[top_author].expertise_files[0]
    reviewers = model.suggest_reviewers(
        touched, top_n=3, max_commits=MAX_COMMITS
    )
    assert isinstance(reviewers, list)
    assert len(reviewers) >= 1
    assert len(reviewers) <= 3
    assert all(isinstance(r, str) for r in reviewers)


def test_suggest_reviewers_excludes_author(
    model: DeveloperModel,
    profiles: dict[str, DeveloperProfile],
    top_author: str,
) -> None:
    touched = profiles[top_author].expertise_files[0]
    reviewers = model.suggest_reviewers(
        touched,
        exclude_author=top_author,
        top_n=5,
        max_commits=MAX_COMMITS,
    )
    assert top_author not in reviewers
