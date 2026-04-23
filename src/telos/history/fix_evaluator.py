"""Fix Evaluator — rank candidate fixes using counterfactual, historical, and
developer signals.

Given a target node that is causing (or at risk of causing) a problem and a
list of candidate interventions (places where a fallback / change could be
introduced), this module scores each candidate by combining:

* **Counterfactual blast-radius reduction** — how many downstream nodes the
  fix would save (via :class:`CounterfactualAnalyzer`).
* **Historical risk** — how often the changed file has appeared in bug-fix
  commits (via :class:`GitLearner`).
* **Author expertise risk** — whether the proposed author is familiar with
  the file they'd be editing (via :class:`DeveloperModel`).
* **Pragmatic / epistemic value** — normalised active-inference style
  scores derived from the above.

The total score is a weighted blend; higher is better.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FixCandidate:
    """A proposed intervention to evaluate."""

    description: str
    intervention_at: str
    change_file: str = ""


@dataclass
class FixScore:
    """The scored result for a single :class:`FixCandidate`."""

    candidate: "FixCandidate"
    blast_radius_reduction: int = 0
    historical_risk: float = 0.0
    author_expertise_risk: float = 0.0
    pragmatic_score: float = 0.0
    epistemic_score: float = 0.5
    total_score: float = 0.0
    reasoning: list[str] = field(default_factory=list)


class FixEvaluator:
    """Rank candidate fixes by combining counterfactual, historical, and
    developer risk signals.
    """

    def __init__(
        self,
        counterfactual_analyzer,
        git_learner=None,
        developer_model=None,
        project_memory=None,
    ) -> None:
        self._counterfactual = counterfactual_analyzer
        self._git = git_learner
        self._developer = developer_model
        self._memory = project_memory

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def evaluate(
        self,
        target: str,
        candidates: list[FixCandidate],
        author: str = "",
    ) -> list[FixScore]:
        """Score each candidate and return a ranked list (best first)."""
        if not candidates:
            return []

        # -- 1. Counterfactual reductions for every candidate -------------
        reductions: list[int] = []
        cf_results: list[dict] = []
        for c in candidates:
            try:
                cf = self._counterfactual.analyze(
                    target, intervention_at=c.intervention_at
                )
            except Exception:
                cf = {
                    "without_count": 0,
                    "with_count": 0,
                    "reduction": 0,
                }
            cf_results.append(cf)
            reductions.append(int(cf.get("reduction", 0)))

        max_reduction = max(reductions) if reductions else 0

        # -- 2. Build bug-prone lookup once -------------------------------
        bug_prone_map: dict[str, float] = {}
        have_git_data = False
        if self._git is not None:
            try:
                entries = self._git.bug_prone_files(top_n=100)
                for e in entries:
                    bug_prone_map[e["file_path"]] = float(e.get("bug_rate", 0.0))
                have_git_data = True
            except Exception:
                have_git_data = False

        # -- 3. Score each candidate --------------------------------------
        scores: list[FixScore] = []
        for c, cf, reduction in zip(candidates, cf_results, reductions):
            reasoning: list[str] = []

            # Pragmatic: normalised blast-radius reduction in [0, 1].
            if max_reduction > 0:
                pragmatic = reduction / max_reduction
            else:
                pragmatic = 0.0
            reasoning.append(
                f"Counterfactual reduction = {reduction} "
                f"(without={cf.get('without_count', 0)}, "
                f"with={cf.get('with_count', 0)})."
            )

            # Historical risk from git bug-prone data.
            historical_risk = 0.0
            if have_git_data and c.change_file:
                historical_risk = bug_prone_map.get(c.change_file, 0.0)
                if historical_risk > 0:
                    reasoning.append(
                        f"{c.change_file!r} has historical bug rate "
                        f"{historical_risk:.2f}."
                    )
                else:
                    reasoning.append(
                        f"{c.change_file!r} has no significant bug-fix "
                        "history."
                    )
            elif not have_git_data:
                reasoning.append("No git history available for risk lookup.")

            # Author expertise risk.
            author_risk = 0.0
            if self._developer is not None and author and c.change_file:
                try:
                    risk_info = self._developer.risk_score_for_change(
                        author, c.change_file
                    )
                    author_risk = float(risk_info.get("risk", 0.0))
                    reason_text = risk_info.get("reasoning", "")
                    if reason_text:
                        reasoning.append(reason_text)
                except Exception:
                    author_risk = 0.0

            # Epistemic: do we actually have historical evidence?
            if have_git_data:
                epistemic = 1.0
                reasoning.append("Historical data grounds this estimate.")
            else:
                epistemic = 0.5

            total = (
                pragmatic * 0.6
                - historical_risk * 0.2
                - author_risk * 0.2
            )

            scores.append(
                FixScore(
                    candidate=c,
                    blast_radius_reduction=reduction,
                    historical_risk=historical_risk,
                    author_expertise_risk=author_risk,
                    pragmatic_score=pragmatic,
                    epistemic_score=epistemic,
                    total_score=total,
                    reasoning=reasoning,
                )
            )

        scores.sort(key=lambda s: s.total_score, reverse=True)
        return scores

    # ------------------------------------------------------------------
    # Human-friendly output
    # ------------------------------------------------------------------

    def rank_and_explain(
        self,
        target: str,
        candidates: list[FixCandidate],
        author: str = "",
    ) -> dict:
        """Return a JSON-serialisable ranking with a top-pick recommendation."""
        ranked = self.evaluate(target, candidates, author=author)

        ranked_out = []
        for i, score in enumerate(ranked):
            ranked_out.append(
                {
                    "rank": i + 1,
                    "description": score.candidate.description,
                    "intervention_at": score.candidate.intervention_at,
                    "change_file": score.candidate.change_file,
                    "total_score": round(score.total_score, 4),
                    "blast_radius_reduction": score.blast_radius_reduction,
                    "historical_risk": round(score.historical_risk, 4),
                    "author_expertise_risk": round(
                        score.author_expertise_risk, 4
                    ),
                    "pragmatic_score": round(score.pragmatic_score, 4),
                    "epistemic_score": round(score.epistemic_score, 4),
                    "reasoning": list(score.reasoning),
                }
            )

        if ranked:
            best = ranked[0]
            recommendation = (
                f"Best candidate: {best.candidate.description!r} at "
                f"{best.candidate.intervention_at!r} "
                f"(score={best.total_score:.3f}, "
                f"saves {best.blast_radius_reduction} downstream nodes)."
            )
        else:
            recommendation = "No candidates to evaluate."

        return {
            "target": target,
            "ranked_candidates": ranked_out,
            "recommendation": recommendation,
        }
