"""
adaptive_replanner.py
=====================
Adjusts a learner's active roadmap based on live performance signals:
  - Assessment scores (struggling → reinforce, excelling → accelerate)
  - Course completion status
  - Skipped blocks
  - Goal changes

This module reads from the SQLite persistence layer and modifies
the roadmap ordering / difficulty in-place.
"""

import logging
from typing import Optional

log = logging.getLogger(__name__)


class AdaptiveReplanner:
    """
    Re-plans a roadmap based on learner performance signals.
    """

    # Thresholds for triggering re-planning actions
    STRUGGLE_THRESHOLD = 40.0     # assessment score % below which we reinforce
    ACCELERATE_THRESHOLD = 85.0   # assessment score % above which we accelerate

    def __init__(self, roadmap_blocks: list, assessment_history: list = None):
        """
        Parameters
        ----------
        roadmap_blocks : list of dict
            Current roadmap blocks, each with keys:
            skill_title, difficulty_level, course_title, is_completed
        assessment_history : list of dict
            Past assessment attempts with keys:
            topic, difficulty, score
        """
        self.blocks = roadmap_blocks
        self.assessments = assessment_history or []

    def analyze_performance(self) -> dict:
        """
        Analyzes assessment history to determine per-topic performance.

        Returns dict: {topic_lower: {"avg_score": float, "attempts": int, "trend": str}}
        """
        topic_scores = {}

        for attempt in self.assessments:
            topic = attempt.get("topic", "").lower().strip()
            score = attempt.get("score", 0.0)

            if topic not in topic_scores:
                topic_scores[topic] = {"scores": [], "attempts": 0}

            topic_scores[topic]["scores"].append(score)
            topic_scores[topic]["attempts"] += 1

        result = {}
        for topic, data in topic_scores.items():
            scores = data["scores"]
            avg = sum(scores) / len(scores) if scores else 0
            trend = "improving" if len(scores) > 1 and scores[-1] > scores[0] else "stable"
            if len(scores) > 1 and scores[-1] < scores[0]:
                trend = "declining"

            result[topic] = {
                "avg_score": avg,
                "attempts": data["attempts"],
                "trend": trend,
            }

        return result

    def replan(self) -> dict:
        """
        Executes the adaptive re-planning logic.

        Returns a dict with:
            - adjusted_blocks: the modified roadmap block list
            - actions_taken: list of human-readable adjustment descriptions
            - reinforcement_topics: topics that need extra practice
            - acceleration_topics: topics where learner can skip ahead
        """
        performance = self.analyze_performance()

        actions = []
        reinforce = []
        accelerate = []
        adjusted_blocks = list(self.blocks)  # shallow copy

        for block in adjusted_blocks:
            skill = block.get("skill_title", "").lower().strip()

            # Find matching performance data
            perf = None
            for topic, data in performance.items():
                if topic in skill or skill in topic:
                    perf = data
                    break

            if perf is None:
                continue

            avg_score = perf["avg_score"]

            # STRUGGLING: Reinforce this block
            if avg_score < self.STRUGGLE_THRESHOLD:
                reinforce.append(block["skill_title"])
                block["_action"] = "reinforce"
                block["_reason"] = f"Average score {avg_score:.0f}% is below {self.STRUGGLE_THRESHOLD}% threshold"
                actions.append(
                    f"🔴 REINFORCE '{block['skill_title']}': "
                    f"Avg score {avg_score:.0f}% — recommend additional practice courses and easier assessments."
                )

            # EXCELLING: Can accelerate
            elif avg_score >= self.ACCELERATE_THRESHOLD:
                accelerate.append(block["skill_title"])
                block["_action"] = "accelerate"
                block["_reason"] = f"Average score {avg_score:.0f}% exceeds {self.ACCELERATE_THRESHOLD}% threshold"

                # If this block isn't completed yet, mark as skippable
                if not block.get("is_completed"):
                    actions.append(
                        f"🟢 ACCELERATE '{block['skill_title']}': "
                        f"Avg score {avg_score:.0f}% — learner can skip to the next stage."
                    )

            # MODERATE: No change needed
            else:
                block["_action"] = "on_track"
                block["_reason"] = f"Average score {avg_score:.0f}% is within normal range"

        if not actions:
            actions.append("✅ All roadmap blocks are on track. No adjustments needed.")

        return {
            "adjusted_blocks": adjusted_blocks,
            "actions_taken": actions,
            "reinforcement_topics": reinforce,
            "acceleration_topics": accelerate,
        }

    @staticmethod
    def get_difficulty_adjustment(current_difficulty: str, action: str) -> str:
        """
        Suggests a new difficulty level based on the re-planning action.
        """
        levels = ["Easy", "Medium", "Difficult"]

        try:
            idx = levels.index(current_difficulty)
        except ValueError:
            return current_difficulty

        if action == "reinforce" and idx > 0:
            return levels[idx - 1]
        elif action == "accelerate" and idx < len(levels) - 1:
            return levels[idx + 1]

        return current_difficulty
