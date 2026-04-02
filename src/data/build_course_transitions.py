"""
build_course_transitions.py
============================
Step 4 of the data pipeline.

Reads  : data/processed/synthetic_learner_interactions.csv
         data/processed/enriched_courses.csv
Writes : data/processed/course_transitions.csv

Extracts adjacent course-to-course transitions from chronological learner
sequences and aggregates them into a transition probability table.
"""

from __future__ import annotations

import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
INTERACTIONS_PATH = ROOT / "data" / "processed" / "synthetic_learner_interactions.csv"
ENRICHED_PATH = ROOT / "data" / "processed" / "enriched_courses.csv"
OUT_PATH = ROOT / "data" / "processed" / "course_transitions.csv"


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading interactions …")
    interactions = pd.read_csv(INTERACTIONS_PATH, low_memory=False)
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])

    log.info(f"Loading enriched courses …")
    enriched = pd.read_csv(ENRICHED_PATH, low_memory=False)
    enriched["course_id"] = enriched["course_id"].astype(str)
    # Build lookup dict manually to handle any duplicate course_ids
    course_meta: dict[str, dict] = {}
    for _, row in enriched[["course_id", "inferred_domain", "difficulty_score", "workload_score"]].iterrows():
        cid = str(row["course_id"])
        if cid not in course_meta:
            course_meta[cid] = {
                "inferred_domain": row["inferred_domain"],
                "difficulty_score": row["difficulty_score"],
                "workload_score": row["workload_score"],
            }

    log.info("Building transition pairs …")

    # For each learner, sort by timestamp and extract (prev, next) pairs
    transition_data: defaultdict[tuple[str, str], list[dict]] = defaultdict(list)

    for learner_id, g in interactions.groupby("learner_id"):
        g = g.sort_values("timestamp")
        courses = g["course_id"].tolist()
        timestamps = g["timestamp"].tolist()
        outcomes = g["learning_outcome"].tolist()
        completion_rates = g["completion_rate"].tolist()

        for j in range(len(courses) - 1):
            prev_cid = courses[j]
            next_cid = courses[j + 1]
            if prev_cid == next_cid:
                continue  # skip self-loops

            time_gap = (timestamps[j + 1] - timestamps[j]).days
            outcome = outcomes[j]
            comp = completion_rates[j]

            transition_data[(prev_cid, next_cid)].append({
                "time_gap_days": abs(time_gap),
                "from_outcome": outcome,
                "from_completion": comp,
            })

    log.info(f"  Found {len(transition_data):,} unique transition pairs")

    # Compute prev_course total counts for probability calculation
    prev_totals: defaultdict[str, int] = defaultdict(int)
    for (prev, next_), records in transition_data.items():
        prev_totals[prev] += len(records)

    # Build rows
    rows = []
    for (prev_cid, next_cid), records in transition_data.items():
        count = len(records)
        prev_total = prev_totals[prev_cid]
        prob = count / prev_total

        avg_gap = float(np.mean([r["time_gap_days"] for r in records]))

        # High success: outcome is excellent or good
        high_success_count = sum(
            1 for r in records
            if r["from_outcome"] in ("excellent", "good") and r["from_completion"] > 0.65
        )
        success_rate = high_success_count / count

        # Meta from enriched
        prev_meta = course_meta.get(prev_cid, {})
        next_meta = course_meta.get(next_cid, {})

        prev_domain = prev_meta.get("inferred_domain", "General Studies")
        next_domain = next_meta.get("inferred_domain", "General Studies")
        prev_diff = float(prev_meta.get("difficulty_score", 2))
        next_diff = float(next_meta.get("difficulty_score", 2))

        same_domain = int(prev_domain == next_domain)
        diff_prog = int(next_diff >= prev_diff)  # not going backward in difficulty

        # high_success_transition_score: combine probability, success_rate, same_domain bonus
        h_score = round(
            0.40 * min(prob * 10, 1.0) +     # boost probabilities, cap
            0.35 * success_rate +
            0.15 * same_domain +
            0.10 * diff_prog,
            4
        )

        rows.append({
            "prev_course_id": prev_cid,
            "next_course_id": next_cid,
            "prev_domain": prev_domain,
            "next_domain": next_domain,
            "prev_difficulty_score": prev_diff,
            "next_difficulty_score": next_diff,
            "transition_count": count,
            "transition_probability": round(prob, 6),
            "same_domain_flag": same_domain,
            "difficulty_progression_flag": diff_prog,
            "avg_time_gap_days": round(avg_gap, 2),
            "high_success_transition_score": h_score,
        })

    transitions_df = pd.DataFrame(rows)
    transitions_df = transitions_df.sort_values("transition_count", ascending=False).reset_index(drop=True)

    transitions_df.to_csv(OUT_PATH, index=False)
    log.info(f"Saved {len(transitions_df):,} transitions → {OUT_PATH}")
    log.info(f"  Max transition count: {transitions_df['transition_count'].max()}")
    log.info(f"  Same-domain transitions: {transitions_df['same_domain_flag'].mean():.1%}")
    log.info(f"  Difficulty-progressive transitions: {transitions_df['difficulty_progression_flag'].mean():.1%}")


if __name__ == "__main__":
    main()
