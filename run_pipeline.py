"""
run_pipeline.py
================
Master orchestrator for the Hybrid Course Recommendation System data pipeline.

Run from the project root:
    python run_pipeline.py

Steps:
    1. Enrich courses       → data/processed/enriched_courses.csv
    2. Generate learners    → data/processed/synthetic_learner_interactions.csv
    3. Build profiles       → data/processed/learner_profiles.csv
    4. Build transitions    → data/processed/course_transitions.csv
"""

from __future__ import annotations

import sys
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_step(name: str, func) -> float:
    log.info(f"\n{'='*60}")
    log.info(f"  STEP: {name}")
    log.info(f"{'='*60}")
    t0 = time.time()
    func()
    elapsed = time.time() - t0
    log.info(f"  ✓ {name} completed in {elapsed:.1f}s")
    return elapsed


def check_raw_data() -> None:
    raw_path = ROOT / "data" / "raw" / "coursera_courses.csv"
    if not raw_path.exists():
        log.error(f"Raw data not found: {raw_path}")
        log.error("Please copy coursera_courses.csv to data/raw/coursera_courses.csv")
        sys.exit(1)
    log.info(f"✓ Raw data found: {raw_path}")


def main():
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║   Hybrid Course Recommendation System — Data Pipeline    ║")
    log.info("╚══════════════════════════════════════════════════════════╝")

    check_raw_data()

    # ── Step 1: Enrich courses ────────────────────────────────────────────────
    from src.data.enrich_courses import main as enrich_main
    t1 = run_step("1/4  Enrich courses", enrich_main)

    # ── Step 2: Generate synthetic learner interactions ───────────────────────
    from src.data.generate_synthetic_learners import main as learner_main
    t2 = run_step("2/4  Generate synthetic learner interactions", learner_main)

    # ── Step 3: Build learner profiles ────────────────────────────────────────
    from src.data.build_learner_profiles import main as profile_main
    t3 = run_step("3/4  Build learner profiles", profile_main)

    # ── Step 4: Build course transitions ─────────────────────────────────────
    from src.data.build_course_transitions import main as transitions_main
    t4 = run_step("4/4  Build course transitions", transitions_main)

    total = t1 + t2 + t3 + t4

    log.info(f"\n{'='*60}")
    log.info(f"  ✅  ALL STEPS COMPLETE  ({total:.1f}s total)")
    log.info(f"{'='*60}")
    log.info("\nOutput files:")
    for f in (ROOT / "data" / "processed").glob("*.csv"):
        size_mb = f.stat().st_size / 1_048_576
        log.info(f"  {f.name:<50} {size_mb:.2f} MB")

    log.info("\nNext step:")
    log.info("  streamlit run src/app/streamlit_app.py")


if __name__ == "__main__":
    main()
