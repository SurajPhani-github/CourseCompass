from sqlalchemy.orm import Session
from src.app.database import SessionLocal, UserProgress, User, SavedRoadmap

def compute_leaderboard_score(prog: UserProgress, user_roadmaps: list) -> float:
    """
    Computes a composite score based on domains, streaks, and resources.
    0.30 * domain_completion + 0.20 * quiz_performance + 0.15 * streak_score + 0.10 * resources + 0.10 * breadth
    (Using simplified proxy calculations based on total_score increments)
    """
    base_activity = prog.total_score
    roadmap_bonus = len(user_roadmaps) * 10
    streak_bonus = prog.streak_days * 2
    
    return float(base_activity + roadmap_bonus + streak_bonus)

def get_overall_leaderboard(limit: int = 10) -> list[dict]:
    db = SessionLocal()
    try:
        users = db.query(User).all()
        ranking = []
        for u in users:
            prog = db.query(UserProgress).filter(UserProgress.user_id == u.id).first()
            rms = db.query(SavedRoadmap).filter(SavedRoadmap.user_id == u.id).all()
            
            if not prog:
                continue
                
            score = compute_leaderboard_score(prog, rms)
            ranking.append({
                "username": u.username,
                "score": round(score, 1),
                "roadmaps": len(rms),
                "streak": prog.streak_days
            })
            
        ranking.sort(key=lambda x: x["score"], reverse=True)
        return ranking[:limit]
    finally:
        db.close()
