from src.app.database import SessionLocal, UserProgress, User, SavedRoadmap, RoadmapBlock

def get_user_dashboard_stats(user_id: int):
    db = SessionLocal()
    try:
        prog = db.query(UserProgress).filter(UserProgress.user_id == user_id).first()
        roadmaps = db.query(SavedRoadmap).filter(SavedRoadmap.user_id == user_id).all()
        
        total_rm = len(roadmaps)
        
        # Calculate skill block completion
        total_blocks = 0
        completed_blocks = 0
        for rm in roadmaps:
            total_blocks += len(rm.blocks)
            completed_blocks += sum(1 for b in rm.blocks if b.is_completed)
            
        skill_completion_rate = (completed_blocks / total_blocks) * 100 if total_blocks > 0 else 0
        
        return {
            "total_roadmaps": total_rm,
            "skill_completion_rate": skill_completion_rate,
            "resources_clicked": prog.resources_clicked if prog else 0,
            "notes_generated": prog.notes_generated if prog else 0,
            "total_score": prog.total_score if prog else 0.0,
            "streak_days": prog.streak_days if prog else 0
        }
    finally:
        db.close()

def log_resource_click(user_id: int):
    db = SessionLocal()
    try:
        prog = db.query(UserProgress).filter(UserProgress.user_id == user_id).first()
        if prog:
            prog.resources_clicked += 1
            prog.total_score += 1.5 # +1.5 points for resources
            db.commit()
    finally:
        db.close()
        
def log_notes_generated(user_id: int):
    db = SessionLocal()
    try:
        prog = db.query(UserProgress).filter(UserProgress.user_id == user_id).first()
        if prog:
            prog.notes_generated += 1
            prog.total_score += 5.0 # +5 points for deep study
            db.commit()
    finally:
        db.close()
