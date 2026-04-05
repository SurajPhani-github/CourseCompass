import os
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
# Using a local SQLite fallback. Change this connection URI to mysql+pymysql://user:pass@host/db for cloud.
DB_PATH = ROOT / "data" / "coursecompass.db"
os.makedirs(DB_PATH.parent, exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    roadmaps = relationship("SavedRoadmap", back_populates="user")
    assessments = relationship("AssessmentAttempt", back_populates="user")

class SavedRoadmap(Base):
    __tablename__ = "saved_roadmaps"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    goal_name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="roadmaps")
    blocks = relationship("RoadmapBlock", back_populates="roadmap", cascade="all, delete")

class RoadmapBlock(Base):
    __tablename__ = "roadmap_blocks"
    id = Column(Integer, primary_key=True, index=True)
    roadmap_id = Column(Integer, ForeignKey("saved_roadmaps.id"))
    step_order = Column(Integer)
    skill_title = Column(String)
    difficulty_level = Column(String)
    course_id_reference = Column(String) # Raw CSV course ID string
    course_title = Column(String)
    is_completed = Column(Boolean, default=False)
    
    roadmap = relationship("SavedRoadmap", back_populates="blocks")

class AssessmentAttempt(Base):
    __tablename__ = "assessment_attempts"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    topic = Column(String)
    difficulty = Column(String) # User chosen: Easy, Medium, Hard
    score = Column(Float) # percentage
    taken_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="assessments")

class UserProgress(Base):
    __tablename__ = "user_progress"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    total_score = Column(Float, default=0.0)
    domain_mastery_keys = Column(String, default="") # JSON or comma separated string of domains
    resources_clicked = Column(Integer, default=0)
    notes_generated = Column(Integer, default=0)
    streak_days = Column(Integer, default=0)
    last_active = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", backref="progress")

class SavedNote(Base):
    __tablename__ = "saved_notes"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    skill_topic = Column(String)
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

# Initialize schemas safely
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
