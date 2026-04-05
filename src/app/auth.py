import streamlit as st
from datetime import datetime
from src.app.database import SessionLocal, User, UserProgress

def get_current_user():
    """Returns the currently logged in User object, or None if not logged in."""
    if "current_username" not in st.session_state:
        return None
        
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == st.session_state.current_username).first()
        return user
    finally:
        db.close()

def login_ui():
    """Renders a simple login/register UI in Streamlit."""
    st.markdown('<div class="section-header">🔐 Welcome to CourseCompass</div>', unsafe_allow_html=True)
    st.markdown("Please log in to build your adaptive roadmaps, save your progress, and climb the leaderboards.")
    
    with st.container():
        username = st.text_input("Username (creates a new account if it doesn't exist)", key="login_input")
        
        if st.button("Log In / Register", use_container_width=True):
            if username.strip():
                _process_login(username.strip())
            else:
                st.warning("Please enter a valid username.")

def _process_login(username: str):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user:
            # Register new user
            user = User(username=username)
            db.add(user)
            db.commit()
            db.refresh(user)
            
            # Init empty progress
            prog = UserProgress(user_id=user.id)
            db.add(prog)
            db.commit()
            
            st.success(f"Welcome, {username}! Account created successfully.")
        else:
            # Update last active
            prog = db.query(UserProgress).filter(UserProgress.user_id == user.id).first()
            if prog:
                prog.last_active = datetime.utcnow()
                db.commit()
            st.success(f"Welcome back, {username}!")
            
        st.session_state.current_username = username
        st.rerun()
    finally:
        db.close()
        
def logout():
    """Logs out the user."""
    if "current_username" in st.session_state:
        del st.session_state.current_username
    st.rerun()
