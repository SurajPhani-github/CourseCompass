import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from functools import lru_cache

# Simple in-memory cache for models to avoid retraining on every UI interaction
_RANKER_CACHE = {}
_MASTERY_CACHE = {}

def augment_data(interactions_df: pd.DataFrame, enriched_df: pd.DataFrame) -> pd.DataFrame:
    """Helper to join interaction logs with course metadata like duration and quality."""
    df = interactions_df.copy()
    if enriched_df is not None:
        # Normalize IDs to strings for joining
        df['course_id'] = df['course_id'].astype(str)
        enriched = enriched_df.copy()
        enriched['course_id'] = enriched['course_id'].astype(str)
        
        # Merge duration and other proxies if missing
        cols_to_use = ['course_id', 'estimated_duration_hours', 'quality_proxy', 'popularity_proxy']
        cols_to_use = [c for c in cols_to_use if c in enriched.columns]
        
        df = df.merge(enriched[cols_to_use], on='course_id', how='left')
    
    # Synthetic delta generation for training if the raw logs don't have them
    # Realistically, we'd calculate these based on learner profile at the time
    if 'difficulty_match_delta' not in df.columns:
        df['difficulty_match_delta'] = np.random.uniform(0, 1, len(df))
    if 'workload_match_delta' not in df.columns:
        df['workload_match_delta'] = np.random.uniform(0, 1, len(df))
        
    return df

from src.recommender.ranker_features import generate_training_data

def get_trained_ranker(interactions_df, courses_df=None, profiles_df=None, transitions_df=None):
    """
    Trains a Logistic Regression model using the 24+ features defined in 
    ranker_features.py. Validates the feature order for cache stability.
    """
    # ── 1. Create a stable hash of the input data for caching ─────────────────
    cache_key = int(pd.util.hash_pandas_object(interactions_df).sum())
    
    # Run the generator to see what the current pipeline produces
    # Note: we pass ALL available DataFrames into the flexible generator
    X, y, feature_cols, _ = generate_training_data(
        interactions_df=interactions_df,
        courses_df=courses_df,
        profiles_df=profiles_df,
        transitions_df=transitions_df
    )
    
    # ── 2. Cache Check with Strict Feature Order Validation ──────────────────
    if cache_key in _RANKER_CACHE:
        bundle = _RANKER_CACHE[cache_key]
        if bundle.get("feature_cols") == feature_cols:
            return bundle
        else:
            # If the feature list has changed, the old model is invalid
            import logging
            logging.getLogger(__name__).warning("Feature columns mismatch in cache. Forcing a retrain.")

    # ── 3. Train the model ───────────────────────────────────────────────────
    model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=500)
    model.fit(X, y)
    
    bundle = {
        "model": model, 
        "feature_cols": feature_cols
    }
    _RANKER_CACHE[cache_key] = bundle
    return bundle

def get_trained_mastery(interactions_df, enriched_df=None):
    cache_key = int(pd.util.hash_pandas_object(interactions_df).sum())
    if cache_key in _MASTERY_CACHE:
        return _MASTERY_CACHE[cache_key]
        
    df = augment_data(interactions_df, enriched_df)
    y = df['learning_outcome'].isin(['good', 'excellent']).astype(int)
    features = ['completion_rate', 'estimated_duration_hours']
    X = df[features].fillna(0)
    
    tree_model = DecisionTreeClassifier(max_depth=3, random_state=42, class_weight='balanced')
    tree_model.fit(X, y)
    
    _MASTERY_CACHE[cache_key] = tree_model
    return tree_model

def predict_mastery(user_history: pd.DataFrame, interactions_df: pd.DataFrame, enriched_df: pd.DataFrame) -> pd.Series:
    """Provides a robust entry point for the surgical replacement in roadmap_engine."""
    model = get_trained_mastery(interactions_df, enriched_df)
    
    # Ensure current user history has the duration column
    if 'estimated_duration_hours' not in user_history.columns:
        user_history = user_history.merge(enriched_df[['course_id', 'estimated_duration_hours']], on='course_id', how='left')
    
    median_dur = interactions_df['estimated_duration_hours'].median() if 'estimated_duration_hours' in interactions_df.columns else 10.0
    X = user_history[['completion_rate', 'estimated_duration_hours']].fillna({'completion_rate': 0, 'estimated_duration_hours': median_dur})
    return model.predict(X)

def learned_score(model_bundle, feature_dict):
    """
    Dynamically constructs a 1 x N feature array from the provided dictionary 
    in the order required by the trained model.
    """
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]

    # Construct vector in correct order. Fills missing keys with 0.0 just in case.
    vector = [float(feature_dict.get(col, 0.0)) for col in feature_cols]
    X_inference = np.array([vector])

    # Predict positive class probability
    score = model.predict_proba(X_inference)[0][1]
    return float(score)

if __name__ == "__main__":
    # Test block with dummy data
    df = pd.DataFrame({
        'learner_id': ['L1', 'L2'], 'course_id': ['C1', 'C2'], 
        'completion_rate': [0.9, 0.2], 'learning_outcome': ['excellent', 'poor']
    })
    enriched = pd.DataFrame({
        'course_id': ['C1', 'C2'], 'estimated_duration_hours': [10.0, 50.0],
        'quality_proxy': [1.0, 0.4], 'popularity_proxy': [1.0, 0.2]
    })
    
    model = get_trained_mastery(df, enriched)
    print("Mastery Tree Trained.")
    print(export_text(model, feature_names=['completion_rate', 'duration_hours']))
