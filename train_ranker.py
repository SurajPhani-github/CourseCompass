"""
train_ranker.py
================
Standalone training script for the LightGBM ranking model.

Usage (from project root):
    python train_ranker.py

Outputs:
    models/lgbm_ranker.pkl   — trained LGBMClassifier + feature list
    (prints AUC-ROC, precision, recall, top feature importances)
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Optional imports (fail gracefully) ────────────────────────────────────────
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    log.warning("LightGBM not installed. Run: pip install lightgbm")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    log.warning("joblib not installed. Run: pip install joblib")

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

from src.recommender.ranker_features import generate_training_data


MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "lgbm_ranker.pkl"


def train(
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    early_stopping_rounds: int = 30,
    test_size: float = 0.15,
    random_state: int = 42,
) -> None:
    if not LGBM_AVAILABLE:
        log.error("Cannot train — LightGBM is not installed.")
        sys.exit(1)
    if not JOBLIB_AVAILABLE:
        log.error("Cannot save model — joblib is not installed.")
        sys.exit(1)

    # ── 1. Load features ──────────────────────────────────────────────────────
    log.info("Generating training data…")
    X, y, feature_cols, df = generate_training_data()

    # Use recency_weight as sample weight (more recent events matter more)
    sample_weights = df["recency_weight"].values if "recency_weight" in df.columns else None

    # ── 2. Train / validation split ───────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if sample_weights is not None:
        idx_train = X_train.index
        w_train = sample_weights[X.index.get_indexer(idx_train)]
    else:
        w_train = None

    log.info(
        f"Train: {len(X_train)} rows | Val: {len(X_val)} rows | "
        f"Positive rate train={y_train.mean():.2%}  val={y_val.mean():.2%}"
    )

    # ── 3. Build + train LightGBM ─────────────────────────────────────────────
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        objective="binary",
        metric="auc",
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        class_weight="balanced",
        random_state=random_state,
        verbose=-1,
    )

    log.info(f"Training LightGBM (n_estimators={n_estimators}, lr={learning_rate})…")
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    y_prob  = model.predict_proba(X_val)[:, 1]
    y_pred  = model.predict(X_val)
    auc     = roc_auc_score(y_val, y_prob)

    log.info(f"Validation AUC-ROC: {auc:.4f}")
    log.info("\n" + classification_report(y_val, y_pred, target_names=["No Success", "Success"]))

    # ── 5. Feature importances (top 15) ───────────────────────────────────────
    importances = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    log.info("Top-15 feature importances:")
    for fname, imp in importances[:15]:
        bar = "█" * int(imp / max(i for _, i in importances) * 30)
        log.info(f"  {fname:<40} {bar} ({imp})")

    # ── 6. Save model ─────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {"model": model, "feature_cols": feature_cols, "auc": round(auc, 4)}
    joblib.dump(artifact, MODEL_PATH)
    log.info(f"✅ Model saved → {MODEL_PATH}")
    log.info(f"   AUC={auc:.4f}  |  best_iteration={model.best_iteration_}")


if __name__ == "__main__":
    train()
