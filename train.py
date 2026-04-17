"""
Phishing Email Detection System
================================
Training pipeline: data loading → preprocessing → feature engineering →
model training → evaluation → model export.

Dataset expected: CEAS phishing email dataset (CSV)
Columns used: subject, body, label  (label: 1 = phishing, 0 = safe)
"""

import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV dataset and validate required columns exist."""
    print(f"[INFO] Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)

    required_cols = {"subject", "body", "label"}
    missing = required_cols - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    df.columns = df.columns.str.lower()
    print(f"[INFO] Dataset loaded — {len(df):,} rows")
    print(f"[INFO] Label distribution:\n{df['label'].value_counts().to_string()}\n")
    return df


def preprocess_text(text: str) -> str:
    """
    Clean raw email text:
      - lowercase
      - remove HTML tags
      - remove URLs (replaced with token 'urltoken')
      - remove special chars / extra whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)                      # strip HTML
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)  # replace URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)                  # keep alphanumeric
    text = re.sub(r"\s+", " ", text).strip()                   # normalise whitespace
    return text


def build_feature_column(df: pd.DataFrame) -> pd.Series:
    """Combine subject + body into a single text feature."""
    subject = df["subject"].fillna("").astype(str)
    body    = df["body"].fillna("").astype(str)
    combined = subject + " " + body
    return combined.apply(preprocess_text)


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING  (inside Pipeline)
# ─────────────────────────────────────────────

def build_pipeline(class_weights: dict) -> Pipeline:
    """
    sklearn Pipeline:
      TF-IDF  →  Logistic Regression

    TF-IDF config:
      - sublinear_tf: dampens high-frequency term dominance
      - ngram_range (1,2): captures bigrams (e.g. "click here", "verify account")
      - max_features: keeps vocabulary manageable → avoids overfitting
      - min_df=3: ignore very rare terms (typos, noise)
    """
    tfidf = TfidfVectorizer(
        sublinear_tf=True,
        ngram_range=(1, 2),
        max_features=50_000,
        min_df=3,
        strip_accents="unicode",
        analyzer="word",
    )

    clf = LogisticRegression(
        C=1.0,                    # regularisation strength (inverse)
        solver="lbfgs",
        max_iter=1000,
        class_weight=class_weights,  # handles imbalance
        random_state=42,
    )

    return Pipeline([("tfidf", tfidf), ("clf", clf)])


# ─────────────────────────────────────────────
# 3. MODEL TRAINING
# ─────────────────────────────────────────────

def train(df: pd.DataFrame):
    """Full training run with cross-validation."""

    X = build_feature_column(df)
    y = df["label"].astype(int)

    # Class imbalance handling
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    class_weight_dict = dict(zip(classes, weights))
    print(f"[INFO] Class weights (imbalance correction): {class_weight_dict}")

    # Train / test split — stratified to preserve label ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")

    pipeline = build_pipeline(class_weight_dict)

    # 5-fold cross-validation on training set
    print("[INFO] Running 5-fold cross-validation …")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
    print(f"[CV]   F1 per fold : {np.round(cv_f1, 4)}")
    print(f"[CV]   Mean F1     : {cv_f1.mean():.4f}  ±  {cv_f1.std():.4f}\n")

    # Final fit on full training set
    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test


# ─────────────────────────────────────────────
# 4. EVALUATION
# ─────────────────────────────────────────────

def evaluate(pipeline: Pipeline, X_test, y_test):
    """Print full classification report + confusion matrix."""
    y_pred = pipeline.predict(X_test)

    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score  : {f1_score(y_test, y_pred):.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Safe", "Phishing"]))

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    print(f"  True Negatives  (Safe→Safe)          : {cm[0][0]:,}")
    print(f"  False Positives (Safe→Phishing)       : {cm[0][1]:,}")
    print(f"  False Negatives (Phishing→Safe)       : {cm[1][0]:,}")
    print(f"  True Positives  (Phishing→Phishing)   : {cm[1][1]:,}")
    print("=" * 50)


# ─────────────────────────────────────────────
# 5. SAVING MODEL
# ─────────────────────────────────────────────

def save_model(pipeline: Pipeline, output_dir: str = "model"):
    """Persist the fitted pipeline to disk."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "phishing_pipeline.pkl")
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n[INFO] Model saved → {path}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "data/ceas_dataset.csv"

    df       = load_data(dataset_path)
    pipeline, X_test, y_test = train(df)
    evaluate(pipeline, X_test, y_test)
    save_model(pipeline)
    