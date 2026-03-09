"""
AI Credit Risk Analyzer - Machine Learning Model
Trains and saves Logistic Regression and Random Forest models.
"""

import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_processing import load_dataset, preprocess_for_training

MODEL_PATH = Path("models/credit_model.pkl")


def train_model() -> dict:
    """
    Train ensemble of Logistic Regression and Random Forest.
    Returns dict with models and metadata.
    """
    df = load_dataset()
    X, y = preprocess_for_training(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Logistic Regression - good baseline for probability
    lr_model = LogisticRegression(
        max_iter=2000,
        random_state=42,
        class_weight="balanced",
    )
    lr_model.fit(X_train_scaled, y_train)
    
    # Random Forest - no scaling needed, uses raw features
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )
    rf_model.fit(X_train, y_train)
    
    # Store ensemble for inference (LR uses scaled, RF uses raw)
    model_package = {
        "lr_model": lr_model,
        "rf_model": rf_model,
        "scaler": scaler,
        "feature_columns": list(X.columns),
        "lr_coefficients": lr_model.coef_[0] if hasattr(lr_model, "coef_") else None,
        "rf_feature_importances": rf_model.feature_importances_,
    }
    
    return model_package


def save_model(model_package: dict, path: Path = MODEL_PATH) -> None:
    """Save trained model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_package, path)


def load_model(path: Path = MODEL_PATH) -> dict | None:
    """Load model from disk. Returns None if not found."""
    if not path.exists():
        return None
    return joblib.load(path)


def get_or_train_model() -> dict:
    """
    Load saved model if exists, otherwise train and save.
    """
    model = load_model()
    if model is not None:
        return model
    
    model_package = train_model()
    save_model(model_package)
    return model_package
