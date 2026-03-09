"""
AI Credit Risk Analyzer - Data Processing
Handles dataset loading, preprocessing, and feature preparation.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Expected feature columns for model training and prediction
FEATURE_COLUMNS = [
    "age",
    "income",
    "employment_length",
    "credit_history_length",
    "loan_amount",
    "debt_to_income",
    "existing_loans",
    "late_payments",
]


def load_dataset(filepath: str = "dataset/loan_dataset.csv") -> pd.DataFrame:
    """Load loan dataset from CSV."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    return pd.read_csv(path)


def preprocess_for_training(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for model training.
    Returns (X, y) where X is features and y is target.
    """
    # Map column names for batch upload flexibility
    column_mapping = {
        "age": "age",
        "income": "income",
        "annual_income": "income",
        "employment_length": "employment_length",
        "employment": "employment_length",
        "credit_history_length": "credit_history_length",
        "credit_history": "credit_history_length",
        "loan_amount": "loan_amount",
        "loan": "loan_amount",
        "debt_to_income": "debt_to_income",
        "debt_to_income_ratio": "debt_to_income",
        "dti": "debt_to_income",
        "existing_loans": "existing_loans",
        "num_loans": "existing_loans",
        "number_of_loans": "existing_loans",
        "late_payments": "late_payments",
        "late_payment_history": "late_payments",
    }
    
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    for old, new in column_mapping.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    
    # Ensure all required columns exist
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    X = df[FEATURE_COLUMNS].copy()
    X = X.fillna(X.median())
    
    if "default" in df.columns:
        y = df["default"]
    else:
        y = None
    
    return X, y


def prepare_single_applicant(data: dict) -> pd.DataFrame:
    """
    Convert single applicant form data to DataFrame for prediction.
    """
    df = pd.DataFrame([{
        "age": int(data.get("age", 0)),
        "income": float(data.get("income", 0)),
        "employment_length": float(data.get("employment_length", 0)),
        "credit_history_length": float(data.get("credit_history_length", 0)),
        "loan_amount": float(data.get("loan_amount", 0)),
        "debt_to_income": float(data.get("debt_to_income", 0)),
        "existing_loans": int(data.get("existing_loans", 0)),
        "late_payments": int(data.get("late_payments", 0)),
    }])
    return df[FEATURE_COLUMNS]


def validate_batch_upload(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validate batch CSV for required columns.
    Returns (is_valid, message).
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    
    column_mapping = {
        "annual_income": "income",
        "employment": "employment_length",
        "credit_history": "credit_history_length",
        "loan": "loan_amount",
        "debt_to_income_ratio": "debt_to_income",
        "dti": "debt_to_income",
        "num_loans": "existing_loans",
        "number_of_loans": "existing_loans",
        "late_payment_history": "late_payments",
    }
    
    for old, new in column_mapping.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    
    if len(df) == 0:
        return False, "Dataset is empty."
    
    return True, f"Valid. {len(df)} applicants found."
