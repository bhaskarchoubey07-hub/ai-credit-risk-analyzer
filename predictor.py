"""
AI Credit Risk Analyzer - Prediction Logic
Handles default probability prediction and explainability.
"""

import pandas as pd
import numpy as np

from model import get_or_train_model
from utils import (
    calculate_credit_score,
    get_risk_category,
    get_approval_recommendation,
    get_risk_color,
)
from data_processing import FEATURE_COLUMNS


def predict_default_probability(X: pd.DataFrame) -> np.ndarray:
    """
    Predict default probability for each row.
    Uses ensemble of Logistic Regression and Random Forest.
    """
    model_package = get_or_train_model()
    lr = model_package["lr_model"]
    rf = model_package["rf_model"]
    scaler = model_package.get("scaler")
    
    X_aligned = X[FEATURE_COLUMNS].copy()
    X_aligned = X_aligned.fillna(X_aligned.median())
    
    # LR uses scaled features, RF uses raw
    if scaler is not None:
        X_scaled = scaler.transform(X_aligned)
        lr_proba = lr.predict_proba(X_scaled)[:, 1]
    else:
        lr_proba = lr.predict_proba(X_aligned)[:, 1]
    rf_proba = rf.predict_proba(X_aligned)[:, 1]
    ensemble_proba = (lr_proba + rf_proba) / 2
    
    return ensemble_proba


def get_prediction_explanation(
    model_package: dict,
    applicant: dict,
    default_prob: float,
) -> list[dict]:
    """
    Explain which factors influenced the credit risk decision.
    Returns list of {factor, impact, direction} dicts.
    """
    lr_coef = model_package.get("lr_coefficients")
    rf_imp = model_package.get("rf_feature_importances")
    feature_names = model_package.get("feature_columns", FEATURE_COLUMNS)
    
    # Human-readable factor names
    display_names = {
        "age": "Age",
        "income": "Annual Income",
        "employment_length": "Employment Length",
        "credit_history_length": "Credit History Length",
        "loan_amount": "Loan Amount",
        "debt_to_income": "Debt-to-Income Ratio",
        "existing_loans": "Number of Existing Loans",
        "late_payments": "Late Payment History",
    }
    
    explanations = []
    
    # Combine LR coefficients and RF importance (normalized)
    if lr_coef is not None and rf_imp is not None:
        lr_norm = np.abs(lr_coef) / (np.abs(lr_coef).sum() + 1e-9)
        rf_norm = rf_imp / (rf_imp.sum() + 1e-9)
        combined = 0.5 * lr_norm + 0.5 * rf_norm
    elif rf_imp is not None:
        combined = rf_imp / (rf_imp.sum() + 1e-9)
    else:
        return []
    
    # Determine direction (positive coef = increases risk)
    for i, name in enumerate(feature_names):
        if i >= len(combined):
            break
        impact = float(combined[i])
        direction = "increases" if lr_coef is not None and lr_coef[i] > 0 else "decreases"
        explanations.append({
            "factor": display_names.get(name, name),
            "impact": impact,
            "direction": direction,
        })
    
    # Sort by impact descending
    explanations.sort(key=lambda x: x["impact"], reverse=True)
    return explanations


def predict_single_applicant(applicant: dict) -> dict:
    """
    Full prediction pipeline for single applicant.
    Returns dict with credit_score, default_probability, risk_category,
    approval_recommendation, risk_color, and explanation.
    """
    import pandas as pd
    from data_processing import prepare_single_applicant
    
    X = prepare_single_applicant(applicant)
    model_package = get_or_train_model()
    
    default_prob = float(predict_default_probability(X)[0])
    risk_category = get_risk_category(default_prob)
    approval = get_approval_recommendation(risk_category)
    risk_color = get_risk_color(risk_category)
    
    credit_score = calculate_credit_score(
        income=applicant.get("income", 0),
        employment_length=applicant.get("employment_length", 0),
        credit_history_length=applicant.get("credit_history_length", 0),
        debt_to_income=applicant.get("debt_to_income", 0),
        late_payments=applicant.get("late_payments", 0),
        existing_loans=applicant.get("existing_loans", 0),
        loan_amount=applicant.get("loan_amount", 0),
    )
    
    explanation = get_prediction_explanation(model_package, applicant, default_prob)
    
    return {
        "credit_score": credit_score,
        "default_probability": default_prob,
        "risk_category": risk_category,
        "approval_recommendation": approval,
        "risk_color": risk_color,
        "explanation": explanation,
        "applicant": applicant,
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict for batch of applicants. Adds columns:
    default_probability, credit_score, risk_category, approval_recommendation
    """
    from data_processing import preprocess_for_training
    
    X, _ = preprocess_for_training(df)
    
    probs = predict_default_probability(X)
    
    result = df.copy()
    result["default_probability"] = probs
    result["risk_category"] = [get_risk_category(p) for p in probs]
    result["approval_recommendation"] = [
        get_approval_recommendation(get_risk_category(p)) for p in probs
    ]
    
    # Credit score per row (handle alternate column names)
    def _val(row, *keys):
        for k in keys:
            if k in row.index and pd.notna(row.get(k)):
                return row[k]
        return 0

    scores = []
    for _, row in result.iterrows():
        score = calculate_credit_score(
            income=float(_val(row, "income", "annual_income")),
            employment_length=float(_val(row, "employment_length", "employment")),
            credit_history_length=float(_val(row, "credit_history_length", "credit_history")),
            debt_to_income=float(_val(row, "debt_to_income", "dti", "debt_to_income_ratio")),
            late_payments=int(_val(row, "late_payments", "late_payment_history")),
            existing_loans=int(_val(row, "existing_loans", "num_loans", "number_of_loans")),
            loan_amount=float(_val(row, "loan_amount", "loan")),
        )
        scores.append(score)
    result["credit_score"] = scores
    
    return result
