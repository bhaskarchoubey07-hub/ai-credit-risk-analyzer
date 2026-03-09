"""
AI Credit Risk Analyzer - Utility Functions
Handles credit score calculation and risk classification.
"""


def calculate_credit_score(
    income: float,
    employment_length: float,
    credit_history_length: float,
    debt_to_income: float,
    late_payments: int,
    existing_loans: int,
    loan_amount: float,
) -> int:
    """
    Generate credit score (300-850) based on multiple factors.
    Higher score = lower risk.
    
    Factors and weights:
    - Income stability: 25%
    - Payment history (late payments): 25%
    - Debt level (DTI, existing loans): 25%
    - Credit history length: 25%
    """
    base_score = 300
    max_additive = 550
    
    # Income stability (0-1): Higher income relative to loan = better
    income_ratio = min(income / (loan_amount + 1), 5) / 5
    income_score = income_ratio * 0.25
    
    # Payment history (0-1): Fewer late payments = better
    late_penalty = min(late_payments / 10, 1) * 0.25
    payment_score = (1 - late_penalty) * 0.25
    
    # Debt level (0-1): Lower DTI and fewer loans = better
    dti_penalty = min(debt_to_income / 0.6, 1) * 0.125
    loan_penalty = min(existing_loans / 8, 1) * 0.125
    debt_score = (1 - dti_penalty - loan_penalty) * 0.25
    
    # Credit history (0-1): Longer history = better
    history_score = min(credit_history_length / 20, 1) * 0.25
    employment_bonus = min(employment_length / 15, 1) * 0.05
    
    total_factor = income_score + payment_score + debt_score + history_score + employment_bonus
    score = int(base_score + total_factor * max_additive)
    
    return min(max(score, 300), 850)


def get_risk_category(default_probability: float) -> str:
    """
    Classify risk based on default probability.
    0-20%: Low Risk
    20-50%: Medium Risk
    50-100%: High Risk
    """
    if default_probability < 0.20:
        return "Low Risk"
    elif default_probability < 0.50:
        return "Medium Risk"
    else:
        return "High Risk"


def get_approval_recommendation(risk_category: str) -> str:
    """
    Determine loan approval based on risk category.
    Low Risk → Approved
    Medium Risk → Review Required
    High Risk → Reject
    """
    if risk_category == "Low Risk":
        return "Approved"
    elif risk_category == "Medium Risk":
        return "Review Required"
    else:
        return "Reject"


def get_risk_color(risk_category: str) -> str:
    """Return color code for risk category."""
    colors = {
        "Low Risk": "#22c55e",      # Green
        "Medium Risk": "#eab308",   # Yellow
        "High Risk": "#ef4444",     # Red
    }
    return colors.get(risk_category, "#6b7280")
