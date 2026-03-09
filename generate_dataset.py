"""
Generate realistic loan dataset for AI Credit Risk Analyzer.
Run once to create dataset/loan_dataset.csv
"""
import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 2000

# Generate realistic loan application data
age = np.random.randint(22, 70, n_samples)
income = np.random.exponential(50000, n_samples).astype(int) + 20000
income = np.clip(income, 15000, 250000)
employment_length = np.random.randint(0, 35, n_samples)
credit_history_length = np.random.randint(1, 30, n_samples)
loan_amount = np.random.exponential(25000, n_samples).astype(int) + 5000
loan_amount = np.clip(loan_amount, 1000, 150000)
debt_to_income = np.random.beta(2, 5, n_samples) * 0.6 + 0.1
debt_to_income = np.clip(debt_to_income, 0.1, 0.7)
existing_loans = np.random.poisson(1.5, n_samples)
existing_loans = np.clip(existing_loans, 0, 10)
late_payments = np.random.poisson(0.8, n_samples)
late_payments = np.clip(late_payments, 0, 15)

# Default probability influenced by risk factors
default_prob = (
    0.1 * (1 - age / 70) +
    0.2 * (1 - income / 250000) +
    0.25 * debt_to_income +
    0.2 * (late_payments / 15) +
    0.15 * (existing_loans / 10) +
    0.1 * (1 - credit_history_length / 30)
)
default_prob = np.clip(default_prob + np.random.normal(0, 0.1, n_samples), 0, 1)
default = (default_prob > 0.5).astype(int)

df = pd.DataFrame({
    'age': age,
    'income': income,
    'employment_length': employment_length,
    'credit_history_length': credit_history_length,
    'loan_amount': loan_amount,
    'debt_to_income': debt_to_income,
    'existing_loans': existing_loans,
    'late_payments': late_payments,
    'default': default
})

df.to_csv('dataset/loan_dataset.csv', index=False)
print(f"Generated {len(df)} samples. Default rate: {default.mean():.2%}")
