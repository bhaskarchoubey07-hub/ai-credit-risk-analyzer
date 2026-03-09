# AI Credit Risk Analyzer

A production-ready banking-style web platform that predicts loan default risk and provides approval recommendations using machine learning.

![Screenshot Placeholder - Run the app to see the dashboard]

## Overview

The AI Credit Risk Analyzer is a fintech application that:

- **Predicts loan default probability** using an ensemble of Logistic Regression and Random Forest
- **Generates credit scores** (300-850) based on income stability, payment history, debt level, and credit history
- **Classifies risk** as Low (0-20%), Medium (20-50%), or High (50-100%)
- **Recommends approval** (Approved / Review Required / Reject)
- **Explains predictions** by showing which factors influenced the decision

## Features

| Feature | Description |
|---------|-------------|
| **Loan Application Form** | Single applicant input (Age, Income, Employment, Loan Amount, DTI, etc.) |
| **Credit Score Generator** | 300-850 score from multiple risk factors |
| **Default Prediction** | ML-based probability (e.g., 0.23 = 23%) |
| **Risk Category** | Low / Medium / High |
| **Approval Recommendation** | Approved / Review Required / Reject |
| **Dashboard** | Risk gauge, Income vs Loan, Debt analysis charts |
| **Batch Analysis** | Upload CSV for multiple applicants |
| **Download Report** | Text report with applicant details and decision |

## Installation

### Prerequisites

- Python 3.11+
- pip

### Steps

1. **Clone or navigate to the project:**

   ```bash
   cd credit-risk-analyzer
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   # source venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Generate the dataset (if not present):**

   ```bash
   python generate_dataset.py
   ```

   This creates `dataset/loan_dataset.csv` with 2000 samples.

5. **Run the application:**

   ```bash
   streamlit run app.py
   ```

   The model trains automatically on first run if `models/credit_model.pkl` does not exist.

## Project Structure

```
credit-risk-analyzer/
├── app.py              # Main Streamlit application
├── model.py            # ML model training (Logistic Regression + Random Forest)
├── predictor.py        # Prediction and explainability logic
├── data_processing.py  # Dataset loading and preprocessing
├── utils.py            # Credit score formula and risk classification
├── generate_dataset.py # Dataset generation script
├── requirements.txt
├── README.md
├── dataset/
│   └── loan_dataset.csv
└── models/
    └── credit_model.pkl  # Created on first run
```

## Example Predictions

| Applicant | Credit Score | Default Prob | Risk | Recommendation |
|-----------|--------------|--------------|------|----------------|
| Low risk  | 720          | 0.12         | Low  | Approved       |
| Medium    | 580          | 0.35         | Medium | Review Required |
| High risk | 420          | 0.68         | High | Reject         |

## Dataset Schema

| Column | Description |
|--------|-------------|
| age | Applicant age |
| income | Annual income ($) |
| employment_length | Years employed |
| credit_history_length | Years of credit history |
| loan_amount | Requested loan amount ($) |
| debt_to_income | DTI ratio (0-1) |
| existing_loans | Number of current loans |
| late_payments | Count of late payments |
| default | Target (0=no default, 1=default) |

## Batch Upload

For batch analysis, upload a CSV with columns matching the dataset schema. Alternate names are supported:

- `annual_income` → income  
- `employment` → employment_length  
- `credit_history` → credit_history_length  
- `loan` → loan_amount  
- `dti` / `debt_to_income_ratio` → debt_to_income  
- `num_loans` / `number_of_loans` → existing_loans  
- `late_payment_history` → late_payments  

## Future Improvements

- [ ] Add XGBoost/LightGBM for improved accuracy
- [ ] SHAP-based explainability
- [ ] User authentication and audit logging
- [ ] API endpoints for integration
- [ ] A/B testing for model versions
- [ ] Real-time monitoring dashboard

## License

MIT
