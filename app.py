"""
AI Credit Risk Analyzer - Main Application
Banking-style web platform for loan default prediction and approval recommendations.
Run: streamlit run app.py
"""

import io
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from data_processing import (
    load_dataset,
    preprocess_for_training,
    prepare_single_applicant,
    validate_batch_upload,
    FEATURE_COLUMNS,
)
from predictor import predict_single_applicant, predict_batch
from utils import get_risk_color, get_risk_category, get_approval_recommendation


# Page config - must be first Streamlit command
st.set_page_config(
    page_title="AI Credit Risk Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for fintech dashboard styling
st.markdown("""
<style>
    /* Main container */
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.5rem 0;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
    }
    .metric-card h3 { font-size: 0.85rem; color: #94a3b8; margin: 0; }
    .metric-card .value { font-size: 1.75rem; font-weight: 700; margin-top: 0.25rem; }
    
    /* Risk badges */
    .risk-low { color: #22c55e; }
    .risk-medium { color: #eab308; }
    .risk-high { color: #ef4444; }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Header */
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.5rem;
    }
    .sub-header { color: #94a3b8; font-size: 1rem; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)


def render_risk_gauge(probability: float, risk_color: str) -> go.Figure:
    """Create risk gauge chart (0-100% scale)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 32}},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": risk_color, "thickness": 0.75},
            "bgcolor": "rgba(30, 41, 59, 0.5)",
            "borderwidth": 2,
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, 20], "color": "rgba(34, 197, 94, 0.3)"},
                {"range": [20, 50], "color": "rgba(234, 179, 8, 0.3)"},
                {"range": [50, 100], "color": "rgba(239, 68, 68, 0.3)"},
            ],
            "threshold": {
                "line": {"color": risk_color, "width": 4},
                "thickness": 0.75,
                "value": probability * 100,
            },
        },
        title={"text": "Default Risk Level", "font": {"size": 16}},
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc", "family": "Inter, sans-serif"},
    )
    return fig


def render_income_loan_ratio(income: float, loan_amount: float) -> go.Figure:
    """Income vs Loan ratio bar chart."""
    ratio = loan_amount / income if income > 0 else 0
    fig = go.Figure(go.Bar(
        x=["Annual Income", "Loan Amount"],
        y=[income, loan_amount],
        marker_color=["#3b82f6", "#8b5cf6"],
        text=[f"${income:,.0f}", f"${loan_amount:,.0f}"],
        textposition="outside",
        textfont={"size": 14},
    ))
    fig.update_layout(
        title="Income vs Loan Amount",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30, 41, 59, 0.3)",
        font={"color": "#f8fafc"},
        xaxis={"gridcolor": "#334155"},
        yaxis={"gridcolor": "#334155"},
        showlegend=False,
    )
    fig.add_annotation(
        text=f"Ratio: {ratio:.2f}x",
        xref="paper", yref="paper",
        x=0.5, y=1.08, showarrow=False,
        font={"size": 14, "color": "#94a3b8"},
    )
    return fig


def render_debt_analysis(applicant: dict) -> go.Figure:
    """Debt analysis pie/donut chart."""
    dti = applicant.get("debt_to_income", 0)
    existing_loans = applicant.get("existing_loans", 0)
    late_payments = applicant.get("late_payments", 0)

    labels = ["DTI Ratio", "Existing Loans", "Late Payments"]
    values = [
        min(dti * 100, 100),
        min(existing_loans * 15, 100),
        min(late_payments * 10, 100),
    ]
    colors = ["#ef4444", "#eab308", "#22c55e"]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker={"colors": colors},
        textinfo="label+percent",
        textposition="outside",
    ))
    fig.update_layout(
        title="Debt & Payment Factors",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"},
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    return fig


def render_single_applicant_form() -> dict | None:
    """Render loan application form and return applicant dict or None."""
    st.markdown("### 📋 Loan Application Form")
    st.markdown("Enter applicant details below.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=90, value=35, step=1)
        income = st.number_input(
            "Annual Income ($)", min_value=0, value=65000, step=1000
        )
        employment_length = st.number_input(
            "Employment Length (years)", min_value=0, max_value=50, value=5
        )
    with col2:
        credit_history_length = st.number_input(
            "Credit History Length (years)", min_value=0, max_value=50, value=8
        )
        loan_amount = st.number_input(
            "Loan Amount ($)", min_value=0, value=25000, step=1000
        )
        debt_to_income = st.slider(
            "Debt-to-Income Ratio", 0.0, 1.0, 0.35, 0.01
        )
    with col3:
        existing_loans = st.number_input(
            "Number of Existing Loans", min_value=0, max_value=20, value=2
        )
        late_payments = st.number_input(
            "Late Payment History (count)", min_value=0, max_value=50, value=1
        )

    if st.button("🔍 Analyze Credit Risk", type="primary", use_container_width=True):
        return {
            "age": age,
            "income": income,
            "employment_length": employment_length,
            "credit_history_length": credit_history_length,
            "loan_amount": loan_amount,
            "debt_to_income": debt_to_income,
            "existing_loans": existing_loans,
            "late_payments": late_payments,
        }
    return None


def render_batch_upload() -> pd.DataFrame | None:
    """Render batch CSV upload and return validated DataFrame or None."""
    st.markdown("### 📁 Batch Analysis")
    st.markdown("Upload a CSV file with multiple loan applicants.")

    uploaded = st.file_uploader(
        "Choose CSV file",
        type=["csv"],
        help="Required columns: age, income, employment_length, credit_history_length, loan_amount, debt_to_income, existing_loans, late_payments",
    )

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            is_valid, msg = validate_batch_upload(df)
            if is_valid:
                st.success(msg)
                return df
            else:
                st.error(msg)
        except Exception as e:
            st.error(f"Error reading file: {e}")
    return None


def render_dashboard(result: dict) -> None:
    """Render dashboard with metrics, charts, and explain section."""
    credit_score = result["credit_score"]
    default_prob = result["default_probability"]
    risk_category = result["risk_category"]
    approval = result["approval_recommendation"]
    risk_color = result["risk_color"]
    applicant = result["applicant"]
    explanation = result.get("explanation", [])

    # Metric cards
    st.markdown("### 📊 Risk Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Credit Score", credit_score, delta=None)
        st.markdown(f'<p style="font-size:0.8rem;color:#94a3b8;">300-850 scale</p>', unsafe_allow_html=True)
    with col2:
        st.metric("Default Probability", f"{default_prob:.1%}", delta=None)
    with col3:
        st.markdown(f'<div class="metric-card"><h3>Risk Category</h3><p class="value" style="color:{risk_color};">{risk_category}</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><h3>Recommendation</h3><p class="value" style="color:{risk_color};">{approval}</p></div>', unsafe_allow_html=True)

    # Charts row
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(render_risk_gauge(default_prob, risk_color), use_container_width=True)
    with c2:
        st.plotly_chart(
            render_income_loan_ratio(applicant.get("income", 0), applicant.get("loan_amount", 0)),
            use_container_width=True,
        )
    with c3:
        st.plotly_chart(render_debt_analysis(applicant), use_container_width=True)

    # Explain Prediction
    st.markdown("---")
    st.markdown("### 🔍 Explain Prediction")
    st.markdown("Factors that most influenced this credit risk decision:")

    if explanation:
        for i, exp in enumerate(explanation[:5], 1):
            pct = exp["impact"] * 100
            direction = exp["direction"]
            st.markdown(f"**{i}. {exp['factor']}** — *{direction}* risk (impact: {pct:.1f}%)")
    else:
        st.info("Explanation data not available for this prediction.")

    # Download report
    st.markdown("---")
    report = generate_report(result)
    st.download_button(
        label="📥 Download Risk Report",
        data=report,
        file_name=f"credit_risk_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        use_container_width=True,
    )


def generate_report(result: dict) -> str:
    """Generate downloadable text report."""
    lines = [
        "=" * 50,
        "AI CREDIT RISK ANALYZER - RISK REPORT",
        "=" * 50,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "APPLICANT DETAILS",
        "-" * 30,
    ]
    for k, v in result.get("applicant", {}).items():
        lines.append(f"  {k}: {v}")
    lines.extend([
        "",
        "ANALYSIS RESULTS",
        "-" * 30,
        f"  Credit Score: {result['credit_score']}",
        f"  Default Probability: {result['default_probability']:.2%}",
        f"  Risk Category: {result['risk_category']}",
        f"  Approval Decision: {result['approval_recommendation']}",
        "",
        "=" * 50,
    ])
    return "\n".join(lines)


def main():
    # Sidebar
    st.sidebar.markdown("# 📊 AI Credit Risk Analyzer")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigation",
        ["Single Applicant", "Batch Analysis"],
        index=0,
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Risk Categories")
    st.sidebar.markdown("- **Low** (0-20%): Approved")
    st.sidebar.markdown("- **Medium** (20-50%): Review Required")
    st.sidebar.markdown("- **High** (50-100%): Reject")

    # Main content
    st.markdown('<p class="main-header">AI Credit Risk Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Banking-style platform for loan default prediction and approval recommendations</p>', unsafe_allow_html=True)

    if page == "Single Applicant":
        applicant = render_single_applicant_form()
        if applicant is not None:
            with st.spinner("Analyzing credit risk..."):
                result = predict_single_applicant(applicant)
            render_dashboard(result)

    else:  # Batch Analysis
        df = render_batch_upload()
        if df is not None and st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
            with st.spinner("Processing batch..."):
                result_df = predict_batch(df)

            st.markdown("### 📊 Batch Results")
            st.dataframe(result_df, use_container_width=True)

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Applicants", len(result_df))
            with col2:
                approved = (result_df["approval_recommendation"] == "Approved").sum()
                st.metric("Recommended Approvals", approved)
            with col3:
                rejected = (result_df["approval_recommendation"] == "Reject").sum()
                st.metric("Recommended Rejections", rejected)

            # Download batch CSV
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Batch Report (CSV)",
                data=csv,
                file_name=f"batch_credit_risk_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
