import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data/df.csv")  # replace with your filename

# Sidebar
st.sidebar.title("Credit Risk Dashboard")
st.sidebar.write("For filtering go to the bottom.")

# Main title
st.title("Credit Risk & Marketing Strategy Simulation for a Lending Product")

# Section 1: Project Summary
st.header("Project Overview")
st.subheader("Problem Statement")
st.write("""
In today’s competitive lending market, it is essential for banks and financial institutions to reduce loan defaults while acquiring reliable customers to drive profitability. Accurately assessing borrower risk remains a challenge, impacting both approval decisions and portfolio health. This project focuses on developing a credit risk prediction model to identify borrowers with a high likelihood of default, segmenting customers based on their financial characteristics to inform targeted marketing strategies, and simulating lending policies to optimise risk and growth trade-offs. By combining risk analysis and customer insights, the project aims to enable data-driven strategies for better loan portfolio management.
""")
st.subheader("Project Goals")
st.write("""
This dashboard showcases the credit risk prediction model and customer segmentation insights for a lending product to optimize approval strategies and business growth.
""")

st.subheader("Dataset Overview")
st.write("**Sample Data:**")
st.dataframe(df.head(10))
st.write("**Total Records:**", df.shape[0])
st.write("**Features:**", df.columns.tolist())

# Section 2: Model Performance
st.header("Model Performance")
st.write("**Final Model Accuracy:** 93.25%")
st.write("**Final Model ROC-AUC:** 0.98")
st.image("assets/Confusion_matrix.png", caption="Confusion Matrix")
st.image("assets/roc_curve.png", caption="ROC Curve")

# Section 3: Business Segmentation Analysis
st.header("Business Segmentation")

# Display Segment Summary Table
segment_summary = pd.read_csv("data/segment_summary.csv")
st.dataframe(segment_summary)

# Barplot of avg risk per segment
st.subheader("Average Predicted Risk by Segment")
fig, ax = plt.subplots(figsize=(10,8))
sns.barplot(x='avg_risk', y='business_segment', data=segment_summary.sort_values('avg_risk', ascending=False), ax=ax)
ax.set_xlabel("Average Predicted Risk")
ax.set_ylabel("Business Segment")
st.pyplot(fig)

# Section 4: Recommendations
st.header("Recommendations")
st.write("""
- High-risk segments (>0.30 avg risk) require strict evaluation or rejection.
- Low-risk segments (<0.10 avg risk) can be targeted for marketing and instant approvals.
- Medium risk segments may require tailored strategies or manual underwriting.
""")

# 🔷 INTERACTIVE FILTERS & KPI METRICS 🔷
st.sidebar.subheader("Filter by Income Group")
income_filter = st.sidebar.selectbox("Select Income Group", options=df['income_segment'].unique())

st.sidebar.subheader("Filter by Credit Band")
credit_filter = st.sidebar.selectbox("Select Credit Band", options=df['credit_segment'].unique())

st.sidebar.subheader("Set Risk Threshold")
risk_threshold = st.sidebar.slider("Select Risk Threshold", 0.0, 1.0, 0.3)

# Apply filters
filtered_df = df[(df['income_segment'] == income_filter) & (df['credit_segment'] == credit_filter)]
high_risk_df = filtered_df[filtered_df['predicted_risk'] > risk_threshold]

# Display filtered data insights
st.write(f"## Insights for Income Group: {income_filter}, Credit Band: {credit_filter}")

# ➡️ KPI Metrics
st.metric(label="Total Customers (Filtered)", value=filtered_df.shape[0])
st.metric(label="High-Risk Customers", value=high_risk_df.shape[0])
st.metric(label="Average Risk (Filtered)", value=f"{filtered_df['predicted_risk'].mean():.2f}")

# ➡️ Show top loan purposes within this filtered group
st.write("**Top Loan Purposes:**")
top_purposes = filtered_df['loan_intent'].value_counts().head(5)
st.bar_chart(top_purposes)

# ➡️ Download button for high-risk customers
st.download_button("Download High-Risk Customers CSV", high_risk_df.to_csv(index=False), "high_risk_customers.csv")

# 🔷 INTERACTIVE LOAN PURPOSE ANALYSIS 🔷
st.sidebar.subheader("Analyze Loan Purpose")
purpose_filter = st.sidebar.selectbox("Select Loan Purpose", options=df['loan_intent'].unique())
purpose_df = df[df['loan_intent'] == purpose_filter]
st.write(f"### Average Predicted Risk for {purpose_filter}: {purpose_df['predicted_risk'].mean():.2f}")
