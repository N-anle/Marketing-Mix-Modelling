import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import shap

st.set_page_config(page_title="Marketing Mix Model", layout="wide")

st.title("Marketing Mix Modeling Dashboard")

# =========================
# Load Data
# =========================
@st.cache_data
def load_df():
    df = pd.read_csv("../data/processed/ml_df.csv")
    df["calendar_week"] = pd.to_datetime(df["calendar_week"])
    return df

df = load_df()

# =========================
# Load Model + Features
# =========================
rf_model = joblib.load("../models/MMM Random forest model.pkl")
feature_names = joblib.load("../models/feature_names.pkl")

# =========================
# KPIs
# =========================
st.subheader("Key Business Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Sales", f"{df['sales'].sum():,.0f}")
col2.metric("Avg Weekly Sales", f"{df['sales'].mean():,.0f}")
col3.metric("Max Weekly Sales", f"{df['sales'].max():,.0f}")

# =========================
# Actual Sales Over Time
# =========================
st.subheader("Actual Sales Over Time")

chart = alt.Chart(df).mark_line().encode(
    x="calendar_week",
    y="sales",
    tooltip=["calendar_week", "sales"]
).interactive()

st.altair_chart(chart, use_container_width=True)

# =========================
# Historical Validation
# =========================
st.subheader("Historical Prediction vs Actual")

selected_week = st.selectbox(
    "Select Week",
    df["calendar_week"].sort_values().unique()
)

week_row = df[df["calendar_week"] == selected_week]

if not week_row.empty:
    X_week = week_row[feature_names]
    actual = week_row["sales"].values[0]
    predicted = rf_model.predict(X_week)[0]

    colA, colB = st.columns(2)
    colA.metric("Actual Sales", f"{actual:,.0f}")
    colB.metric("Model Prediction", f"{predicted:,.0f}")

# =========================
# WHAT-IF SIMULATION
# =========================
st.subheader("Media What-If Simulator")

base_row = df.iloc[-1].copy()
input_data = {}

adstock_features = [f for f in feature_names if f.endswith("_adstock")]

for feature in adstock_features:
    min_val = float(df[feature].min())
    max_val = float(df[feature].max())
    default_val = float(base_row[feature])

    input_data[feature] = st.slider(
        feature.replace("_adstock", "").replace("_", " ").title(),
        min_value=min_val,
        max_value=max_val,
        value=default_val
    )

# Keep controls fixed
for col in feature_names:
    if col not in input_data:
        input_data[col] = base_row[col]

input_df = pd.DataFrame([input_data])[feature_names]

sim_prediction = rf_model.predict(input_df)[0]

st.metric("Simulated Predicted Sales", f"{sim_prediction:,.0f}")

# =========================
# CHANNEL CONTRIBUTION (Permutation Importance Proxy)
# =========================
st.subheader("Channel Contribution Estimate")

importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

channel_importance = importance_df[
    importance_df["feature"].str.endswith("_adstock")
].sort_values("importance", ascending=False)

bar_chart = alt.Chart(channel_importance).mark_bar().encode(
    x="importance",
    y=alt.Y("feature", sort="-x")
)

st.altair_chart(bar_chart, use_container_width=True)

# =========================
# SHAP EXPLANATION
# =========================
st.subheader("Model Explanation (SHAP)")

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(input_df)

shap_df = pd.DataFrame({
    "feature": feature_names,
    "shap_value": shap_values[0]
}).sort_values("shap_value", key=abs, ascending=False)

shap_chart = alt.Chart(shap_df.head(15)).mark_bar().encode(
    x="shap_value",
    y=alt.Y("feature", sort="-x")
)

st.altair_chart(shap_chart, use_container_width=True)