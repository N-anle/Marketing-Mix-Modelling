import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import shap

st.set_page_config(page_title="Marketing Mix Model", layout="wide")

st.title("Marketing Mix Modeling Dashboard")

#Data loading
@st.cache_data
def load_df():
    df = pd.read_csv("../data/processed/ml_df.csv")
    df["calendar_week"] = pd.to_datetime(df["calendar_week"])
    return df

df = load_df()


rf_model = joblib.load("../models/MMM Random forest model.pkl")
feature_names = joblib.load("../models/feature_names.pkl")

#Business metrics section
st.subheader("Key Business Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Sales", f"{df['sales'].sum():,.0f}")
col2.metric("Avg Weekly Sales", f"{df['sales'].mean():,.0f}")
col3.metric("Max Weekly Sales", f"{df['sales'].max():,.0f}")


st.subheader("Actual Sales Over Time")

chart = alt.Chart(df).mark_line().encode(
    x="calendar_week",
    y="sales",
    tooltip=["calendar_week", "sales"]
).interactive()

st.altair_chart(chart,width = 'stretch')

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


for col in feature_names:
    if col not in input_data:
        input_data[col] = base_row[col]

input_df = pd.DataFrame([input_data])[feature_names]

sim_prediction = rf_model.predict(input_df)[0]

st.metric("Simulated Predicted Sales", f"{sim_prediction:,.0f}")

#Channel contribution
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

st.altair_chart(bar_chart, width = 'stretch')

st.subheader("Simulated Marketing Mix Share")


sim_shares = []
for feature in channel_importance['feature']:
    current_val = input_df[feature].values[0]
    base_imp = channel_importance.loc[channel_importance['feature'] == feature, 'importance'].values[0]
    
    impact = abs(current_val * base_imp)
    sim_shares.append({"Channel": feature.replace("_adstock", "").title(), "Impact": impact})

sim_share_df = pd.DataFrame(sim_shares)

dynamic_pie = alt.Chart(sim_share_df).mark_arc(innerRadius=50).encode(
    theta=alt.Theta(field="Impact", type="quantitative"),
    color=alt.Color(field="Channel", type="nominal"),
    tooltip=["Channel", "Impact"]
).properties(height=400)

st.altair_chart(dynamic_pie, width = 'stretch')

#SHAP explanation
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

st.altair_chart(shap_chart, width = 'stretch')