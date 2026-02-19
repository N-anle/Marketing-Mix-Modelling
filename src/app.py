import streamlit as st
import pandas as pd
import altair as alt
import joblib
import numpy as np

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Marketing Mix Dashboard",
    layout="wide"
)

st.title("Marketing Mix Dashboard")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_df():
    df = pd.read_csv("../data/raw/Sample Media Spend Data.csv")
    df.columns = map(str.lower, df.columns)
    df['calendar_week'] = pd.to_datetime(df['calendar_week'])
    return df

df = load_df()

# -------------------------------
# Load ML Model
# -------------------------------
rf_model = joblib.load("../models/MMM Random forest model.pkl")
feature_names = joblib.load("../models/feature_names.pkl")  # ensure it matches training

# -------------------------------
# About the dataset
# -------------------------------
with st.expander("About the dataset"):
    st.header("About The Dataset")
    st.write(
        "This dataset contains marketing metrics for company X, including impressions, views, and weekly sales."
    )
    st.dataframe(df)

# -------------------------------
# KPIs
# -------------------------------
st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"{df['sales'].sum():,.0f}")
col2.metric("Max Weekly Sales", f"{df['sales'].max():,.0f}")
top_channel = df[['paid_views','organic_views','google_impressions',
                  'email_impressions','facebook_impressions','affiliate_impressions',
                  'overall_views']].sum().idxmax()
col3.metric("Top Channel (by total)", top_channel.replace('_',' ').title())

# -------------------------------
# Metric selection for visualization
# -------------------------------
metrics = [col for col in df.columns if col not in ['division','calendar_week','sales']]
selected_metrics = st.multiselect(
    "Select one or more metrics to visualize with Sales",
    options=metrics,
    default=['google_impressions','facebook_impressions']
)

# -------------------------------
# Time Series Visualization
# -------------------------------
if selected_metrics:
    st.subheader("Time Series: Sales vs Selected Metrics")
    ts_df = df[['calendar_week','sales'] + selected_metrics].set_index('calendar_week')
    st.line_chart(ts_df)

# -------------------------------
# Scatter Plots: Metric vs Sales
# -------------------------------
st.subheader("Scatter Plots: Metrics vs Sales")
for metric in selected_metrics:
    st.write(f"### {metric.replace('_',' ').title()} vs Sales")
    scatter_chart = alt.Chart(df).mark_circle(size=60, opacity=0.6).encode(
        x=metric,
        y='sales',
        tooltip=['calendar_week', metric, 'sales']
    ).interactive()
    st.altair_chart(scatter_chart, use_container_width=True)

# -------------------------------
# Division Analysis
# -------------------------------
with st.expander("Sales by Division"):
    division_df = df.groupby('division', as_index=False)['sales'].sum()
    division_chart = alt.Chart(division_df).mark_bar(color='teal').encode(
        x='division',
        y='sales',
        tooltip=['division','sales']
    )
    st.altair_chart(division_chart, use_container_width=True)

# -------------------------------
# Month Analysis
# -------------------------------
with st.expander("Sales by Month"):
    df['month'] = df['calendar_week'].dt.month
    month_df = df.groupby('month', as_index=False)['sales'].sum()
    month_chart = alt.Chart(month_df).mark_line(point=True, color='orange').encode(
        x='month',
        y='sales',
        tooltip=['month','sales']
    )
    st.altair_chart(month_chart, use_container_width=True)

# -------------------------------
# ML Prediction Panel
# -------------------------------
st.subheader("Predict Sales with Your Inputs")

# Interactive sliders for main marketing channels
user_inputs = {}
input_cols = ['paid_views','organic_views','google_impressions',
              'email_impressions','facebook_impressions','affiliate_impressions']

for col in input_cols:
    min_val = int(df[col].min())
    max_val = int(df[col].max())
    default_val = int(df[col].mean())
    user_inputs[col] = st.slider(
        f"{col.replace('_',' ').title()}",
        min_value=min_val,
        max_value=max_val,
        value=default_val,
        step=1
    )

# Trend and adstock dummy placeholders (simplified)
user_inputs['trend'] = len(df)  # simple placeholder

# Month dummies
month_num = st.selectbox("Select Month for Prediction", options=list(range(1,13)), index=0)
for m in range(2,13):
    user_inputs[f'month_{m}'] = 1 if month_num == m else 0

# Division dummies
division_sel = st.selectbox("Select Division", options=sorted(df['division'].unique()))
for d in sorted(df['division'].unique()):
    if d == 'A':
        continue  # drop first to match training
    user_inputs[f'division_{d}'] = 1 if division_sel == d else 0

# Build input dataframe
input_df = pd.DataFrame([user_inputs])
# Reorder columns to match feature_names
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# Predict
predicted_sales = rf_model.predict(input_df)[0]
st.metric("Predicted Sales", f"{predicted_sales:,.0f}")

# -------------------------------
# Historical Actual vs Predicted
# -------------------------------
st.subheader("Historical Comparison: Actual vs Predicted")
week_sel = st.selectbox("Select Calendar Week", options=df['calendar_week'])
week_data = df[df['calendar_week'] == week_sel]
if not week_data.empty:
    week_X = week_data[feature_names]
    actual = week_data['sales'].values[0]
    pred = rf_model.predict(week_X)[0]
    st.write(f"Actual Sales: {actual}")
    st.write(f"Predicted Sales: {pred:.0f}")
