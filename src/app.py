import streamlit as st
import pandas as pd
import seaborn as sns


st.set_page_config(
    page_title="Marketing Mix Dashboard",
    layout="wide"
)

st.title("Marketing Mix Dashboard")

@st.cache_data
def load_df():
    return pd.read_csv(f"../data/raw/Sample Media Spend Data.csv")

#about the dataset 
with st.expander('About The dataset'):
    st.header("About The dataset")
    st.write("This dataset contains about 5,000 rows of marketing data from company x")
    df = load_df()
    st.dataframe(df)

#Data visualizations 
with st.sidebar:
    st.header("Filters", divider = True)
    metrics = [col for col in df.columns if col not in ['Division', 'Calendar_Week', 'Sales']]
    selected_metric = st.selectbox(label="Select metric", options = metrics, format_func = lambda x: x.replace ("_"," ").title())

formatted_metric = selected_metric.replace("_"," ")

st.header(f"Relationship between {formatted_metric} and sales", divider = True)

st.scatter_chart(data = df, x = selected_metric, y = "Sales")
st.line_chart(data = df, x = "Calendar_Week", y = selected_metric)
