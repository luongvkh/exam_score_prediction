import streamlit as st
import pandas as pd

st.title("🧼 Datenbereinigung")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/exam_score_prediction_cleaned.csv", index_col=0)

df = load_data()

col1, col2 = st.columns([1,1])

col1.metric("Zeilen", len(df))
col2.metric("Features", len(df.columns))