import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff

st.title("🔍 Visualisierung")


@st.cache_data
def load_data():
    return pd.read_csv("data/processed/exam_score_prediction_cleaned.csv", index_col=0)


df = load_data()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Histogramm", "Scatter Plot", "Violinen Plots", "Heatmap"]
)

with tab1:
    st.title("Interaktives Histogramm")
    st.markdown(":grey[Einfluss verschiedener Werte einer kategorialen Spalte auf den Exam Score]")
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    col1, col2 = st.columns([1, 0.25])
    with col1:
        selected_cat = st.selectbox(
            "Wähle eine Kategorie für den Vergleich:",
            options=categorical_columns,
            index=3,
        )
    with col2:
        stack_mode = st.checkbox("Balken stapeln (stacked)", value=False)

    if selected_cat:
        categories = df[selected_cat].unique()

        hist_data = []
        group_labels = []

        for cat in categories:
            subset = df[df[selected_cat] == cat]["exam_score"].dropna()
            if not subset.empty:
                hist_data.append(subset)
                group_labels.append(str(cat))

        if hist_data:
            fig = ff.create_distplot(
                hist_data, group_labels, bin_size=0.5, show_hist=True, show_rug=False
            )

            barmode = "stack" if stack_mode else "overlay"

            fig.update_layout(
                barmode=barmode,
                title_text=f"Einfluss von {selected_cat} auf den Exam Score",
                xaxis_title="Exam Score",
                yaxis_title="Dichte",
                legend_title=selected_cat,
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Keine Daten für diese Auswahl verfügbar.")

with tab2:
    st.title("Interaktiver Scatter-Plot")
    st.markdown(":grey[Beziehung zwischen einer numerischen Spalte und des Exam Scores mit Farbkodierung durch eine Kategorie]")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    num_columns = df.select_dtypes(include=["number"]).columns.tolist()
    all_columns = df.columns.tolist()

    col1, col2, col3 = st.columns([1, 1, 0.5])

    with col1:
        x_axis = st.selectbox("X-Achse", options=num_columns, index=1)

    with col2:
        color_by = st.selectbox("Farbkodierung", options=all_columns, index=7)

    with col3:
        point_size = st.slider("Punktgröße", min_value=1, max_value=10, value=3)

    fig = px.scatter(
        df,
        x=x_axis,
        y="exam_score",
        color=color_by,
        hover_name=color_by,
        opacity=0.7,
        size_max=10,
        template="plotly_white",
        title=f"Zusammenhang: {x_axis} vs. Exam Score (Farbe: {color_by})",
    )

    fig.update_layout(
        xaxis_title=x_axis.replace("_", " ").title(),
        yaxis_title="Exam Score",
        legend_title=color_by.replace("_", " ").title(),
    )

    fig.update_traces(marker=dict(size=point_size))

    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.title("Interaktive Violin Plots")
    st.markdown(":grey[Vergleich der Häufigkeitsverteilung verschiedener Werte einer kategorialen Spalte]")

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if categorical_columns:
        col1, col2 = st.columns([1, 0.5])
        with col1:
            selected_cat = st.selectbox(
                "Wähle die Kategorie für die X-Achse:",
                options=categorical_columns,
                key="violin_cat_selector",
                index=3,
            )
        with col2:
            violin_opacity = st.slider(
                "Transparenz", min_value=0.0, max_value=1.0, value=0.4, step=0.05
            )

        col1, col2 = st.columns(2)

        with col1:
            fig_side = px.violin(
                df,
                x=selected_cat,
                y="exam_score",
                color=selected_cat,
                box=True,
                points=None,
                title="Gruppen im Vergleich",
            )
            fig_side.update_layout(showlegend=False)
            st.plotly_chart(fig_side, use_container_width=True)

        with col2:
            fig_over = px.violin(
                df,
                y="exam_score",
                color=selected_cat,
                box=True,
                points=None,
                title="Direkter Overlay",
            )

            fig_over.update_layout(violinmode="overlay", hovermode=False)
            fig_over.update_traces(opacity=violin_opacity)

            st.plotly_chart(fig_over, use_container_width=True)

        st.info(
            "**Hinweis:** Wenn die Violinen im rechten Plot fast identisch sind, "
            "hat die Kategorie keinen nennenswerten Einfluss auf den Exam Score."
        )
    else:
        st.error("Keine kategorialen Spalten im Datensatz gefunden.")

with tab4:
    st.title("Korrelations-Heatmap")
    st.markdown(":grey[Korrelation aller Attribute mit dem Exam Score]")
    df_encoded = df.copy()
    cat_cols = df_encoded.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df_encoded[col] = df_encoded[col].astype("category").cat.codes

    correlations = df_encoded.corr()[["exam_score"]].sort_values(
        by="exam_score", ascending=False
    )
    correlations = correlations.drop(index="exam_score").T

    fig = px.imshow(
        correlations,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        range_color=[-1, 1],
        aspect="auto",
        labels=dict(color="Korrelation"),
    )

    fig.update_layout(height=400, xaxis_nticks=36, yaxis_showticklabels=False)

    st.plotly_chart(fig, use_container_width=True)
