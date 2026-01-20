import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/exam_score_prediction_cleaned.csv", index_col=0)

def main():
    # Titel
    st.title("🎓 Exam Score Prediction")
    st.markdown("### Datenanalyse und ML-Vorhersage")
    st.markdown("---")

    st.markdown("## Herzlich Willkommen! 😊")
    st.markdown(
        "Diese App analysiert das [Exam Score Prediction Dataset](https://www.kaggle.com/datasets/kundanbedmutha/exam-score-prediction-dataset) und macht Vorhersagen."
    )

    st.markdown("### Inhalte")
    st.markdown("""
                1. 🧼 Datenbereinigung
                2. 🔍 Visualisierung
                3. 🔮 Machine Learning Vorhersage
                """)
    st.markdown("---")

    # Daten laden
    df = load_data()
    filtered_df = df

    # Sidebar: Filter
    # st.sidebar.title("Filter ⚙️")
    # age_range = st.sidebar.slider(
    #     "Exam Score:",
    #     int(df["exam_score"].min()),
    #     int(df["exam_score"].max()),
    #     (30, 70),
    # )

    # show_condition = st.sidebar.radio("Zeige:", ["Alle", "Nur < 50", "Nur > 50"])

    # Daten filtern
    # filtered_df = df[
    #     (df["exam_score"] >= age_range[0]) & (df["exam_score"] <= age_range[1])
    # ]

    # if show_condition == "Nur < 50":
    #     filtered_df = filtered_df[filtered_df["condition"] == 1]
    # elif show_condition == "Nur > 50":
    #     filtered_df = filtered_df[filtered_df["condition"] == 0]

    # Main: Anzeige
    st.subheader("Informationen zum Datensatz")
    st.write(f"Zeige **{len(filtered_df)}** von {len(df)} Studierenden")

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Studierende", len(filtered_df))
    col2.metric("Ø Alter", f"{filtered_df['age'].mean():.1f}")
    col3.metric("Ø Exam Score", f"{filtered_df['exam_score'].mean():.0f}")

    # Daten anzeigen
    if st.checkbox("Zeige Rohdaten"):
        st.dataframe(filtered_df)

    # Visualisierung
    st.subheader("Visualisierung")
    feature = st.selectbox(
        "Feature wählen:",
        [
            "age",
            "gender",
            "course",
            "study_hours",
            "class_attendance",
            "internet_access",
            "sleep_hours",
            "sleep_quality",
            "study_method",
            "facility_rating",
            "exam_difficulty",
            "exam_score",
        ],
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(filtered_df[feature], bins=20, edgecolor="black", color="skyblue")
    ax.set_xlabel(feature)
    ax.set_ylabel("Häufigkeit")
    ax.set_title(f"Verteilung: {feature}")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Footer
    st.markdown("---")
    st.markdown(":grey[DataPy WiSe25/26 - Exam Score Prediction Project]")


if __name__ == "__main__":
    main()
