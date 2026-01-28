import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


@st.cache_data
def load_data():
    return pd.read_csv("data/processed/exam_score_prediction_cleaned.csv", index_col=0)


def main():
    st.title("🎓 Exam Score Prediction")
    st.markdown("### Datenanalyse und ML-Vorhersage")
    st.markdown("---")

    st.title("Herzlich Willkommen! 😊")
    st.markdown(
        "Diese App analysiert das [Exam Score Prediction Dataset](https://www.kaggle.com/datasets/kundanbedmutha/exam-score-prediction-dataset) und macht Vorhersagen."
    )

    st.markdown("### Inhalte")
    st.markdown(
        """
                1. 🔦 Daten-Exploration
                2. 🔍 Visualisierung
                3. 🔮 Machine Learning Vorhersage
                """
    )
    st.markdown("---")

    # Daten laden
    df = load_data()
    filtered_df = df

    # Main: Anzeige
    st.markdown("### Informationen zum Datensatz")

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Studierende", len(filtered_df))
    col2.metric("Ø Alter", f"{filtered_df['age'].mean():.1f}")
    col3.metric("Ø Exam Score", f"{filtered_df['exam_score'].mean():.1f}")

    # Daten anzeigen
    with st.expander("Zeige Rohdaten"):
        st.dataframe(filtered_df)

    st.markdown("---")
    # Häufigkeitsverteilungen
    st.markdown("### Häufigkeitsverteilungen")
    feature = st.selectbox(
        "Wähle ein Feature:",
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
    # Check ob Boolean oder Categorial
    if filtered_df[feature].dtype == "bool" or filtered_df[feature].nunique() <= 5:
        # Für Boolean/Categorial Features: Bar Chart statt Histogram
        value_counts = filtered_df[feature].value_counts()
        ax.bar(
            value_counts.index.astype(str),
            value_counts.values,
            edgecolor="black",
            color="skyblue",
        )
        ax.set_xlabel(feature)
        ax.set_ylabel("Häufigkeit")
        ax.set_title(f"Verteilung: {feature}")
    else:
        # Für numerische Features: Histogram
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
