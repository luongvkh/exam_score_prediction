import streamlit as st
import pandas as pd
import joblib

st.title("🔮 Machine Learning Vorhersage")
st.markdown("Prüfungsnote vorhersagen")
st.markdown("---")


@st.cache_data
def load_data():
    return pd.read_csv("data/processed/exam_score_prediction_cleaned.csv", index_col=0)


df = load_data()


@st.cache_resource
def load_model():
    return joblib.load("models/exam_score_prediction_model.pkl")


model = load_model()

tab1, tab2 = st.tabs(
    [
        "🧠 Modell Training",
        "🔮 Vorhersage",
    ]
)

with tab1:
    # Zwei Spalten: Training | Vorhersage
    st.markdown("### Modell Training")
    st.markdown(":grey[Modell: Linear Regression]")

    if st.button("Modell trainieren"):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

    # Daten vorbereiten
    # ...

    # Modell trainieren
    # ...

    # Evaluation
    # ...

    # Modell speichern
    # ...

with tab2:
    st.markdown("### Vorhersage")

    # if "trained_model" in st.session_state:
    if True:
        st.write("Geben Sie hier die Daten des Studierenden ein:")
        # Input-Felder (anpassen an eure Features)

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Alter", 16, 100, 25)
            study_hours = st.number_input("wöchentliche Lernzeit in Stunden(h)", 0, 100)
            sleep_hours = st.number_input("Schlaf pro Tag in Stunden(h)", 0, 24)
            facility_rating = st.selectbox(
                "Ausstattung der Bildungseinrichtung",
                ["schlecht", "durchschnittlich", "gut"],
                key="facility_rating",
            )

        with col2:
            gender = st.selectbox(
                "Geschlecht", ["weiblich", "männlich", "divers"], key="gender"
            )
            class_attendance = st.number_input(
                "Teilnahme an Lehrveranstaltungen in %", 0, 100
            )
            sleep_quality = st.selectbox(
                "Schlafqualität",
                ["schlecht", "durchschnittlich", "gut"],
                key="sleep_quality",
            )
            exam_difficulty = st.selectbox(
                "Schwierigkeitsstufe der Prüfung",
                ["einfach", "mittel", "schwer"],
                key="exam_difficulty",
            )

        with col3:
            course = st.selectbox(
                "Kurs", ["Diploma", "B.Sc.", "B.Eng.", "B.A."], key="course"
            )
            study_method = st.selectbox(
                "Lernmethode",
                ["coaching", "online Videos", "gemischt", "self-study", "group study"],
                key="study_method",
            )
            
            internet_access = st.checkbox("Internetzugang")

        if st.button("🔮 Vorhersage starten"):
            # Hier eure Vorhersage-Logik
            st.info("Implementiere deine Vorhersage hier!")
        else:
            st.info('⚠️ Bitte erst ein Modell unter "🧠 Modell Training" trainieren!')
