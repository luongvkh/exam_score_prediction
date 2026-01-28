import streamlit as st
import pandas as pd
import joblib

st.title("🔮 Machine Learning Vorhersage")


@st.cache_data
def load_data():
    return pd.read_csv("data/processed/exam_score_prediction_cleaned.csv", index_col=0)


df = load_data()


@st.cache_resource
def load_model():
    return joblib.load("models/exam_score_prediction_model.pkl")


model = load_model()

st.markdown("### Exam Score Prediction")
st.markdown(":grey[Modell: Linear Regression]")
st.markdown("")
st.markdown("**Geben Sie hier die Daten des Studierenden ein:**")

# Input-Felder (anpassen an eure Features)
col1, col2, col3 = st.columns(3)
with col1:
    study_hours = st.number_input("Tägliche Lernzeit in Stunden(h)", 0, 24)
    class_attendance = st.number_input("Teilnahme an Lehrveranstaltungen in %", 0, 100)
    sleep_hours = st.number_input("Schlaf pro Tag in Stunden(h)", 0, 24)


if st.button("🔮 Vorhersage starten"):
    # Feature-Vektor erstellen
    input_data = pd.DataFrame(
        {
            "study_hours": [study_hours],
            "class_attendance": [class_attendance],
            "sleep_hours": [sleep_hours],
        }
    )
    # Vorhersage
    prediction = model.predict(input_data)[0]
    if prediction > 100:
        prediction = 100
    # Ergebnis anzeigen
    st.success(f"🪄 Vorhergesagtes Prüfungsergebnis: {prediction:.2f}%!")

st.markdown("---")
st.markdown(":grey[Alle Angaben ohne Gewähr.]")


# ----- Modell hat nur study_hours, class_attendance und sleep_hours -----
# with col1:
#     age = st.number_input("Alter", 16, 100, 25, key="age")
#     study_hours = st.number_input("wöchentliche Lernzeit in Stunden(h)", 0, 100)
#     sleep_hours = st.number_input("Schlaf pro Tag in Stunden(h)", 0, 24)
#     facility_rating = st.selectbox(
#         "Ausstattung der Bildungseinrichtung",
#         ["schlecht", "durchschnittlich", "gut"],
#         key="facility_rating",
#     )
# with col2:
#     gender = st.selectbox(
#         "Geschlecht", ["weiblich", "männlich", "divers"], key="gender"
#     )
#     class_attendance = st.number_input("Teilnahme an Lehrveranstaltungen in %", 0, 100)
#     sleep_quality = st.selectbox(
#         "Schlafqualität",
#         ["schlecht", "durchschnittlich", "gut"],
#         key="sleep_quality",
#     )
#     exam_difficulty = st.selectbox(
#         "Schwierigkeitsstufe der Prüfung",
#         ["einfach", "mittel", "schwer"],
#         key="exam_difficulty",
#     )
# with col3:
#     course = st.selectbox("Kurs", ["Diploma", "B.Sc.", "B.Eng.", "B.A."], key="course")
#     study_method = st.selectbox(
#         "Lernmethode",
#         ["coaching", "online Videos", "gemischt", "self-study", "group study"],
#         key="study_method",
#     )
#     internet_access = st.checkbox("Internetzugang")

# if st.button("🔮 Vorhersage starten"):
#     # Feature-Vektor erstellen
#     input_data = pd.DataFrame(
#         {
#             # "age": [age],
#             # "gender": [gender],
#             # "course": [course],
#             "study_hours": [study_hours],
#             "class_attendance": [class_attendance],
#             # "internet_access": [internet_access],
#             "sleep_hours": [sleep_hours],
#             # "sleep_quality": [sleep_quality],
#             # "study_method": [study_method],
#             # "facility_rating": [facility_rating],
#             # "exam_difficulty": [exam_difficulty],
#         }
#     )
#     # Vorhersage
#     prediction = model.predict(input_data)[0]
#     # Ergebnis anzeigen
#     st.success(f"🪄 Vorhergesagtes Prüfungsergebnis: {prediction:.2f}%!")

# -----
