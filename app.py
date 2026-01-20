import streamlit as st

# Page Config
# st.set_page_config(page_title="Exam Score Prediction", page_icon=":material/home:")

pages = {
    "Seiten": [
        st.Page("pages/index.py", title="Homepage", icon="🏠", default=True),
        st.Page(
            "pages/data_cleaning.py",
            title="Datenbereinigung",
            icon="🧼",
        ),
        st.Page(
            "pages/visualization.py",
            title="Visualisierung",
            icon="🔍",
        ),
        st.Page(
            "pages/machine_learning.py",
            title="Machine Learning",
            icon="🔮",
        ),
    ]
}

pg = st.navigation(pages)
pg.run()

st.set_page_config(
    page_title=f"{pg.title}", 
    page_icon=pg.icon,
    layout="wide"
)