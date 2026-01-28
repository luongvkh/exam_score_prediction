import streamlit as st
import pandas as pd

st.title("🔦 Daten-Exploration")


@st.cache_data
def load_data():
    return pd.read_csv("data/processed/exam_score_prediction_cleaned.csv", index_col=0)


df = load_data()

tab1, tab2, tab3 = st.tabs(
    [
        "💡 Übersicht",
        "🔍 Datenqualität",
        "📊 Statistiken",
    ]
)

with tab1:
    st.title("Dataset Übersicht")

    st.markdown("### Dimensionen")
    st.markdown(f"{len(df)} Zeilen, {len(df.columns)} Spalten")

    st.markdown("### Daten-Vorschau")
    row_count = st.slider("Anzahl Zeilen", min_value=0, max_value=len(df), value=10)
    st.dataframe(df.head(row_count), use_container_width=True)

    st.markdown("### Datentypen")
    dtypes_df = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": df.dtypes.values,
            "Count": df.count().values,
        }
    )
    st.dataframe(dtypes_df, use_container_width=True)

with tab2:
    st.title("Datenqualität")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Fehlende Werte")
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            st.warning(f"⚠️ Datensatz hat {missing_count} fehlende Werte!")
        else:
            st.success("✅ Keine fehlenden Werte!")

        missing_values_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Count": df.count().values,  # Gefüllte Zellen
                "Missing": df.isnull().sum().values,  # Fehlende Werte
                "Total": len(df),  # Gesamtanzahl Zeilen
            }
        )
        st.dataframe(missing_values_df, use_container_width=True)

    with col2:
        st.markdown("### Duplikate")
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            st.warning(f"⚠️ Datensatz hat {duplicate_count} Duplikate!")
        else:
            st.success("✅ Keine Duplikate!")

        duplicates_info = pd.DataFrame(
            {
                "Metric": [
                    "Total Rows",
                    "Unique Rows",
                    "Duplicate Rows",
                    "Duplicate %",
                ],
                "Value": [
                    len(df),
                    len(df.drop_duplicates()),
                    df.duplicated().sum(),
                    f"{(df.duplicated().sum() / len(df) * 100):.2f}%",
                ],
            }
        )

        st.dataframe(duplicates_info, use_container_width=True)

# Optional: Duplikate anzeigen
if df.duplicated().any():
    if st.checkbox("Duplizierte Zeilen anzeigen"):
        st.dataframe(df[df.duplicated(keep=False)], use_container_width=True)

with tab3:
    st.title("Statistische Zusammenfassung")
    st.dataframe(df.describe())
    st.markdown("### Kategoriale Variablen")
    categorical_cols = list(df.select_dtypes(include=["object", "category"]).columns)
    for cat_col in categorical_cols:
        with st.expander(f"📊 {cat_col}"):
            counts_df = (
                df[cat_col].value_counts().reset_index()
            )  # value_counts gibt eine series zurück, reset_index macht daraus ein df
            counts_df.columns = [cat_col, "Count"]  # columns benennt die spalten um
            counts_df["Percentage"] = (
                counts_df["Count"] / counts_df["Count"].sum() * 100
            ).round(2)
            st.dataframe(counts_df)
