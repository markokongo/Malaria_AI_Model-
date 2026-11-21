
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Malaria AI Surveillance Dashboard",
    page_icon="🦟",
    layout="wide"
)

st.title("🦟 Malaria AI and Mathematical Modelling Dashboard")
st.subheader("Precision Malaria Surveillance, Early Warning and Intervention")

data_file = Path("malaria_surveillance_data.csv")

if data_file.exists():
    df = pd.read_csv(data_file)

    st.success("Malaria surveillance dataset loaded successfully.")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total recorded cases", int(df["Observed_cases"].sum()))
    col2.metric("Average daily cases", round(df["Observed_cases"].mean(), 2))
    col3.metric("Maximum daily cases", int(df["Observed_cases"].max()))

    st.subheader("Daily malaria cases")
    st.line_chart(df.set_index("Day")["Observed_cases"])

    st.subheader("Climate and malaria surveillance")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Day"], df["Temperature"], label="Temperature")
    ax.plot(df["Day"], df["Rainfall"], label="Rainfall")
    ax.set_xlabel("Day")
    ax.set_ylabel("Climate measurement")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Surveillance data")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download surveillance dataset",
        data=df.to_csv(index=False),
        file_name="malaria_surveillance_data.csv",
        mime="text/csv"
    )

else:
    st.error("malaria_surveillance_data.csv was not found.")

st.caption(
    "Research prototype. The supplied surveillance dataset is synthetic "
    "and does not represent actual Kenyan malaria cases."
)
