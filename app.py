import streamlit as st
from components import app_sidebar

app_sidebar.create_sidebar()

st.set_page_config(
    page_title="Planetary Insight Engine",
    page_icon="assets/icons/icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

