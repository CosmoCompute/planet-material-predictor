import streamlit as st
from components import app_sidebar
from components import local_def


st.set_page_config(
    page_title="Planetary Insight Engine",
    page_icon="assets/icons/icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

local_def.load_css("assets/style.css")

app_sidebar.create_sidebar()

st.markdown("""
            
""", unsafe_allow_html=True)