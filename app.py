import streamlit as st
from components import app_sidebar

st.set_page_config(
    page_title="Planetary Insight Engine",
    page_icon="assets/icons/icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

app_sidebar.create_sidebar()

st.markdown(
    """
    <style>
        .top-banner {
            width: 100%;
            height: 150px;
            background-image: url('https://via.placeholder.com/1500x150');
            background-size: cover;
            background-position: center;
            margin-bottom: 20px;
        }
    </style>
    <div class="top-banner"></div>
    """,
    unsafe_allow_html=True
)