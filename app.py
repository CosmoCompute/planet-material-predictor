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
            background-image: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="20" cy="20" r="2" fill="white" opacity="0.1"/><circle cx="80" cy="30" r="1.5" fill="white" opacity="0.1"/><circle cx="40" cy="70" r="1" fill="white" opacity="0.1"/><circle cx="90" cy="80" r="2.5" fill="white" opacity="0.1"/><circle cx="10" cy="60" r="1.2" fill="white" opacity="0.1"/></svg>');
            background-size: cover;
            background-position: center;
            margin-bottom: 20px;
        }
    </style>
    <div class="top-banner"></div>
    """,
    unsafe_allow_html=True
)