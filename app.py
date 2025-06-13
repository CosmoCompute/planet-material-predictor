import streamlit as st
from components import app_sidebar, local_def, upload_page, notfoundpage

st.set_page_config(
    page_title="Planetary Insight Engine",
    page_icon="assets/icons/icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

page=app_sidebar.create_sidebar()

local_def.load_css("assets/style.css")
st.markdown("""
    <div class="main-header">
        <div style="position: relative; z-index=1;">
            <div class="planet-animation" style="font-size: 4rem; margin-bottom: 1rem;">🪐</div>
            <h1 class="Edu">Planet Material Predictor</h1>
            <p>Advanced analysis for planetary composition prediction</p>
            <div style="margin-top: 2rem;">
                <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 10px; margin: 0 0.5rem;">✨ ML Powered</span>
                <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 10px; margin: 0 0.5rem;">🎯 95%+ Accuracy</span>
                <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 10px; margin: 0 0.5rem;">⚡ Real-time</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

if page == "Home":
    st.title("home")

elif page == "Upload":
    expect=upload_page.upload()
    if expect is None:
        st.subheader("Upload a File")

else:
    notfoundpage.notfound()