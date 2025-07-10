import streamlit as st  
# Importing custom-built components like the sidebar, page handlers, and utilities.
from components import *
from utils import db_utils  # Utility functions for database (if used)

# Setting the basic layout and configuration of the web page.
st.set_page_config(
    page_title="Planetary Insight Engine",      # Title of the browser tab
    page_icon="assets/icons/icon.png",          # Favicon (small icon on tab)
    layout="wide",                              # Page will take full screen width
    initial_sidebar_state="expanded"            # Sidebar will be expanded by default
)

# Creating a navigation sidebar and saving the selected page name to "page"
page = app_sidebar.create_sidebar()

# Loading custom CSS to improve the look and feel of the page.
local_def.load_css("assets/style.css")

# Adding a header to the web page using HTML inside Streamlit.
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
""", unsafe_allow_html=True)  # Allow raw HTML to style content

# Depending on what the user selects from the sidebar, show the correct page

g_planet = None
phi_planet = None # Initialize with None

if page == "Home":
    home_page.home()

elif page == "Upload":
    upload_page.upload()

elif page == "Surface Material Prediction":
    surf_model.material_prediction()
   
elif page == "temp Analysis":
    temp_model.main()  # Run the Data Analysis module when selected

elif page == "About Team":
    about_page.about_us()

else:
    notfoundpage.notfound()  # If the page doesn't exist, show a "Not Found" message