import streamlit as st  
# Importing custom-built components like the sidebar, page handlers, and utilities.
from components import *
from utils import db_utils  # Utility functions for database (if used)

# Setting the basic layout and configuration of the web page.
st.set_page_config(
    page_title="Planetary Insight Engine",      # Title of the browser tab
    page_icon="assets/icons/icon1.png",          # Favicon (small icon on tab)
    layout="wide",                              # Page will take full screen width
    initial_sidebar_state="expanded"            # Sidebar will be expanded by default
)

# Creating a navigation sidebar and saving the selected page name to "page"
page = app_sidebar.create_sidebar()

# Loading custom CSS to improve the look and feel of the page.
local_def.load_css("assets/style.css")

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