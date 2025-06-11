import streamlit as st
from components import local_def

st.sidebar.title("h,kmg")
local_def.load_css("assets/style.css")

# Apply full background image with custom CSS
st.markdown("""
            
        <h1>
            hello
        </h1>
""", unsafe_allow_html=True)

# Optional: Display transparent content
st.markdown("<br><br><h1 style='color:white; text-align:center;'>Your App</h1>", unsafe_allow_html=True)
