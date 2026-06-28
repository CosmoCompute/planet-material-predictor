# Import the Streamlit library
import streamlit as st

# Define a function to load and apply a CSS file to the Streamlit app
def load_css(file_name):
    # Open the CSS file using the provided file name
    with open(file_name) as f:
        # Read the content of the CSS file and inject it into the app's HTML using Markdown
        # The 'unsafe_allow_html=True' flag allows raw HTML to be rendered
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
