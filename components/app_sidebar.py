import streamlit as st

def create_sidebar():
 st.sidebar.title("🚀 Navigator")
 page_options ={
    "🏠 Home": "Home",
    "🔮 Predict Materials": "Predict Materials",
    "📊 Data Analysis": "Data Analysis",
    "🧪 Upload": "Upload",
    "👥 About Team": "About"
 }

 selected_page=st.sidebar.selectbox(
  "Choose a Section",
  list(page_options.keys())
 )

 page=page_options[selected_page]

 return page