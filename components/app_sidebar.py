import streamlit as st

def create_sidebar():
   with st.sidebar:
        st.markdown("# 🪐 Navigation")
        st.markdown("---")
        
        page = st.selectbox(
            "Select Page:",
            ["Home", "Data Analysis", "Mars Weather", "Upload", "👥 About Team"],
            format_func=lambda x: {
                "Home": "🏠 Home",
                "Data Analysis": "📊 Data Analysis", 
                "Mars Weather": "🌌 Mars Weather",
                "Upload": "📤 Upload",
                "👥 About Team": "👥 About Team"  # Changed from "About" to "👥 About Team"
            }[x]
        )
   st.markdown("---")
   return page