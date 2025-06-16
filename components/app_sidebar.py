import streamlit as st

# This function creates the sidebar for navigation in the web app
def create_sidebar():
   # Everything inside this block will appear in the sidebar
   with st.sidebar:
        # Display a header in the sidebar
        st.markdown("# 🪐 Navigation")

        # A horizontal line for separation
        st.markdown("---")

        # Dropdown menu for selecting which page to view
        page = st.selectbox(
            "Select Page:",  # Label shown above the dropdown
            ["Home", "Data Analysis", "Mars Weather", "Upload", "👥 About Team"],  # List of pages
            format_func=lambda x: {
                "Home": "🏠 Home",                   # Add emojis for better UI
                "Data Analysis": "📊 Data Analysis",
                "Upload": "📤 Upload",
                "👥 About Team": "👥 About Team"      # Team/about section
            }[x]  # This function replaces the text with icons + labels
        )

   # Add a horizontal line after the sidebar selection (outside the sidebar)
   st.markdown("---")

   # Return the selected page name to be used in the main app
   return page
