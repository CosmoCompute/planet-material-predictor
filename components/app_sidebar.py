import streamlit as st

# This function creates the sidebar for navigation in the web app
def create_sidebar():
    # Define display names and internal page values
    pages = {
        "🏠 Home": "Home",
        "📊 Data Analysis": "Data Analysis",
        "📤 Upload": "Upload",
        "👥 About Team": "About Team"
    }

    with st.sidebar:
        st.markdown("# 🪐 Navigation")
        st.markdown("---")

        # Dropdown menu with pretty display names
        display_name = st.selectbox("Select Page:", list(pages.keys()))

    st.markdown("---")

    # Return the actual page value
    return pages[display_name]
