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

def upload_page_sidebar():
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.subheader("📁 File Preferences")
        auto_detect_encoding = st.checkbox("Auto-detect encoding", value=True)
        max_file_size = st.slider("Max file size (MB)", 1, 200, 50)
        
        st.subheader("📊 Analysis Options")
        show_stats = st.checkbox("Show statistics", value=True)
        generate_report = st.checkbox("Generate PDF report", value=False)
        
        st.subheader("⚡ Processing Options")
        handle_missing = st.selectbox("Handle missing values", ["Keep as-is", "Drop rows", "Fill with mean"])
        
        st.markdown("---")
        st.markdown("💡 Tips:")
        st.markdown("• Upload CSV or Excel files up to 200MB")
        st.markdown("• Use descriptive filenames")
        st.markdown("• Check data quality before analysis")
    