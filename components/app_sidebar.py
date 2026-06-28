import streamlit as st
import pandas as pd
import base64
from components import local_def

local_def.load_css("assets/style.css")

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception:
        return None

# This function creates the sidebar for navigation in the web app.
def create_sidebar():
    # Define display names and internal page values
    pages = {
        "🏠 Home": "Home",
        "🌡 Temperature Analysis": "temp Analysis",
        "🪨 Surface Material Prediction": "Surface Material Prediction",
        "📤 Upload": "Upload",
        "👥 About Team": "About Team"
    }

    # Initialize session state for current page if not exists
    if "current_page" not in st.session_state:
        st.session_state.current_page = "🏠 Home"
        
    # Check if redirect flag is set
    if st.session_state.get("redirect_to_temp_analysis", False):
        st.session_state.sidebar_page_select = "🌡 Temperature Analysis"
        st.session_state.current_page = "🌡 Temperature Analysis"
        st.session_state.redirect_to_temp_analysis = False

    # Check query params for redirect
    query_page = st.query_params.get("page", None)
    if query_page:
        # Normalize plus signs to spaces
        query_page = query_page.replace("+", " ")
        # Find matching display name
        for k, v in pages.items():
            if v == query_page:
                st.session_state.sidebar_page_select = k
                st.session_state.current_page = k
                break
        if "page" in st.query_params:
            del st.query_params["page"]

    with st.sidebar:
        img_b64 = get_base64_image("assets/icons/logo1.png")
        if img_b64:
            st.markdown(
                f'<div style="background-color: #0f172a; padding: 15px; border-radius: 12px; margin-bottom: 1.5rem; display: flex; justify-content: center; align-items: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">'
                f'<img src="data:image/png;base64,{img_b64}" style="max-width: 100%; height: auto; object-fit: contain;">'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.image("assets/icons/logo1.png", width='stretch')
            st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

        # Map current page to default index
        default_index = 0
        if st.session_state.current_page in pages:
            default_index = list(pages.keys()).index(st.session_state.current_page)

        # Dropdown menu with pretty display names
        display_name = st.selectbox(
            "Select Page:", 
            list(pages.keys()), 
            index=default_index,
            key="sidebar_page_select"
        )
        st.session_state.current_page = display_name

    # Return the actual page value
    return pages[display_name]

def upload_page_sidebar():
    st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    if 'analysis_settings' not in st.session_state:
        st.session_state.analysis_settings = {
            'max_file_size': 50,
            'handle_missing': 'Forward fill',
            'show_detailed_info': True,
            'sample_size': 10,
            'show_statistics': True,
            'chart_theme': 'plotly',
        }
    
    st.sidebar.markdown("""
        <div class="sidebar-section">
            <h3>🎛️ Analysis Settings</h3>
            <p style="margin: 0; font-size: 14px;">Configure your data analysis preferences</p>
        </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("📁 File Upload Settings")

    max_file_size = st.sidebar.slider(
        "Maximum File Size (MB)",
        min_value=10,
        max_value=200,
        value=st.session_state.analysis_settings['max_file_size'],
        step=10,
        help="Set the maximum allowed file size for uploads"
    )
   
    st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.sidebar.markdown("Data Processing")

    handle_missing = st.sidebar.selectbox(
        "Handle Missing Values",
        options=['Keep as-is', 'Drop rows', 'Fill with mean', 'Fill with median', 'Forward fill'],
        index=['Keep as-is', 'Drop rows', 'Fill with mean', 'Fill with median', 'Forward fill'].index(
            st.session_state.analysis_settings['handle_missing']
        ),
        help="Choose how to handle missing values in your dataset"
    )
        
    st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.sidebar.markdown("Analysis Display")
    
    show_detailed_info = st.sidebar.checkbox(
        "Show Detailed Column Info",
        value=st.session_state.analysis_settings['show_detailed_info'],
        help="Display comprehensive column information including data types and sample values"
    )

    show_statistics = st.sidebar.checkbox(
        "Show Basic Statistics",
        value=st.session_state.analysis_settings['show_statistics'],
        help="Display basic statistical summary (mean, median, std, etc.)"
    )

    sample_size = st.sidebar.slider(
        "Sample Size for Preview",
        min_value=5,
        max_value=100,
        value=st.session_state.analysis_settings['sample_size'],
        step=5,
        help="Number of rows to show in data preview"
    )
    
    st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.sidebar.markdown("Visualization Settings")

    chart_theme = st.sidebar.selectbox(
        "Chart Theme",
        options=['plotly', 'plotly_white', 'plotly_dark', 'seaborn', 'simple_white'],
        index=['plotly', 'plotly_white', 'plotly_dark', 'seaborn', 'simple_white'].index(
            st.session_state.analysis_settings['chart_theme']
        ),
        help="Choose the visual theme for charts and graphs"
    )
      
    st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.sidebar.markdown("Quick Actions")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Reset Settings", help="Reset all settings to default values"):
            st.session_state.analysis_settings = {
                'max_file_size': 50,
                'auto_detect_encoding': True,
                'handle_missing': 'Keep as-is',
                'show_detailed_info': True,
                'show_advanced_stats': False,
                'sample_size': 10,
                'show_statistics': True,
                'correlation_threshold': 0.5,
                'chart_theme': 'plotly',
                'decimal_places': 2,
                'date_format': 'auto'
            }
            st.rerun()

    st.sidebar.markdown("Settings Management")

    if st.sidebar.button("📤 Export Settings"):
        settings_json = pd.Series(st.session_state.analysis_settings).to_json()
        st.sidebar.download_button(
            label="Download Settings JSON",
            data=settings_json,
            file_name="analysis_settings.json",
            mime="application/json"
        )
    uploaded_settings = st.sidebar.file_uploader(
        "📥 Import Settings",
        type=['json'],
        help="Upload a previously exported settings file"
    )
    
    if uploaded_settings is not None:
        try:
            import json
            settings_data = json.load(uploaded_settings)
            st.session_state.analysis_settings.update(settings_data)
            st.sidebar.success("Settings imported successfully!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error importing settings: {str(e)}")
    
    st.session_state.analysis_settings.update({
        'max_file_size': max_file_size,
        'handle_missing': handle_missing,
        'show_detailed_info': show_detailed_info,
        'sample_size': sample_size,
        'show_statistics': show_statistics,
        'chart_theme': chart_theme
    })
    
    return st.session_state.analysis_settings

def surface_material_page_sidebar():
    st.sidebar.markdown("""
        <div class="sidebar-section">
            <h3>🎛️ Analysis Settings</h3>
            <p style="margin: 0; font-size: 14px;">Configure your data analysis preferences</p>
        </div>
    """, unsafe_allow_html=True)