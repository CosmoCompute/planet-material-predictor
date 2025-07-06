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

import streamlit as st
import pandas as pd

def upload_page_sidebar():
    """Enhanced sidebar for upload page with comprehensive analysis settings"""
    
    # Custom CSS for professional sidebar styling
    st.markdown("""
        <style>
        .sidebar-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            color: white;
        }
        
        .sidebar-section h3 {
            color: white !important;
            margin-top: 0 !important;
            font-size: 18px !important;
        }
        
        .sidebar-divider {
            border-top: 2px solid #e0e0e0;
            margin: 20px 0;
        }
        
        .setting-help {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state for analysis settings
    if 'analysis_settings' not in st.session_state:
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
    
    st.sidebar.markdown("""
        <div class="sidebar-section">
            <h3>🎛️ Analysis Settings</h3>
            <p style="margin: 0; font-size: 14px;">Configure your data analysis preferences</p>
        </div>
    """, unsafe_allow_html=True)
    
    # File Upload Settings
    st.sidebar.markdown("### 📁 File Upload Settings")
    
    # Maximum file size
    max_file_size = st.sidebar.slider(
        "Maximum File Size (MB)",
        min_value=10,
        max_value=500,
        value=st.session_state.analysis_settings['max_file_size'],
        step=10,
        help="Set the maximum allowed file size for uploads"
    )
    
    # Auto-detect encoding
    auto_detect_encoding = st.sidebar.checkbox(
        "Auto-detect File Encoding",
        value=st.session_state.analysis_settings['auto_detect_encoding'],
        help="Automatically detect file encoding for CSV files"
    )
    
    # Date format handling
    date_format = st.sidebar.selectbox(
        "Date Format Detection",
        options=['auto', 'DD/MM/YYYY', 'MM/DD/YYYY', 'YYYY-MM-DD', 'custom'],
        index=['auto', 'DD/MM/YYYY', 'MM/DD/YYYY', 'YYYY-MM-DD', 'custom'].index(
            st.session_state.analysis_settings['date_format']
        ),
        help="Choose how dates should be parsed"
    )
    
    st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # Data Processing Settings
    st.sidebar.markdown("### 🔧 Data Processing")
    
    # Missing value handling
    handle_missing = st.sidebar.selectbox(
        "Handle Missing Values",
        options=['Keep as-is', 'Drop rows', 'Fill with mean', 'Fill with median', 'Forward fill'],
        index=['Keep as-is', 'Drop rows', 'Fill with mean', 'Fill with median', 'Forward fill'].index(
            st.session_state.analysis_settings['handle_missing']
        ),
        help="Choose how to handle missing values in your dataset"
    )
    
    # Decimal places for display
    decimal_places = st.sidebar.slider(
        "Decimal Places",
        min_value=0,
        max_value=6,
        value=st.session_state.analysis_settings['decimal_places'],
        help="Number of decimal places to display in results"
    )
    
    st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # Analysis Display Settings
    st.sidebar.markdown("### 📊 Analysis Display")
    
    # Show detailed information
    show_detailed_info = st.sidebar.checkbox(
        "Show Detailed Column Info",
        value=st.session_state.analysis_settings['show_detailed_info'],
        help="Display comprehensive column information including data types and sample values"
    )
    
    # Show advanced statistics
    show_advanced_stats = st.sidebar.checkbox(
        "Enable Advanced Statistics",
        value=st.session_state.analysis_settings['show_advanced_stats'],
        help="Include advanced statistical analysis (skewness, kurtosis, normality tests)"
    )
    
    # Show basic statistics
    show_statistics = st.sidebar.checkbox(
        "Show Basic Statistics",
        value=st.session_state.analysis_settings['show_statistics'],
        help="Display basic statistical summary (mean, median, std, etc.)"
    )
    
    # Sample size for preview
    sample_size = st.sidebar.slider(
        "Sample Size for Preview",
        min_value=5,
        max_value=100,
        value=st.session_state.analysis_settings['sample_size'],
        step=5,
        help="Number of rows to show in data preview"
    )
    
    st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # Visualization Settings
    st.sidebar.markdown("### 📈 Visualization Settings")
    
    # Chart theme
    chart_theme = st.sidebar.selectbox(
        "Chart Theme",
        options=['plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn', 'simple_white'],
        index=['plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn', 'simple_white'].index(
            st.session_state.analysis_settings['chart_theme']
        ),
        help="Choose the visual theme for charts and graphs"
    )
    
    # Correlation threshold
    correlation_threshold = st.sidebar.slider(
        "Correlation Threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.analysis_settings['correlation_threshold'],
        step=0.1,
        help="Minimum correlation value to highlight in correlation analysis"
    )
    
    st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # Quick Actions
    st.sidebar.markdown("### ⚡ Quick Actions")
    
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
    
    with col2:
        if st.button("💾 Save Settings", help="Save current settings as default"):
            st.sidebar.success("Settings saved!")
    
    # Export/Import Settings
    st.sidebar.markdown("### 📋 Settings Management")
    
    # Export settings
    if st.sidebar.button("📤 Export Settings"):
        settings_json = pd.Series(st.session_state.analysis_settings).to_json()
        st.sidebar.download_button(
            label="Download Settings JSON",
            data=settings_json,
            file_name="analysis_settings.json",
            mime="application/json"
        )
    
    # Import settings
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
    
    st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # System Information
    st.sidebar.markdown("### 🔍 System Information")
    
    # Memory usage indicator
    import psutil
    memory_percent = psutil.virtual_memory().percent
    
    st.sidebar.markdown(f"""
        <div style="background: {'#ff6b6b' if memory_percent > 80 else '#4ecdc4' if memory_percent > 60 else '#45b7d1'}; 
                    padding: 10px; border-radius: 5px; color: white; text-align: center;">
            <strong>Memory Usage: {memory_percent:.1f}%</strong>
        </div>
    """, unsafe_allow_html=True)
    
    # Update session state with current settings
    st.session_state.analysis_settings.update({
        'max_file_size': max_file_size,
        'auto_detect_encoding': auto_detect_encoding,
        'handle_missing': handle_missing,
        'show_detailed_info': show_detailed_info,
        'show_advanced_stats': show_advanced_stats,
        'sample_size': sample_size,
        'show_statistics': show_statistics,
        'correlation_threshold': correlation_threshold,
        'chart_theme': chart_theme,
        'decimal_places': decimal_places,
        'date_format': date_format
    })
    
    # Display current settings summary at bottom
    with st.sidebar.expander("📋 Current Settings Summary"):
        st.write("**File Settings:**")
        st.write(f"- Max Size: {max_file_size} MB")
        st.write(f"- Auto Encoding: {'✅' if auto_detect_encoding else '❌'}")
        
        st.write("**Processing:**")
        st.write(f"- Missing Values: {handle_missing}")
        st.write(f"- Decimal Places: {decimal_places}")
        
        st.write("**Display:**")
        st.write(f"- Sample Size: {sample_size}")
        st.write(f"- Advanced Stats: {'✅' if show_advanced_stats else '❌'}")
        
        st.write("**Visualization:**")
        st.write(f"- Theme: {chart_theme}")
        st.write(f"- Correlation Threshold: {correlation_threshold}")
    
    return st.session_state.analysis_settings