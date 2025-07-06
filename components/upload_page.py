import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from components import local_def

def create_professional_header():
    
    
    st.markdown("""
    <div class="main-header-upload-section">
        <h1 class="header-title-upload-section">Professional Data Analytics Platform</h1>
        <p class="header-subtitle-upload-section">Advanced file upload, analysis, and data modeling capabilities</p>
    </div>
    """, unsafe_allow_html=True)


def analyze_dataframe(df, file_name):
    """Perform comprehensive analysis of the uploaded DataFrame"""
    st.markdown('<div class="section-title-upload-section">📈 Data Analysis Results</div>', unsafe_allow_html=True)
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Data Quality", "📈 Visualizations", "📋 Sample Data"])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="stats-card">
                <p class="metric-value-upload-section">{len(df)}</p>
                <p class="metric-label-upload-section">Total Rows</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stats-card">
                <p class="metric-value-upload-section">{len(df.columns)}</p>
                <p class="metric-label-upload-section">Total Columns</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.markdown(f"""
            <div class="stats-card">
                <p class="metric-value-upload-section">{memory_usage:.2f} MB</p>
                <p class="metric-label-upload-section">Memory Usage</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            st.markdown(f"""
            <div class="stats-card-upload-section">
                <p class="metric-value-upload-section">{numeric_cols}</p>
                <p class="metric-label-upload-section">Numeric Columns</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Column information
        st.subheader("📋 Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(col_info, use_container_width=True)
    
    with tab2:
        st.subheader("🔍 Data Quality Assessment")
        
        # Missing values heatmap
        if df.isnull().sum().sum() > 0:
            fig_missing = px.imshow(
                df.isnull().astype(int),
                title="Missing Values Heatmap",
                color_continuous_scale="Reds",
                aspect="auto"
            )
            fig_missing.update_layout(height=400)
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("✅ No missing values found in the dataset!")
        
        # Duplicate rows check
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            st.warning(f"⚠️ Found {duplicates} duplicate rows ({duplicates/len(df)*100:.2f}%)")
        else:
            st.success("✅ No duplicate rows found!")
        
        # Data type recommendations
        st.subheader("💡 Data Type Recommendations")
        recommendations = []
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col])
                    recommendations.append(f"'{col}' could be converted to datetime")
                except:
                    try:
                        pd.to_numeric(df[col])
                        recommendations.append(f"'{col}' could be converted to numeric")
                    except:
                        pass
        
        if recommendations:
            for rec in recommendations:
                st.info(f"💡 {rec}")
        else:
            st.success("✅ Data types appear to be appropriate!")
    
    with tab3:
        st.subheader("📈 Data Visualizations")
        
        # Numeric columns distribution
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select column for distribution plot:", numeric_cols)
            
            col1, col2 = st.columns(2)
            with col1:
                fig_hist = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                fig_box = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
                st.plotly_chart(fig_box, use_container_width=True)
        
        # Correlation matrix for numeric columns
        if len(numeric_cols) > 1:
            st.subheader("🔗 Correlation Matrix")
            corr_matrix = df[numeric_cols].corr()
            fig_corr = px.imshow(
                corr_matrix,
                title="Correlation Matrix",
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab4:
        st.subheader("📋 Sample Data Preview")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**First 10 rows:**")
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.write("**Last 10 rows:**")
            st.dataframe(df.tail(10), use_container_width=True)
        
        # Statistical summary
        st.subheader("📊 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)


def upload_section(section_title, section_key, icon):
    st.markdown(f"""
    <div class="upload-section-upload-section">
        <div class="section-title-upload-section">{icon} {section_title}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for this section
    if f"uploaded_files_{section_key}" not in st.session_state:
        st.session_state[f"uploaded_files_{section_key}"] = []
    
    # File uploader
    uploaded_file = st.file_uploader(
        f"📤 Upload CSV or Excel File for {section_title}",
        type=["csv", "xlsx", "xls"],
        key=f"uploader_{section_key}"
    )
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Check if file already uploaded
            if uploaded_file.name not in [f.name for f in st.session_state[f"uploaded_files_{section_key}"]]:
                st.session_state[f"uploaded_files_{section_key}"].append(uploaded_file)
                st.markdown(f"""
                <div class="success-message-upload-section">
                    ✅ File '{uploaded_file.name}' uploaded successfully!
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-message-upload-section">
                    ⚠️ File '{uploaded_file.name}' is already uploaded.
                </div>
                """, unsafe_allow_html=True)
            
            # Save to DuckDB
            file_name_with_ext = uploaded_file.name
            file_name, _ = os.path.splitext(file_name_with_ext)
            db_name = f"{file_name}_{section_key}.duckdb"
            db_path = os.path.join("data", db_name)
            table_name = f"{file_name}_{section_key}".replace(" ", "_").replace("-", "_")
            os.makedirs("data", exist_ok=True)
            
            con = duckdb.connect(db_path)
            con.execute(f"DROP TABLE IF EXISTS {table_name}")
            con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
            con.close()
            
            st.success(f"✅ Saved to database: `{db_path}` → table: `{table_name}`")
            
            # Perform analysis
            analyze_dataframe(df, uploaded_file.name)
            
            # Display uploaded files
            if st.session_state[f"uploaded_files_{section_key}"]:
                st.subheader(f"📁 Uploaded Files - {section_title}")
                for i, file in enumerate(st.session_state[f"uploaded_files_{section_key}"]):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"📄 {file.name}")
                    with col2:
                        file_size = len(file.getvalue()) / 1024
                        st.write(f"{file_size:.1f} KB")
                    with col3:
                        if st.button("🗑️ Delete", key=f"delete_{section_key}_{i}"):
                            del st.session_state[f"uploaded_files_{section_key}"][i]
                            st.rerun()
            
            return df
            
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            return None
    
    else:
        st.info(f"ℹ️ Please upload a file to begin analysis for {section_title}")
        return None


def upload():
    """Main application function"""
    # Set page config
    st.set_page_config(
        page_title="Professional Data Uploader",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Create professional header
    create_professional_header()
    
    # Sidebar with additional options
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # File format preferences
        st.subheader("📁 File Preferences")
        auto_detect_encoding = st.checkbox("Auto-detect encoding", value=True)
        max_file_size = st.slider("Max file size (MB)", 1, 200, 50)
        
        # Analysis options
        st.subheader("📊 Analysis Options")
        show_advanced_stats = st.checkbox("Show advanced statistics", value=True)
        generate_report = st.checkbox("Generate PDF report", value=False)
        
        # Data processing options
        st.subheader("⚡ Processing Options")
        handle_missing = st.selectbox("Handle missing values", ["Keep as-is", "Drop rows", "Fill with mean"])
        
        st.markdown("---")
        st.markdown("**💡 Tips:**")
        st.markdown("• Upload CSV or Excel files up to 200MB")
        st.markdown("• Use descriptive filenames")
        st.markdown("• Check data quality before analysis")
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        df1 = upload_section("Primary Dataset", "primary", "🎯")
    
    with col2:
        df2 = upload_section("Surface Model Dataset", "surface", "🌊")
    
    # Compare datasets if both are uploaded
    if 'df1' in locals() and 'df2' in locals() and df1 is not None and df2 is not None:
        st.markdown("---")
        st.markdown('<div class="section-title-upload-section">🔗 Dataset Comparison</div>', unsafe_allow_html=True)
        
        comp_col1, comp_col2, comp_col3 = st.columns(3)
        
        with comp_col1:
            st.markdown(f"""
            <div class="stats-card-upload-section">
                <p class="metric-value-upload-section">{abs(len(df1) - len(df2))}</p>
                <p class="metric-label-upload-section">Row Difference</p>
            </div>
            """, unsafe_allow_html=True)
        
        with comp_col2:
            st.markdown(f"""
            <div class="stats-card-upload-section">
                <p class="metric-value-upload-section">{abs(len(df1.columns) - len(df2.columns))}</p>
                <p class="metric-label-upload-section">Column Difference</p>
            </div>
            """, unsafe_allow_html=True)
        
        with comp_col3:
            common_cols = len(set(df1.columns) & set(df2.columns))
            st.markdown(f"""
            <div class="stats-card-upload-section">
                <p class="metric-value-upload-section">{common_cols}</p>
                <p class="metric-label-upload-section">Common Columns</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show common columns
        if common_cols > 0:
            st.subheader("🔗 Common Columns")
            common_column_names = list(set(df1.columns) & set(df2.columns))
            st.write(", ".join(common_column_names))
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 2rem;">
        <p>Professional Data Analytics Platform | Built by team Cosmo Compute</p>
        <p>Upload • Analyze • Visualize • Model</p>
    </div>
    """, unsafe_allow_html=True)