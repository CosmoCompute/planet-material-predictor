import streamlit as st
import pandas as pd
import duckdb
import os
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from components import app_sidebar, local_def

local_def.load_css("assets/style.css")

def header():
    st.markdown("""
        <div class="main-header-upload-section">
            <h1 class="header-title-upload-section">Professional Data Analytics Platform</h1>
            <p class="header-subtitle-upload-section">Advanced file upload, analysis, and data modeling capabilities</p>
        </div>
    """, unsafe_allow_html=True)

def analyze_dataframe(df, file_name, analysis_settings=None):
    """Enhanced data analysis with configurable options"""
    if analysis_settings is None:
        analysis_settings = st.session_state.get('analysis_settings', {})
    
    st.markdown('<div class="section-title-upload-section">📈 Data Analysis Results</div>', unsafe_allow_html=True)
    
    # Create tabs based on user preferences
    tab_names = ["📊 Overview", "🔍 Data Quality", "📈 Visualizations", "📋 Sample Data"]
    if analysis_settings.get('show_advanced_stats', False):
        tab_names.append("🔬 Advanced Statistics")
    
    tabs = st.tabs(tab_names)

    with tabs[0]:  # Overview tab
        # Apply data handling preferences
        processed_df = apply_data_handling(df, analysis_settings)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="stats-card-professional">
                    <div class="metric-content">
                        <p class="metric-value-professional">{len(processed_df):,}</p>
                        <p class="metric-label-professional">Total Rows</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="stats-card-professional">
                    <div class="metric-content">
                        <p class="metric-value-professional">{len(processed_df.columns)}</p>
                        <p class="metric-label-professional">Total Columns</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            memory_usage = processed_df.memory_usage(deep=True).sum() / 1024 / 1024
            st.markdown(f"""
                <div class="stats-card-professional">
                    <div class="metric-content">
                        <p class="metric-value-professional">{memory_usage:.2f} MB</p>
                        <p class="metric-label-professional">Memory Usage</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            numeric_cols = len(processed_df.select_dtypes(include=[np.number]).columns)
            st.markdown(f"""
                <div class="stats-card-professional">
                    <div class="metric-content">
                        <p class="metric-value-professional">{numeric_cols}</p>
                        <p class="metric-label-professional">Numeric Columns</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Enhanced column information
        if analysis_settings.get('show_detailed_info', True):
            st.markdown("### 📋 Column Information")
            col_info = create_column_info_table(processed_df)
            st.dataframe(col_info, use_container_width=True, height=300)
    
    with tabs[1]:  # Data Quality tab
        st.markdown("### 🔍 Data Quality Assessment")
        
        # Missing values analysis
        missing_analysis = analyze_missing_values(processed_df)
        if missing_analysis['has_missing']:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.plotly_chart(missing_analysis['heatmap'], use_container_width=True)
            with col2:
                st.markdown("#### Missing Values Summary")
                st.dataframe(missing_analysis['summary'], use_container_width=True)
        else:
            st.success("✅ No missing values found in the dataset!")

        # Duplicate analysis
        duplicates = processed_df.duplicated().sum()
        if duplicates > 0:
            st.warning(f"⚠️ Found {duplicates:,} duplicate rows ({duplicates/len(processed_df)*100:.2f}%)")
            if st.button("Remove Duplicates"):
                processed_df = processed_df.drop_duplicates()
                st.success("Duplicates removed successfully!")
        else:
            st.success("✅ No duplicate rows found!")

        # Data type consistency check
        st.markdown("#### Data Type Analysis")
        dtype_analysis = analyze_data_types(processed_df)
        st.dataframe(dtype_analysis, use_container_width=True)
    
    with tabs[2]:  # Visualizations tab
        st.markdown("### 📈 Data Visualizations")
        
        create_interactive_visualizations(processed_df, analysis_settings)
    
    with tabs[3]:  # Sample Data tab
        st.markdown("### 📋 Sample Data Preview")
        
        sample_size = analysis_settings.get('sample_size', 10)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**First {sample_size} rows:**")
            st.dataframe(processed_df.head(sample_size), use_container_width=True)
        
        with col2:
            st.markdown(f"**Last {sample_size} rows:**")
            st.dataframe(processed_df.tail(sample_size), use_container_width=True)

        if analysis_settings.get('show_statistics', True):
            st.markdown("### 📊 Statistical Summary")
            st.dataframe(processed_df.describe(), use_container_width=True)
    
    # Advanced Statistics tab (if enabled)
    if len(tabs) > 4:
        with tabs[4]:
            st.markdown("### 🔬 Advanced Statistical Analysis")
            create_advanced_statistics(processed_df)

def apply_data_handling(df, settings):
    """Apply data handling preferences from sidebar"""
    processed_df = df.copy()
    
    handle_missing = settings.get('handle_missing', 'Keep as-is')
    
    if handle_missing == 'Drop rows':
        processed_df = processed_df.dropna()
    elif handle_missing == 'Fill with mean':
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].mean())
    elif handle_missing == 'Fill with median':
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].median())
    elif handle_missing == 'Forward fill':
        processed_df = processed_df.fillna(method='ffill')
    
    return processed_df

def create_column_info_table(df):
    """Create enhanced column information table"""
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Null %': (df.isnull().sum() / len(df) * 100).round(2),
        'Unique Values': df.nunique(),
        'Sample Values': [str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else 'N/A' for col in df.columns]
    })
    return col_info

def analyze_missing_values(df):
    """Analyze missing values in the dataset"""
    missing_sum = df.isnull().sum()
    has_missing = missing_sum.sum() > 0
    
    result = {'has_missing': has_missing}
    
    if has_missing:
        # Create heatmap
        fig_missing = px.imshow(
            df.isnull().astype(int),
            title="Missing Values Heatmap",
            color_continuous_scale="Reds",
            aspect="auto",
            labels=dict(color="Missing")
        )
        fig_missing.update_layout(height=400)
        result['heatmap'] = fig_missing
        
        # Create summary table
        missing_summary = pd.DataFrame({
            'Column': missing_sum.index,
            'Missing Count': missing_sum.values,
            'Missing %': (missing_sum / len(df) * 100).round(2)
        })
        missing_summary = missing_summary[missing_summary['Missing Count'] > 0]
        result['summary'] = missing_summary
    
    return result

def analyze_data_types(df):
    """Analyze data types and suggest improvements"""
    dtype_info = []
    
    for col in df.columns:
        col_data = df[col].dropna()
        current_type = str(df[col].dtype)
        
        # Suggest optimal data type
        suggested_type = current_type
        memory_usage = df[col].memory_usage(deep=True)
        
        if df[col].dtype == 'object':
            # Check if it could be categorical
            if col_data.nunique() / len(col_data) < 0.5:
                suggested_type = 'category'
            # Check if it could be datetime
            elif any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated']):
                suggested_type = 'datetime'
        
        dtype_info.append({
            'Column': col,
            'Current Type': current_type,
            'Suggested Type': suggested_type,
            'Memory Usage (bytes)': memory_usage,
            'Unique Values': col_data.nunique(),
            'Sample Value': str(col_data.iloc[0]) if len(col_data) > 0 else 'N/A'
        })
    
    return pd.DataFrame(dtype_info)

def create_interactive_visualizations(df, settings):
    """Create interactive visualizations based on user preferences"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(numeric_cols) > 0:
        st.markdown("#### 📊 Numeric Data Visualizations")
        
        col1, col2 = st.columns(2)
        with col1:
            selected_col = st.selectbox("Select column for distribution analysis:", numeric_cols)
            chart_type = st.selectbox("Chart type:", ["Histogram", "Box Plot", "Violin Plot", "Density Plot"])
        
        with col2:
            color_by = st.selectbox("Color by (optional):", ["None"] + list(categorical_cols))
            
        # Create visualization based on selection
        if chart_type == "Histogram":
            fig = px.histogram(df, x=selected_col, 
                             color=color_by if color_by != "None" else None,
                             title=f"Distribution of {selected_col}")
        elif chart_type == "Box Plot":
            fig = px.box(df, y=selected_col, 
                        color=color_by if color_by != "None" else None,
                        title=f"Box Plot of {selected_col}")
        elif chart_type == "Violin Plot":
            fig = px.violin(df, y=selected_col, 
                           color=color_by if color_by != "None" else None,
                           title=f"Violin Plot of {selected_col}")
        else:  # Density Plot
            fig = px.density_contour(df, x=selected_col, 
                                   title=f"Density Plot of {selected_col}")
        
        st.plotly_chart(fig, use_container_width=True)

    # Correlation analysis
    if len(numeric_cols) > 1:
        st.markdown("#### 🔗 Correlation Analysis")
        
        corr_method = st.selectbox("Correlation method:", ["pearson", "spearman", "kendall"])
        corr_matrix = df[numeric_cols].corr(method=corr_method)
        
        fig_corr = px.imshow(
            corr_matrix,
            title=f"Correlation Matrix ({corr_method.title()})",
            color_continuous_scale="RdBu",
            aspect="auto",
            text_auto=True
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)

    # Categorical data analysis
    if len(categorical_cols) > 0:
        st.markdown("#### 📋 Categorical Data Analysis")
        
        selected_cat = st.selectbox("Select categorical column:", categorical_cols)
        value_counts = df[selected_cat].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            fig_bar = px.bar(x=value_counts.index, y=value_counts.values,
                           title=f"Distribution of {selected_cat}")
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            fig_pie = px.pie(values=value_counts.values, names=value_counts.index,
                           title=f"Proportion of {selected_cat}")
            st.plotly_chart(fig_pie, use_container_width=True)

def create_advanced_statistics(df):
    """Create advanced statistical analysis"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        st.markdown("#### 📊 Distribution Analysis")
        
        for col in numeric_cols:
            with st.expander(f"Analysis for {col}"):
                col_data = df[col].dropna()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Skewness", f"{col_data.skew():.3f}")
                with col2:
                    st.metric("Kurtosis", f"{col_data.kurtosis():.3f}")
                with col3:
                    st.metric("Std Dev", f"{col_data.std():.3f}")
                
                # Normality test visualization
                from scipy import stats
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=col_data, name="Data", opacity=0.7))
                
                # Add normal distribution overlay
                mu, sigma = col_data.mean(), col_data.std()
                x = np.linspace(col_data.min(), col_data.max(), 100)
                y = stats.norm.pdf(x, mu, sigma) * len(col_data) * (col_data.max() - col_data.min()) / 100
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Normal Distribution'))
                
                fig.update_layout(title=f"Distribution Analysis: {col}", height=300)
                st.plotly_chart(fig, use_container_width=True)

def upload_section_enhanced(section_title, section_key, icon):
    """Enhanced upload section with better error handling and progress tracking"""
    st.markdown(f"""
        <div class="upload-section-professional">
            <div class="section-header">
                <div class="section-icon">{icon}</div>
                <div class="section-title-professional">{section_title}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if f"uploaded_files_{section_key}" not in st.session_state:
        st.session_state[f"uploaded_files_{section_key}"] = []

    # Get analysis settings from sidebar
    analysis_settings = st.session_state.get('analysis_settings', {})
    max_file_size = analysis_settings.get('max_file_size', 50) * 1024 * 1024  # Convert to bytes

    # File uploader with enhanced options
    uploaded_file = st.file_uploader(
        f"📤 Upload CSV or Excel File for {section_title}",
        type=["csv", "xlsx", "xls"],
        key=f"uploader_{section_key}",
        help=f"Maximum file size: {analysis_settings.get('max_file_size', 50)} MB"
    )

    if uploaded_file is not None:
        # Check file size
        file_size = len(uploaded_file.getvalue())
        if file_size > max_file_size:
            st.error(f"❌ File size ({file_size/1024/1024:.1f} MB) exceeds maximum allowed size ({analysis_settings.get('max_file_size', 50)} MB)")
            return None

        try:
            # Show progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Reading file...")
            progress_bar.progress(25)
            
            # Read file with encoding detection if enabled
            if analysis_settings.get('auto_detect_encoding', True):
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                else:
                    df = pd.read_excel(uploaded_file)
            else:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
            
            progress_bar.progress(50)
            status_text.text("Processing data...")
            
            # Check if file already uploaded
            if uploaded_file.name not in [f.name for f in st.session_state[f"uploaded_files_{section_key}"]]:
                st.session_state[f"uploaded_files_{section_key}"].append(uploaded_file)
                
                progress_bar.progress(75)
                status_text.text("Saving to database...")
                
                # Save to database
                save_to_database(df, uploaded_file.name, section_key)
                
                progress_bar.progress(100)
                status_text.text("✅ Upload completed successfully!")
                
                st.markdown(f"""
                    <div class="success-message-professional">
                        <div class="message-icon">✅</div>
                        <div class="message-text">File '{uploaded_file.name}' uploaded successfully!</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Clear progress indicators after a moment
                import time
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
            else:
                st.markdown(f"""
                    <div class="warning-message-professional">
                        <div class="message-icon">⚠️</div>
                        <div class="message-text">File '{uploaded_file.name}' is already uploaded.</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Analyze the dataframe
            analyze_dataframe(df, uploaded_file.name, analysis_settings)
            
            # Show uploaded files list
            display_uploaded_files(section_title, section_key)
            
            return df

        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            
            # Provide helpful error messages
            if "encoding" in str(e).lower():
                st.info("💡 Try enabling 'Auto-detect encoding' in the sidebar settings.")
            elif "memory" in str(e).lower():
                st.info("💡 Try reducing the file size or increasing the memory limit.")
            
            return None
    else:
        st.markdown(f"""
            <div class="info-message-professional">
                <div class="message-icon">ℹ️</div>
                <div class="message-text">Please upload a file to begin analysis for {section_title}</div>
            </div>
        """, unsafe_allow_html=True)
        return None

def save_to_database(df, file_name, section_key):
    """Save dataframe to DuckDB database"""
    file_name_without_ext = os.path.splitext(file_name)[0]
    db_name = f"{file_name_without_ext}_{section_key}.duckdb"
    db_path = os.path.join("data", "data_temp", db_name)
    table_name = f"{file_name_without_ext}_{section_key}".replace(" ", "_").replace("-", "_")
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Save to database
    con = duckdb.connect(db_path)
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    con.close()

def display_uploaded_files(section_title, section_key):
    """Display uploaded files with enhanced UI"""
    if st.session_state[f"uploaded_files_{section_key}"]:
        st.markdown(f"### 📁 Uploaded Files - {section_title}")
        
        for i, file in enumerate(st.session_state[f"uploaded_files_{section_key}"]):
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.markdown(f"**📄 {file.name}**")
                
                with col2:
                    file_size = len(file.getvalue()) / 1024
                    if file_size > 1024:
                        st.write(f"{file_size/1024:.1f} MB")
                    else:
                        st.write(f"{file_size:.1f} KB")
                
                with col3:
                    upload_time = datetime.now().strftime("%H:%M")
                    st.write(f"🕒 {upload_time}")
                
                with col4:
                    if st.button("🗑️ Remove", key=f"delete_{section_key}_{i}", 
                               help="Remove this file from the session"):
                        st.session_state[f"uploaded_files_{section_key}"].pop(i)
                        st.rerun()
                
                st.markdown("---")

def upload():
    header()
    app_sidebar.upload_page_sidebar()

    st.markdown("Data Upload & Analysis")

    col1, col2 = st.columns(2) 
    with col1:
        df1 = upload_section_enhanced("Temperature Model Dataset", "temp", "🌡️")
    
    with col2:
        df2 = upload_section_enhanced("Surface Model Dataset", "surface", "🪨")
    
    # Additional features section
    if df1 is not None or df2 is not None:
        st.markdown("### 🔧 Additional Tools")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Generate Report", help="Generate a comprehensive analysis report"):
                st.info("Report generation feature coming soon!")
        
        with col2:
            if st.button("💾 Export Data", help="Export processed data"):
                st.info("Data export feature coming soon!")
        
        with col3:
            if st.button("🔄 Refresh Analysis", help="Refresh the analysis with current settings"):
                st.rerun()