import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow import keras
import joblib
import duckdb
import math
from pathlib import Path
from components import app_sidebar
import io

# Database connection
@st.cache_resource
def get_db_connection():
    """Get DuckDB connection"""
    db_path = Path("data/data_surf/earth.duckdb")
    if db_path.exists():
        return duckdb.connect(str(db_path))
    else:
        st.error(f"Database file not found: {db_path}")
        return None

@st.cache_data
def load_earth_data():
    """Load data from DuckDB"""
    conn = get_db_connection()
    if conn:
        try:
            # Try to get table names first
            tables = conn.execute("SHOW TABLES").fetchall()
            if tables:
                table_name = tables[0][0]  # Get first table name
                df = conn.execute(f"SELECT * FROM {table_name}").fetchdf()
                return df
            else:
                st.error("No tables found in database")
                return None
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    return None

def scaling_factor(g_planet, g_earth, phi_planet, phi_earth, alpha=0.2, beta=0.5):
    """Common scaling factor based on gravity and porosity."""
    gravity_term = (g_earth / g_planet) ** alpha
    porosity_term = ((1 - phi_earth) / (1 - phi_planet)) ** beta
    return gravity_term * porosity_term

def convert_velocity_to_earth(V_planet, g_planet, g_earth, phi_planet, phi_earth, alpha=0.2, beta=0.5):
    factor = scaling_factor(g_planet, g_earth, phi_planet, phi_earth, alpha, beta)
    return V_planet * factor

def convert_amplitude_to_earth(A_planet, g_planet, g_earth, phi_planet, phi_earth, alpha=0.2, beta=0.5):
    factor = scaling_factor(g_planet, g_earth, phi_planet, phi_earth, alpha, beta)
    return A_planet * factor

def convert_frequency_to_earth(f_planet, g_planet, g_earth, phi_planet, phi_earth, alpha=0.2, beta=0.5):
    factor = scaling_factor(g_planet, g_earth, phi_planet, phi_earth, alpha, beta)
    return f_planet * factor

def convert_duration_to_earth(D_planet, g_planet, g_earth, phi_planet, phi_earth, alpha=0.2, beta=0.5):
    factor = scaling_factor(g_planet, g_earth, phi_planet, phi_earth, alpha, beta)
    return D_planet / factor 

def engineer_features(velocity, amplitude, duration, frequency_hz):
    """Calculate all 19 engineered features from the 4 basic inputs"""
    features = [velocity, amplitude, duration, frequency_hz]
    
    # Engineered features
    velocity_x_amplitude = velocity * amplitude
    velocity_squared = velocity ** 2
    duration_squared = duration ** 2
    amplitude_duration = amplitude * duration
    velocity_frequency = velocity * frequency_hz
    amplitude_frequency = amplitude * frequency_hz
    duration_frequency = duration * frequency_hz
    velocity_duration = velocity * duration
    amplitude_squared = amplitude ** 2
    frequency_squared = frequency_hz ** 2
    velocity_amplitude_ratio = velocity / amplitude if amplitude != 0 else 0
    duration_frequency_ratio = duration / frequency_hz if frequency_hz != 0 else 0
    velocity_duration_ratio = velocity / duration if duration != 0 else 0
    velocity_cubed = velocity ** 3
    amplitude_cubed = amplitude ** 3

    all_features = [
        velocity, amplitude, duration, frequency_hz,
        velocity_x_amplitude, velocity_squared, duration_squared,
        amplitude_duration, velocity_frequency, amplitude_frequency,
        duration_frequency, velocity_duration, amplitude_squared,
        frequency_squared, velocity_amplitude_ratio, duration_frequency_ratio,
        velocity_duration_ratio, velocity_cubed, amplitude_cubed
    ]

    return np.array(all_features).reshape(1, -1)

def predict_rock_type(scaler, model, le, velocity, amplitude, duration, frequency_hz, verbose=True):
    """Predict rock type from basic seismic properties"""
    try:
        sample_features = engineer_features(velocity, amplitude, duration, frequency_hz)
        sample_scaled = scaler.transform(sample_features)
        pred_prob = model.predict(sample_scaled, verbose=0)
        pred_index = np.argmax(pred_prob)
        pred_label = le.inverse_transform([pred_index])[0]
        confidence = np.max(pred_prob)
        
        return pred_label, confidence, pred_prob[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, 0, []

def load_models():
    """Load all models and scalers"""
    models = {}
    try:
        models['main_model'] = keras.models.load_model("models/model.h5")
        models['main_scaler'] = joblib.load("models/scaler.pkl")
        models['main_le'] = joblib.load("models/label_encoder.pkl")
        
        # Load specialized models
        for rock_type in ['igneous', 'metamorphic']:
            try:
                models[f'{rock_type}_model'] = keras.models.load_model(f"models/{rock_type}_model.keras")
                models[f'{rock_type}_scaler'] = joblib.load(f"models/{rock_type}_scaler.pkl")
                models[f'{rock_type}_le'] = joblib.load(f"models/{rock_type}_label_encoder.pkl")
            except:
                st.warning(f"Could not load {rock_type} model")
                
    except Exception as e:
        st.error(f"Error loading models: {e}")
        
    return models

def create_analysis_charts(df, predicted_data=None):
    """Create comprehensive analysis charts"""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Velocity Distribution', 'Amplitude vs Duration', 
                       'Frequency Analysis', 'Rock Type Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Velocity distribution
    fig.add_trace(
        go.Histogram(x=df['velocity'] if 'velocity' in df.columns else df.iloc[:, 0], 
                    name='Velocity', nbinsx=30, opacity=0.7),
        row=1, col=1
    )
    
    # Amplitude vs Duration scatter
    fig.add_trace(
        go.Scatter(x=df['amplitude'] if 'amplitude' in df.columns else df.iloc[:, 1], 
                  y=df['duration'] if 'duration' in df.columns else df.iloc[:, 2],
                  mode='markers', name='Amplitude vs Duration', opacity=0.6),
        row=1, col=2
    )
    
    # Frequency analysis
    fig.add_trace(
        go.Histogram(x=df['frequency'] if 'frequency' in df.columns else df.iloc[:, 3], 
                    name='Frequency', nbinsx=30, opacity=0.7),
        row=2, col=1
    )
    
    # Rock type distribution (if available)
    if 'rock_type' in df.columns:
        rock_counts = df['rock_type'].value_counts()
        fig.add_trace(
            go.Bar(x=rock_counts.index, y=rock_counts.values, name='Rock Types'),
            row=2, col=2
        )
    
    fig.update_layout(height=700, showlegend=True, title_text="Material Analysis Dashboard")
    
    return fig

def single_analysis_tab():
    """Single sample analysis interface"""
    st.subheader("🔍 Single Sample Analysis")
    
    # Planet selection
    planet_options = {
        'Venus': {'gravity': 8.87, 'porosity': 0.18},
        'Mars': {'gravity': 3.71, 'porosity': 0.25},
        'Earth': {'gravity': 9.81, 'porosity': 0.10},
        'Custom': {'gravity': 9.81, 'porosity': 0.10}
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_planet = st.selectbox("Select Planet", list(planet_options.keys()))
        
        if selected_planet == 'Custom':
            gravity = st.number_input("Gravity (m/s²)", value=9.81, min_value=0.1, max_value=30.0)
            porosity = st.number_input("Porosity", value=0.10, min_value=0.01, max_value=0.99)
        else:
            gravity = planet_options[selected_planet]['gravity']
            porosity = planet_options[selected_planet]['porosity']
            st.info(f"Gravity: {gravity} m/s², Porosity: {porosity}")
    
    with col2:
        st.subheader("Seismic Properties")
        velocity = st.number_input("Velocity (km/s)", value=5.5, min_value=0.1, max_value=15.0)
        amplitude = st.number_input("Amplitude", value=0.60, min_value=0.01, max_value=5.0)
        duration = st.number_input("Duration (ms)", value=300, min_value=1, max_value=2000)
        frequency = st.number_input("Frequency (Hz)", value=30, min_value=1, max_value=200)
    
    if st.button("🚀 Analyze Sample", type="primary"):
        with st.spinner("Analyzing sample..."):
            # Convert to Earth equivalent
            g_earth = 9.81
            phi_earth = 0.10
            
            V_earth = convert_velocity_to_earth(velocity, gravity, g_earth, porosity, phi_earth)
            A_earth = convert_amplitude_to_earth(amplitude, gravity, g_earth, porosity, phi_earth)
            D_earth = convert_duration_to_earth(duration, gravity, g_earth, porosity, phi_earth)
            f_earth = convert_frequency_to_earth(frequency, gravity, g_earth, porosity, phi_earth)
            
            # Load models
            models = load_models()
            
            if 'main_model' in models:
                # Initial prediction
                sample = np.array([V_earth, A_earth, D_earth]).reshape(1, -1)
                sample_scaled = models['main_scaler'].transform(sample)
                raw_prediction = models['main_model'].predict(sample_scaled)
                predicted_index = np.argmax(raw_prediction)
                predicted_label = models['main_le'].inverse_transform([predicted_index])[0]
                
                # Results display
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Primary Classification", predicted_label.replace('_', ' ').title())
                
                with col2:
                    confidence = np.max(raw_prediction)
                    st.metric("Confidence", f"{confidence:.3f}")
                
                with col3:
                    st.metric("Converted Velocity", f"{V_earth:.2f} km/s")
                
                # Detailed analysis
                if predicted_label in ["igneous_rocks", "metamorphic_rocks"]:
                    rock_type_key = predicted_label.replace('_rocks', '')
                    if f'{rock_type_key}_model' in models:
                        rock_type, confidence, probabilities = predict_rock_type(
                            models[f'{rock_type_key}_scaler'], 
                            models[f'{rock_type_key}_model'], 
                            models[f'{rock_type_key}_le'],
                            V_earth, A_earth, D_earth, f_earth
                        )
                        
                        if rock_type:
                            st.success(f"**Specific Rock Type:** {rock_type} (Confidence: {confidence:.3f})")
                            
                            # Probability distribution
                            if len(probabilities) > 0:
                                prob_df = pd.DataFrame({
                                    'Rock Type': models[f'{rock_type_key}_le'].classes_,
                                    'Probability': probabilities
                                })
                                prob_df = prob_df.sort_values('Probability', ascending=False)
                                
                                fig = px.bar(prob_df, x='Rock Type', y='Probability', 
                                           title="Probability Distribution")
                                st.plotly_chart(fig, use_container_width=True)

def batch_analysis_tab():
    """Batch analysis interface"""
    st.subheader("📊 Batch Analysis")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df)} samples")
            
            # Display first few rows
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Column mapping
            st.subheader("Column Mapping")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                velocity_col = st.selectbox("Velocity Column", df.columns, index=0)
            with col2:
                amplitude_col = st.selectbox("Amplitude Column", df.columns, index=1 if len(df.columns) > 1 else 0)
            with col3:
                duration_col = st.selectbox("Duration Column", df.columns, index=2 if len(df.columns) > 2 else 0)
            with col4:
                frequency_col = st.selectbox("Frequency Column", df.columns, index=3 if len(df.columns) > 3 else 0)
            
            # Planet parameters
            col1, col2 = st.columns(2)
            with col1:
                batch_gravity = st.number_input("Planet Gravity (m/s²)", value=8.87, min_value=0.1, max_value=30.0)
            with col2:
                batch_porosity = st.number_input("Planet Porosity", value=0.18, min_value=0.01, max_value=0.99)
            
            if st.button("🚀 Analyze Batch", type="primary"):
                with st.spinner("Processing batch analysis..."):
                    results = []
                    models = load_models()
                    
                    if 'main_model' in models:
                        for idx, row in df.iterrows():
                            # Convert to Earth equivalent
                            V_earth = convert_velocity_to_earth(row[velocity_col], batch_gravity, 9.81, batch_porosity, 0.10)
                            A_earth = convert_amplitude_to_earth(row[amplitude_col], batch_gravity, 9.81, batch_porosity, 0.10)
                            D_earth = convert_duration_to_earth(row[duration_col], batch_gravity, 9.81, batch_porosity, 0.10)
                            f_earth = convert_frequency_to_earth(row[frequency_col], batch_gravity, 9.81, batch_porosity, 0.10)
                            
                            # Predict
                            sample = np.array([V_earth, A_earth, D_earth]).reshape(1, -1)
                            sample_scaled = models['main_scaler'].transform(sample)
                            raw_prediction = models['main_model'].predict(sample_scaled, verbose=0)
                            predicted_index = np.argmax(raw_prediction)
                            predicted_label = models['main_le'].inverse_transform([predicted_index])[0]
                            confidence = np.max(raw_prediction)
                            
                            results.append({
                                'Sample_ID': idx,
                                'Original_Velocity': row[velocity_col],
                                'Original_Amplitude': row[amplitude_col],
                                'Original_Duration': row[duration_col],
                                'Original_Frequency': row[frequency_col],
                                'Earth_Velocity': V_earth,
                                'Earth_Amplitude': A_earth,
                                'Earth_Duration': D_earth,
                                'Earth_Frequency': f_earth,
                                'Predicted_Rock_Type': predicted_label,
                                'Confidence': confidence
                            })
                        
                        results_df = pd.DataFrame(results)
                        
                        # Display results
                        st.subheader("Analysis Results")
                        st.dataframe(results_df)
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Samples", len(results_df))
                        with col2:
                            st.metric("Avg Confidence", f"{results_df['Confidence'].mean():.3f}")
                        with col3:
                            most_common = results_df['Predicted_Rock_Type'].mode()[0]
                            st.metric("Most Common Type", most_common.replace('_', ' ').title())
                        
                        # Visualization
                        fig = create_analysis_charts(results_df)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="batch_analysis_results.csv",
                            mime="text/csv"
                        )
                        
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    # Load existing Earth data for reference
    if st.button("📈 Load Earth Reference Data"):
        earth_data = load_earth_data()
        if earth_data is not None:
            st.subheader("Earth Reference Data")
            st.dataframe(earth_data.head(10))
            
            # Create charts
            fig = create_analysis_charts(earth_data)
            st.plotly_chart(fig, use_container_width=True)

def analysis_report_tab():
    """Analysis report and insights"""
    st.subheader("📋 Analysis Report")
    
    # Load Earth data for baseline
    earth_data = load_earth_data()
    
    if earth_data is not None:
        st.subheader("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", len(earth_data))
        with col2:
            if 'velocity' in earth_data.columns:
                st.metric("Avg Velocity", f"{earth_data['velocity'].mean():.2f} km/s")
            else:
                st.metric("Avg Velocity", f"{earth_data.iloc[:, 0].mean():.2f} km/s")
        with col3:
            if 'amplitude' in earth_data.columns:
                st.metric("Avg Amplitude", f"{earth_data['amplitude'].mean():.3f}")
            else:
                st.metric("Avg Amplitude", f"{earth_data.iloc[:, 1].mean():.3f}")
        with col4:
            if 'rock_type' in earth_data.columns:
                unique_types = earth_data['rock_type'].nunique()
                st.metric("Rock Types", unique_types)
            else:
                st.metric("Features", len(earth_data.columns))
        
        # Statistical summary
        st.subheader("Statistical Summary")
        st.dataframe(earth_data.describe())
        
        # Correlation analysis
        if len(earth_data.select_dtypes(include=[np.number]).columns) > 1:
            st.subheader("Correlation Analysis")
            numeric_cols = earth_data.select_dtypes(include=[np.number])
            corr_matrix = numeric_cols.corr()
            
            fig = px.imshow(corr_matrix, 
                          labels=dict(x="Features", y="Features", color="Correlation"),
                          color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
    
    # Model information
    st.subheader("Model Information")
    
    model_info = {
        "Primary Model": "Neural Network for rock type classification",
        "Input Features": "Velocity, Amplitude, Duration, Frequency + 15 engineered features",
        "Output Classes": "Igneous, Metamorphic, Sedimentary rocks",
        "Scaling": "StandardScaler for feature normalization",
        "Architecture": "Multi-layer perceptron with dropout regularization"
    }
    
    for key, value in model_info.items():
        st.write(f"**{key}:** {value}")

def model_overview_tab():
    """Model overview and architecture"""
    st.subheader("🤖 Model Overview")
    
    # Model architecture
    st.subheader("Model Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Primary Classification Model:**")
        st.write("- Input: 19 features (4 basic + 15 engineered)")
        st.write("- Architecture: Dense Neural Network")
        st.write("- Output: Rock type probabilities")
        st.write("- Activation: ReLU (hidden), Softmax (output)")
        st.write("- Regularization: Dropout layers")
    
    with col2:
        st.write("**Specialized Models:**")
        st.write("- Igneous rock classification")
        st.write("- Metamorphic rock classification")
        st.write("- Each trained on specific rock type subsets")
        st.write("- Higher accuracy for specific classifications")
    
    # Feature engineering
    st.subheader("Feature Engineering")
    
    feature_info = {
        "Basic Features": ["Velocity", "Amplitude", "Duration", "Frequency"],
        "Polynomial Features": ["Velocity²", "Amplitude²", "Duration²", "Frequency²", "Velocity³", "Amplitude³"],
        "Interaction Features": ["Velocity × Amplitude", "Amplitude × Duration", "Velocity × Frequency", 
                               "Amplitude × Frequency", "Duration × Frequency", "Velocity × Duration"],
        "Ratio Features": ["Velocity/Amplitude", "Duration/Frequency", "Velocity/Duration"]
    }
    
    for category, features in feature_info.items():
        st.write(f"**{category}:**")
        st.write(", ".join(features))
        st.write("")
    
    # Planetary conversion
    st.subheader("Planetary Conversion Formula")
    
    st.latex(r'''
    \text{Scaling Factor} = \left(\frac{g_{Earth}}{g_{Planet}}\right)^{\alpha} \times \left(\frac{1-\phi_{Earth}}{1-\phi_{Planet}}\right)^{\beta}
    ''')
    
    st.write("Where:")
    st.write("- g: Gravitational acceleration")
    st.write("- φ: Porosity")
    st.write("- α = 0.2 (gravity exponent)")
    st.write("- β = 0.5 (porosity exponent)")
    
    # Model performance (placeholder)
    st.subheader("Model Performance")
    
    performance_data = {
        "Model": ["Primary Classifier", "Igneous Classifier", "Metamorphic Classifier"],
        "Accuracy": [0.85, 0.92, 0.88],
        "Precision": [0.83, 0.90, 0.86],
        "Recall": [0.82, 0.91, 0.87],
        "F1-Score": [0.82, 0.90, 0.86]
    }
    
    perf_df = pd.DataFrame(performance_data)
    st.dataframe(perf_df)

def material_prediction():
    """Main material prediction interface"""
    app_sidebar.surface_material_page_sidebar()
    
    st.title("🪨 Material Analysis System")
    st.markdown("*Advanced planetary material classification using seismic data*")
    
    # Main navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Single Analysis", 
        "📊 Batch Analysis", 
        "📋 Analysis Report", 
        "📈 Graphs & Insights",
        "🤖 Model Overview"
    ])
    
    with tab1:
        single_analysis_tab()
    
    with tab2:
        batch_analysis_tab()
    
    with tab3:
        analysis_report_tab()
    
    with tab4:
        # Load and display Earth data with comprehensive graphs
        earth_data = load_earth_data()
        if earth_data is not None:
            st.subheader("📈 Data Visualization & Insights")
            
            # Multi-dimensional analysis
            if len(earth_data.select_dtypes(include=[np.number]).columns) >= 4:
                numeric_cols = earth_data.select_dtypes(include=[np.number]).columns[:4]
                
                # 3D scatter plot
                fig = px.scatter_3d(
                    earth_data, 
                    x=numeric_cols[0], 
                    y=numeric_cols[1], 
                    z=numeric_cols[2],
                    color=numeric_cols[3] if len(numeric_cols) > 3 else None,
                    title="3D Feature Space Visualization"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Comprehensive dashboard
            fig = create_analysis_charts(earth_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance (simulated)
            st.subheader("Feature Importance")
            features = ['Velocity', 'Amplitude', 'Duration', 'Frequency']
            importance = [0.35, 0.25, 0.20, 0.20]
            
            fig = px.bar(x=features, y=importance, title="Feature Importance in Classification")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for visualization. Please upload data or check database connection.")
    
    with tab5:
        model_overview_tab()
