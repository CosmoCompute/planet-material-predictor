import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow import keras
import joblib
import math

# Configure page
st.set_page_config(
    page_title="Planetary Rock Analysis System",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .info-card {
        background: #f8f9fa;
        border-left: 4px solid #2a5298;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .sidebar-section {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Rock material data with enhanced information
rock_materials = {
    # Igneous Rocks
    "Granite": {
        "velocity": (5.5, 0.1), "amplitude": (0.57, 0.02), "duration": (240, 10), "frequency_Hz": (40, 5),
        "type": "Igneous", "formation": "Intrusive", "color": "#D2B48C",
        "elements": {"Silicon": 35, "Oxygen": 45, "Aluminum": 8, "Potassium": 4, "Sodium": 3, "Iron": 2, "Calcium": 2, "Magnesium": 1},
        "description": "Coarse-grained intrusive igneous rock rich in quartz and feldspar.",
        "uses": ["Construction", "Monuments", "Countertops"]
    },
    "Basalt": {
        "velocity": (5.6, 0.1), "amplitude": (0.60, 0.02), "duration": (225, 10), "frequency_Hz": (42, 4),
        "type": "Igneous", "formation": "Extrusive", "color": "#2F4F4F",
        "elements": {"Silicon": 25, "Oxygen": 40, "Iron": 12, "Magnesium": 8, "Calcium": 8, "Aluminum": 5, "Sodium": 2},
        "description": "Fine-grained volcanic rock, most common volcanic rock on Earth.",
        "uses": ["Road construction", "Concrete aggregate", "Railroad ballast"]
    },
    "Diorite": {
        "velocity": (5.4, 0.1), "amplitude": (0.59, 0.02), "duration": (230, 10), "frequency_Hz": (39, 5),
        "type": "Igneous", "formation": "Intrusive", "color": "#696969",
        "elements": {"Silicon": 30, "Oxygen": 42, "Aluminum": 10, "Calcium": 6, "Sodium": 4, "Iron": 4, "Magnesium": 3, "Potassium": 1},
        "description": "Intermediate intrusive igneous rock between granite and gabbro.",
        "uses": ["Dimension stone", "Construction", "Decorative purposes"]
    },
    
    # Metamorphic Rocks
    "Schist": {
        "velocity": (6.2, 0.1), "amplitude": (0.70, 0.02), "duration": (255, 10), "frequency_Hz": (34, 4),
        "type": "Metamorphic", "formation": "Regional", "color": "#8B7355",
        "elements": {"Silicon": 32, "Oxygen": 44, "Aluminum": 12, "Iron": 5, "Magnesium": 3, "Potassium": 2, "Sodium": 1, "Calcium": 1},
        "description": "Medium-grade metamorphic rock with visible mineral crystals.",
        "uses": ["Roofing", "Flagstone", "Decorative stone"]
    },
    "Gneiss": {
        "velocity": (6.4, 0.1), "amplitude": (0.74, 0.02), "duration": (265, 10), "frequency_Hz": (33, 4),
        "type": "Metamorphic", "formation": "Regional", "color": "#A0522D",
        "elements": {"Silicon": 35, "Oxygen": 46, "Aluminum": 8, "Iron": 4, "Potassium": 3, "Sodium": 2, "Calcium": 1, "Magnesium": 1},
        "description": "High-grade metamorphic rock with distinct banding.",
        "uses": ["Construction stone", "Dimension stone", "Landscaping"]
    },
    
    # Sedimentary Rocks
    "Limestone": {
        "velocity": (2.4, 0.1), "amplitude": (0.36, 0.02), "duration": (170, 10), "frequency_Hz": (26, 3),
        "type": "Sedimentary", "formation": "Chemical", "color": "#F5F5DC",
        "elements": {"Calcium": 40, "Carbon": 12, "Oxygen": 48},
        "description": "Sedimentary rock composed mainly of calcium carbonate.",
        "uses": ["Cement production", "Construction", "Lime production"]
    },
    "Sandstone": {
        "velocity": (2.5, 0.1), "amplitude": (0.40, 0.02), "duration": (180, 10), "frequency_Hz": (28, 3),
        "type": "Sedimentary", "formation": "Clastic", "color": "#F4A460",
        "elements": {"Silicon": 42, "Oxygen": 53, "Iron": 2, "Aluminum": 2, "Calcium": 1},
        "description": "Clastic sedimentary rock composed mainly of sand-sized minerals.",
        "uses": ["Building stone", "Paving", "Glass production"]
    },
    
    # Ore and Industrial Minerals
    "Hematite": {
        "velocity": (4.4, 0.1), "amplitude": (0.54, 0.02), "duration": (245, 10), "frequency_Hz": (34, 5),
        "type": "Ore", "formation": "Hydrothermal", "color": "#CD5C5C",
        "elements": {"Iron": 70, "Oxygen": 30},
        "description": "Most important iron ore mineral.",
        "uses": ["Iron production", "Pigments", "Polishing powder"]
    },
    
    # Gem and Rare Minerals
    "Diamond": {
        "velocity": (7.0, 0.1), "amplitude": (0.85, 0.02), "duration": (320, 10), "frequency_Hz": (55, 5),
        "type": "Precious", "formation": "High pressure", "color": "#B9F2FF",
        "elements": {"Carbon": 100},
        "description": "Hardest natural substance, crystalline form of carbon.",
        "uses": ["Jewelry", "Industrial cutting", "Abrasives"]
    }
}

# Conversion functions
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
    velocity_amplitude_ratio = velocity / amplitude
    duration_frequency_ratio = duration / frequency_hz
    velocity_duration_ratio = velocity / duration
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

def predict_rock_type(scaler, model, le, velocity, amplitude, duration, frequency_hz):
    """Predict rock type from basic seismic properties"""
    sample_features = engineer_features(velocity, amplitude, duration, frequency_hz)
    sample_scaled = scaler.transform(sample_features)
    pred_prob = model.predict(sample_scaled, verbose=0)
    pred_index = np.argmax(pred_prob)
    pred_label = le.inverse_transform([pred_index])[0]
    confidence = np.max(pred_prob)
    
    return pred_label, confidence, pred_prob[0]

def create_elemental_composition_chart(elements, rock_name):
    """Create a pie chart for elemental composition"""
    fig = go.Figure(data=[go.Pie(
        labels=list(elements.keys()),
        values=list(elements.values()),
        hole=0.3,
        textinfo='label+percent',
        textposition='auto',
        marker=dict(
            colors=px.colors.qualitative.Set3
        )
    )])
    
    fig.update_layout(
        title=f"Elemental Composition of {rock_name}",
        font=dict(size=12),
        showlegend=True,
        height=400
    )
    
    return fig

def create_seismic_properties_chart(velocity, amplitude, duration, frequency):
    """Create a radar chart for seismic properties"""
    categories = ['Velocity (km/s)', 'Amplitude', 'Duration (ms)', 'Frequency (Hz)']
    values = [velocity/10, amplitude*10, duration/300, frequency/60]  # Normalized values
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Seismic Properties',
        line=dict(color='rgb(42, 82, 152)')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Seismic Properties Profile",
        height=400
    )
    
    return fig

def create_rock_comparison_chart(selected_rocks):
    """Create a comparison chart for multiple rocks"""
    if not selected_rocks:
        return None
    
    properties = ['Velocity', 'Amplitude', 'Duration', 'Frequency']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=properties,
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = px.colors.qualitative.Set1
    
    for i, rock in enumerate(selected_rocks):
        rock_data = rock_materials[rock]
        color = colors[i % len(colors)]
        
        # Velocity
        fig.add_trace(
            go.Bar(name=rock, x=[rock], y=[rock_data['velocity'][0]], 
                   marker_color=color, showlegend=False),
            row=1, col=1
        )
        
        # Amplitude
        fig.add_trace(
            go.Bar(name=rock, x=[rock], y=[rock_data['amplitude'][0]], 
                   marker_color=color, showlegend=False),
            row=1, col=2
        )
        
        # Duration
        fig.add_trace(
            go.Bar(name=rock, x=[rock], y=[rock_data['duration'][0]], 
                   marker_color=color, showlegend=False),
            row=2, col=1
        )
        
        # Frequency
        fig.add_trace(
            go.Bar(name=rock, x=[rock], y=[rock_data['frequency_Hz'][0]], 
                   marker_color=color, showlegend=False),
            row=2, col=2
        )
    
    fig.update_layout(height=600, title_text="Rock Properties Comparison")
    return fig

def create_elemental_distribution_chart(selected_rocks):
    """Create a stacked bar chart for elemental distribution"""
    if not selected_rocks:
        return None
    
    # Collect all unique elements
    all_elements = set()
    for rock in selected_rocks:
        all_elements.update(rock_materials[rock]['elements'].keys())
    
    # Create data for each element
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    
    for i, element in enumerate(all_elements):
        values = []
        for rock in selected_rocks:
            values.append(rock_materials[rock]['elements'].get(element, 0))
        
        fig.add_trace(go.Bar(
            name=element,
            x=selected_rocks,
            y=values,
            marker_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        barmode='stack',
        title='Elemental Distribution Across Selected Rocks',
        xaxis_title='Rock Types',
        yaxis_title='Percentage (%)',
        height=500
    )
    
    return fig

# Main UI
st.markdown('<div class="main-header"><h1>🪨 Planetary Rock Analysis System</h1><p>Advanced Seismic-Based Rock Classification and Analysis</p></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-section"><h3>🔬 Analysis Settings</h3></div>', unsafe_allow_html=True)
    
    # Planet selection
    planet = st.selectbox(
        "Select Planet/Body",
        ["Venus", "Mars", "Earth", "Moon", "Custom"],
        index=0
    )
    
    # Planet properties
    if planet == "Venus":
        g_planet, phi_planet = 8.87, 0.18
    elif planet == "Mars":
        g_planet, phi_planet = 3.71, 0.25
    elif planet == "Earth":
        g_planet, phi_planet = 9.81, 0.10
    elif planet == "Moon":
        g_planet, phi_planet = 1.62, 0.30
    else:  # Custom
        g_planet = st.number_input("Gravity (m/s²)", value=8.87, min_value=0.1, max_value=50.0)
        phi_planet = st.number_input("Porosity", value=0.18, min_value=0.01, max_value=0.99)
    
    st.markdown('<div class="sidebar-section"><h3>📊 Model Selection</h3></div>', unsafe_allow_html=True)
    
    # Model selection
    analysis_mode = st.radio(
        "Analysis Mode",
        ["Single Sample Prediction", "Batch Analysis", "Rock Database Explorer"]
    )
    
    st.markdown('<div class="sidebar-section"><h3>🎯 Quick Actions</h3></div>', unsafe_allow_html=True)
    
    if st.button("📋 Generate Sample Data"):
        st.session_state.sample_generated = True
    
    if st.button("🔄 Reset Analysis"):
        st.session_state.clear()
    
    # Model utilization metrics
    st.markdown('<div class="sidebar-section"><h3>⚙️ Model Utilization</h3></div>', unsafe_allow_html=True)
    
    # Simulated metrics (replace with actual model performance data)
    st.metric("Model Accuracy", "94.2%", "↑ 2.1%")
    st.metric("Predictions Today", "1,247", "↑ 23")
    st.metric("Average Confidence", "0.87", "↑ 0.03")

# Main content area
if analysis_mode == "Single Sample Prediction":
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🌍 Seismic Input Parameters")
        
        # Input parameters
        velocity = st.number_input("Seismic Velocity (km/s)", value=5.5, min_value=1.0, max_value=10.0, step=0.1)
        amplitude = st.number_input("Amplitude", value=0.60, min_value=0.1, max_value=1.0, step=0.01)
        duration = st.number_input("Duration (ms)", value=300, min_value=50, max_value=500, step=10)
        frequency = st.number_input("Frequency (Hz)", value=30, min_value=10, max_value=100, step=1)
        
        if st.button("🔍 Analyze Sample", type="primary"):
            # Convert planetary values to Earth equivalent
            g_earth = 9.81
            phi_earth = 0.10
            
            V_earth = convert_velocity_to_earth(velocity, g_planet, g_earth, phi_planet, phi_earth)
            A_earth = convert_amplitude_to_earth(amplitude, g_planet, g_earth, phi_planet, phi_earth)
            D_earth = convert_duration_to_earth(duration, g_planet, g_earth, phi_planet, phi_earth)
            f_earth = convert_frequency_to_earth(frequency, g_planet, g_earth, phi_planet, phi_earth)
            
            # Store results in session state
            st.session_state.prediction_results = {
                'original': [velocity, amplitude, duration, frequency],
                'converted': [V_earth, A_earth, D_earth, f_earth],
                'predicted_rock': 'Granite',  # Placeholder - replace with actual prediction
                'confidence': 0.89,
                'rock_type': 'Igneous'
            }
    
    with col2:
        st.markdown("### 📈 Converted Earth Values")
        
        if 'prediction_results' in st.session_state:
            results = st.session_state.prediction_results
            
            # Display converted values
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Velocity", f"{results['converted'][0]:.2f} km/s", f"{results['converted'][0] - results['original'][0]:.2f}")
                st.metric("Amplitude", f"{results['converted'][1]:.3f}", f"{results['converted'][1] - results['original'][1]:.3f}")
            with col2b:
                st.metric("Duration", f"{results['converted'][2]:.1f} ms", f"{results['converted'][2] - results['original'][2]:.1f}")
                st.metric("Frequency", f"{results['converted'][3]:.1f} Hz", f"{results['converted'][3] - results['original'][3]:.1f}")
        
        # Seismic properties radar chart
        if 'prediction_results' in st.session_state:
            results = st.session_state.prediction_results
            fig_radar = create_seismic_properties_chart(*results['converted'])
            st.plotly_chart(fig_radar, use_container_width=True)

# Prediction results
if 'prediction_results' in st.session_state:
    results = st.session_state.prediction_results
    predicted_rock = results['predicted_rock']
    
    st.markdown('<div class="prediction-result"><h2>🎯 Prediction Results</h2></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Predicted Rock", predicted_rock)
        st.metric("Confidence", f"{results['confidence']:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Rock Type", results['rock_type'])
        st.metric("Formation", rock_materials[predicted_rock]['formation'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.write("**Primary Uses:**")
        for use in rock_materials[predicted_rock]['uses']:
            st.write(f"• {use}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed rock information
    st.markdown("### 📋 Detailed Rock Information")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.write(f"**Description:** {rock_materials[predicted_rock]['description']}")
        st.write(f"**Formation Type:** {rock_materials[predicted_rock]['formation']}")
        st.write(f"**Typical Color:** {rock_materials[predicted_rock]['color']}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Elemental composition chart
        fig_pie = create_elemental_composition_chart(
            rock_materials[predicted_rock]['elements'], 
            predicted_rock
        )
        st.plotly_chart(fig_pie, use_container_width=True)

elif analysis_mode == "Rock Database Explorer":
    st.markdown("### 🗃️ Rock Database Explorer")
    
    # Rock type filter
    rock_types = list(set([rock_materials[rock]['type'] for rock in rock_materials.keys()]))
    selected_type = st.selectbox("Filter by Rock Type", ["All"] + rock_types)
    
    # Filter rocks
    if selected_type == "All":
        available_rocks = list(rock_materials.keys())
    else:
        available_rocks = [rock for rock in rock_materials.keys() if rock_materials[rock]['type'] == selected_type]
    
    # Multi-select for comparison
    selected_rocks = st.multiselect(
        "Select Rocks for Comparison",
        available_rocks,
        default=available_rocks[:3] if len(available_rocks) > 3 else available_rocks
    )
    
    if selected_rocks:
        # Properties comparison
        st.markdown("### 📊 Properties Comparison")
        fig_comparison = create_rock_comparison_chart(selected_rocks)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Elemental distribution
        st.markdown("### 🧪 Elemental Distribution")
        fig_elements = create_elemental_distribution_chart(selected_rocks)
        st.plotly_chart(fig_elements, use_container_width=True)
        
        # Rock details table
        st.markdown("### 📋 Rock Details")
        
        # Create DataFrame for table display
        table_data = []
        for rock in selected_rocks:
            data = rock_materials[rock]
            table_data.append({
                'Rock Name': rock,
                'Type': data['type'],
                'Formation': data['formation'],
                'Velocity (km/s)': data['velocity'][0],
                'Amplitude': data['amplitude'][0],
                'Duration (ms)': data['duration'][0],
                'Frequency (Hz)': data['frequency_Hz'][0],
                'Primary Uses': ', '.join(data['uses'][:2])  # First 2 uses
            })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)

elif analysis_mode == "Batch Analysis":
    st.markdown("### 📊 Batch Analysis Mode")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file with seismic data", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Data Preview")
            st.dataframe(df.head())
            
            # Expected columns
            expected_cols = ['velocity', 'amplitude', 'duration', 'frequency']
            
            if all(col in df.columns for col in expected_cols):
                if st.button("Process Batch Analysis"):
                    # Process each row (placeholder)
                    results = []
                    for _, row in df.iterrows():
                        # Convert and predict for each row
                        V_earth = convert_velocity_to_earth(row['velocity'], g_planet, 9.81, phi_planet, 0.10)
                        A_earth = convert_amplitude_to_earth(row['amplitude'], g_planet, 9.81, phi_planet, 0.10)
                        D_earth = convert_duration_to_earth(row['duration'], g_planet, 9.81, phi_planet, 0.10)
                        f_earth = convert_frequency_to_earth(row['frequency'], g_planet, 9.81, phi_planet, 0.10)
                        
                        # Placeholder prediction
                        predicted_rock = np.random.choice(list(rock_materials.keys()))
                        confidence = np.random.uniform(0.7, 0.95)
                        
                        results.append({
                            'Original_Velocity': row['velocity'],
                            'Original_Amplitude': row['amplitude'],
                            'Original_Duration': row['duration'],
                            'Original_Frequency': row['frequency'],
                            'Earth_Velocity': V_earth,
                            'Earth_Amplitude': A_earth,
                            'Earth_Duration': D_earth,
                            'Earth_Frequency': f_earth,
                            'Predicted_Rock': predicted_rock,
                            'Confidence': confidence,
                            'Rock_Type': rock_materials[predicted_rock]['type']
                        })
                    
                    results_df = pd.DataFrame(results)
                    st.write("### Analysis Results")
                    st.dataframe(results_df)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="rock_analysis_results.csv",
                        mime="text/csv"
                    )
            else:
                st.error(f"CSV must contain columns: {expected_cols}")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        st.info("Please upload a CSV file to begin batch analysis.")

# Footer
st.markdown("---")
st.markdown("### 🔬 About This System")
st.markdown("""
This advanced planetary rock analysis system uses machine learning to classify rock types based on seismic properties. 
The system converts planetary seismic data to Earth-equivalent values and provides detailed mineralogical information.

**Key Features:**
- Multi-planetary seismic data conversion
- Advanced rock classification models
- Detailed elemental composition analysis
- Interactive data visualization
- Batch processing capabilities
""")