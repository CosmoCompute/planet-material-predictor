import streamlit as st
import numpy as np
from tensorflow import keras
import joblib
import math
import duckdb
import os
import plotly.express as px
import pandas as pd

def scaling_factor(g_planet, g_earth, phi_planet, phi_earth, alpha=0.2, beta=0.5):
    """Common scaling factor based on gravity and porosity."""
    gravity_term = (g_earth / g_planet) ** alpha
    porosity_term = ((1 - phi_earth) / (1 - phi_planet)) ** beta
    return gravity_term * porosity_term

def convert_velocity_to_earth(V_planet, g_planet, g_earth, phi_planet, phi_earth,
                              alpha=0.2, beta=0.5):
    factor = scaling_factor(g_planet, g_earth, phi_planet, phi_earth, alpha, beta)
    return V_planet * factor

def convert_amplitude_to_earth(A_planet, g_planet, g_earth, phi_planet, phi_earth,
                               alpha=0.2, beta=0.5):
    factor = scaling_factor(g_planet, g_earth, phi_planet, phi_earth, alpha, beta)
    return A_planet * factor

def convert_frequency_to_earth(f_planet, g_planet, g_earth, phi_planet, phi_earth,
                               alpha=0.2, beta=0.5):
    factor = scaling_factor(g_planet, g_earth, phi_planet, phi_earth, alpha, beta)
    return f_planet * factor

def convert_duration_to_earth(D_planet, g_planet, g_earth, phi_planet, phi_earth,
                              alpha=0.2, beta=0.5):
    factor = scaling_factor(g_planet, g_earth, phi_planet, phi_earth, alpha, beta)
    return D_planet / factor 

def engineer_features(velocity, amplitude, duration, frequency_hz):
    features = [velocity, amplitude, duration, frequency_hz]

    # Engineered features (same as in training)
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

    # Combine all features in the same order as training
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
    """
    Predict rock type from basic seismic properties

    Args:
        velocity: Seismic velocity (km/s)
        amplitude: Amplitude value
        duration: Duration (ms)
        frequency_hz: Frequency (Hz)
        verbose: Whether to print detailed output

    Returns:
        tuple: (predicted_rock_type, confidence, all_probabilities)
    """

    # Engineer all features
    sample_features = engineer_features(velocity, amplitude, duration, frequency_hz)

    # Scale the features
    sample_scaled = scaler.transform(sample_features)

    # Predict
    pred_prob = model.predict(sample_scaled, verbose=0)
    pred_index = np.argmax(pred_prob)
    pred_label = le.inverse_transform([pred_index])[0]
    confidence = np.max(pred_prob)

    if verbose:
        print(f"\n🪨 Input Sample:")
        print(f"   Velocity: {velocity} km/s")
        print(f"   Amplitude: {amplitude}")
        print(f"   Duration: {duration} ms")
        print(f"   Frequency: {frequency_hz} Hz")
        print(f"\n📊 Prediction Results:")
        print(f"   Predicted Rock Type: {pred_label}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"\n📈 All Probabilities:")
        for i, rock_type in enumerate(le.classes_):
            print(f"   {rock_type}: {pred_prob[0][i]:.3f}")

    return pred_label, confidence, pred_prob[0]

def check_database_files(database_name, surf_data_dir="data\data_surf", label_dir="data\label"):
    """
    Check if the required database files exist in the specified directories.
    
    Args:
        database_name: Name of the database (without .duckdb extension)
        surf_data_dir: Directory containing surface data files (default: "db/surf_data")
        label_dir: Directory containing label data files (default: "db/label")
    
    Returns:
        tuple: (data_surf_exists, label_exists, data_surf_path, label_path)
    """
    data_surf_path = os.path.join(surf_data_dir, f"{database_name}.duckdb")
    label_path = os.path.join(label_dir, f"{database_name}.duckdb")
    
    data_surf_exists = os.path.exists(data_surf_path)
    label_exists = os.path.exists(label_path)
    
    return data_surf_exists, label_exists, data_surf_path, label_path

def get_surface_data(db_path):
    """Get surface data from mars_surface table."""
    try:
        conn = duckdb.connect(db_path)
        query = """
        SELECT 
            rock_type,
            distribution_percent,
            velocity_km_s_mean,
            velocity_km_s_std,
            amplitude_mean,
            amplitude_std,
            duration_ms_mean,
            duration_ms_std,
            frequency_Hz_mean,
            frequency_Hz_std
        FROM mars_surface
        """
        result = conn.execute(query).fetchall()
        columns = [desc[0] for desc in conn.description]
        conn.close()
        return pd.DataFrame(result, columns=columns)
    except Exception as e:
        st.error(f"Error reading surface data: {str(e)}")
        return None

def get_planet_parameters(db_path):
    """Get planet parameters from planet_parameters table."""
    try:
        conn = duckdb.connect(db_path)
        query = """
        SELECT 
            g_force,
            phi_index
        FROM planet_parameters
        ORDER BY timestamp DESC
        LIMIT 1
        """
        result = conn.execute(query).fetchone()
        conn.close()
        if result:
            return result[0], result[1]  # g_force, phi_index
        return None, None
    except Exception as e:
        st.error(f"Error reading planet parameters: {str(e)}")
        return None, None

def create_pie_chart(probabilities, rock_types, title="Material Distribution"):
    """Create a pie chart for material distribution."""
    # Filter out zero probabilities
    non_zero_data = [(prob, rock_type) for prob, rock_type in zip(probabilities, rock_types) if prob > 0]
    
    if not non_zero_data:
        st.warning("No materials with non-zero probabilities found.")
        return
    
    probs, types = zip(*non_zero_data)
    
    df = pd.DataFrame({
        'Material': types,
        'Probability': probs
    })
    
    fig = px.pie(df, values='Probability', names='Material', 
                 title=title,
                 color_discrete_sequence=px.colors.qualitative.Set3)
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

def material_prediction():
    st.title("🪨 Material Prediction from Planetary Data")
    
    # Configuration section for database paths
    st.subheader("📁 Database Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        surf_data_dir = st.text_input("Surface Data Directory:", value="db/surf_data")
    
    with col2:
        label_dir = st.text_input("Label Data Directory:", value="db/label")
    
    # Input for database name
    database_name = st.text_input("Enter database name:", value="mars_data")
    
    # Alternative: Single directory input (if both files are in the same directory)
    st.subheader("📂 Alternative: Single Directory")
    single_dir = st.text_input("Or enter single directory path (if both files are in same location):", value="")
    
    if st.button("Run Material Prediction"):
        # Use single directory if provided, otherwise use separate directories
        if single_dir.strip():
            surf_data_dir = single_dir.strip()
            label_dir = single_dir.strip()
        
        # Check if database files exist
        data_surf_exists, label_exists, data_surf_path, label_path = check_database_files(
            database_name, surf_data_dir, label_dir
        )
        
        if not data_surf_exists:
            st.error(f"❌ Surface data file not found: {data_surf_path}")
            st.info(f"Expected path: {data_surf_path}")
            return
        
        if not label_exists:
            st.error(f"❌ Label data file not found: {label_path}")
            st.info(f"Expected path: {label_path}")
            return
        
        st.success("✅ Database files found successfully!")
        st.info(f"📍 Surface data: {data_surf_path}")
        st.info(f"📍 Label data: {label_path}")
        
        # Get surface data
        surface_data = get_surface_data(data_surf_path)
        if surface_data is None:
            return
        
        # Get planet parameters
        g_planet, phi_planet = get_planet_parameters(label_path)
        if g_planet is None or phi_planet is None:
            st.error("❌ Could not retrieve planet parameters")
            return
        
        st.info(f"🌍 Planet Parameters: g_force = {g_planet:.2f} m/s², phi_index = {phi_planet:.2f}")
        
        # Display surface data
        st.subheader("📊 Surface Data")
        st.dataframe(surface_data)
        
        # Process all rows of surface data for prediction
        if len(surface_data) > 0:
            # Earth reference values
            g_earth = 9.81      # m/s^2
            phi_earth = 0.10
            
            st.subheader("🔄 Processing All Surface Data Samples")
            
            # Load main classification model
            try:
                model = keras.models.load_model("models/model.h5")
                scaler = joblib.load("models/scaler.pkl")
                le = joblib.load("models/label_encoder.pkl")
                
                # Load all secondary models
                model_files = {
                    "igneous_rocks": ("models/igneous_model.keras", "models/igneous_scaler.pkl", "models/igneous_label_encoder.pkl"),
                    "metamorphic_rocks": ("models/metamorphic_model.keras", "models/metamorphic_scaler.pkl", "models/metamorphic_label_encoder.pkl"),
                    "sedimentary_rocks": ("models/sedimentary_model.keras", "models/sedimentary_scaler.pkl", "models/sedimentary_label_encoder.pkl"),
                    "ore_and_industrial_minerals": ("models/Ore and Industrial Minerals_model.keras", "models/Ore and Industrial Minerals_model_scaler.pkl", "models/Ore and Industrial Minerals_model_label_encoder.pkl"),
                    "silicate_minerals": ("models/Silicate_model.keras", "models/Silicate_scaler.pkl", "models/Silicate_label_encoder.pkl"),
                    "evaporites_and_soft_minerals": ("models/Evaporites and Soft Minerals_model.keras", "models/Evaporites and Soft Minerals_scaler.pkl", "models/Evaporites and Soft Minerals_label_encoder.pkl"),
                    "gem_and_rare_minerals": ("models/Gem and Rare Minerals_model.keras", "models/Gem and Rare Minerals_scaler.pkl", "models/Gem and Rare Minerals_label_encoder.pkl")
                }
                
                # Store all predictions
                all_predictions = []
                conversion_data = []
                detailed_predictions = []
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, row in surface_data.iterrows():
                    status_text.text(f"Processing sample {idx + 1}/{len(surface_data)}: {row['rock_type']}")
                    
                    # Set planetary values from current row
                    V_planet = row['velocity_km_s_mean']
                    A_planet = row['amplitude_mean'] 
                    D_planet = row['duration_ms_mean']
                    f_planet = row['frequency_Hz_mean']
                    
                    # Convert all properties to Earth equivalent
                    V_earth = convert_velocity_to_earth(V_planet, g_planet, g_earth, phi_planet, phi_earth)
                    A_earth = convert_amplitude_to_earth(A_planet, g_planet, g_earth, phi_planet, phi_earth)
                    D_earth = convert_duration_to_earth(D_planet, g_planet, g_earth, phi_planet, phi_earth)
                    f_earth = convert_frequency_to_earth(f_planet, g_planet, g_earth, phi_planet, phi_earth)
                    
                    # Store conversion data
                    conversion_data.append({
                        'Original_Rock_Type': row['rock_type'],
                        'Distribution_%': row['distribution_percent'],
                        'V_planet': V_planet,
                        'A_planet': A_planet,
                        'D_planet': D_planet,
                        'f_planet': f_planet,
                        'V_earth': V_earth,
                        'A_earth': A_earth,
                        'D_earth': D_earth,
                        'f_earth': f_earth
                    })
                    
                    # Prepare sample for main classification
                    sample = np.array([V_earth, A_earth, D_earth]).reshape(1, -1)
                    sample_scaled = scaler.transform(sample)
                    
                    # Main prediction
                    raw_prediction = model.predict(sample_scaled, verbose=0)
                    predicted_index = np.argmax(raw_prediction)
                    predicted_label = le.inverse_transform([predicted_index])[0]
                    main_confidence = np.max(raw_prediction)
                    
                    # Store main prediction
                    prediction_data = {
                        'Original_Rock_Type': row['rock_type'],
                        'Distribution_%': row['distribution_percent'],
                        'Predicted_Category': predicted_label,
                        'Main_Confidence': main_confidence
                    }
                    
                    # Add main category probabilities
                    for i, category in enumerate(le.classes_):
                        prediction_data[f'Prob_{category}'] = raw_prediction[0][i]
                    
                    # Secondary classification if model exists
                    if predicted_label in model_files:
                        model_path, scaler_path, le_path = model_files[predicted_label]
                        
                        try:
                            # Load secondary model
                            secondary_model = keras.models.load_model(model_path)
                            secondary_scaler = joblib.load(scaler_path)
                            secondary_le = joblib.load(le_path)
                            
                            # Predict specific rock type
                            rock_type, confidence, probabilities = predict_rock_type(
                                secondary_scaler, secondary_model, secondary_le, 
                                V_earth, A_earth, D_earth, f_earth, verbose=False
                            )
                            
                            prediction_data['Detailed_Prediction'] = rock_type
                            prediction_data['Detailed_Confidence'] = confidence
                            
                            # Store detailed prediction
                            detailed_pred = {
                                'Original_Rock_Type': row['rock_type'],
                                'Distribution_%': row['distribution_percent'],
                                'Main_Category': predicted_label,
                                'Detailed_Prediction': rock_type,
                                'Confidence': confidence
                            }
                            
                            # Add detailed probabilities
                            for i, specific_type in enumerate(secondary_le.classes_):
                                detailed_pred[f'Prob_{specific_type}'] = probabilities[i]
                            
                            detailed_predictions.append(detailed_pred)
                            
                        except Exception as e:
                            prediction_data['Detailed_Prediction'] = f"Error: {str(e)}"
                            prediction_data['Detailed_Confidence'] = 0.0
                    else:
                        prediction_data['Detailed_Prediction'] = "No secondary model"
                        prediction_data['Detailed_Confidence'] = 0.0
                    
                    all_predictions.append(prediction_data)
                    
                    # Update progress
                    progress_bar.progress((idx + 1) / len(surface_data))
                
                status_text.text("Processing complete!")
                
                # Convert to DataFrames
                conversion_df = pd.DataFrame(conversion_data)
                predictions_df = pd.DataFrame(all_predictions)
                
                # Display results
                st.subheader("📊 Conversion Results")
                st.dataframe(conversion_df)
                
                st.subheader("🎯 Prediction Results")
                st.dataframe(predictions_df)
                
                # Aggregate predictions for visualization
                st.subheader("📈 Aggregated Prediction Analysis")
                
                # Main category distribution
                main_category_counts = predictions_df['Predicted_Category'].value_counts()
                st.write("**Main Category Distribution:**")
                fig_main = px.pie(values=main_category_counts.values, names=main_category_counts.index,
                                 title="Distribution of Main Categories Across All Samples")
                st.plotly_chart(fig_main, use_container_width=True)
                
                # Detailed predictions distribution
                if detailed_predictions:
                    detailed_df = pd.DataFrame(detailed_predictions)
                    detailed_counts = detailed_df['Detailed_Prediction'].value_counts()
                    st.write("**Detailed Prediction Distribution:**")
                    fig_detailed = px.pie(values=detailed_counts.values, names=detailed_counts.index,
                                         title="Distribution of Detailed Predictions Across All Samples")
                    st.plotly_chart(fig_detailed, use_container_width=True)
                
                # Confidence analysis
                st.subheader("📊 Confidence Analysis")
                avg_main_confidence = predictions_df['Main_Confidence'].mean()
                st.metric("Average Main Classification Confidence", f"{avg_main_confidence:.3f}")
                
                if 'Detailed_Confidence' in predictions_df.columns:
                    detailed_conf_df = predictions_df[predictions_df['Detailed_Confidence'] > 0]
                    if len(detailed_conf_df) > 0:
                        avg_detailed_confidence = detailed_conf_df['Detailed_Confidence'].mean()
                        st.metric("Average Detailed Classification Confidence", f"{avg_detailed_confidence:.3f}")
                
                # Summary statistics
                st.subheader("📋 Summary Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Samples Processed", len(surface_data))
                
                with col2:
                    st.metric("Unique Main Categories", len(main_category_counts))
                
                with col3:
                    if detailed_predictions:
                        st.metric("Unique Detailed Predictions", len(detailed_counts))
                    else:
                        st.metric("Unique Detailed Predictions", 0)
                
            except Exception as e:
                st.error(f"Error loading models: {str(e)}")
                return
        
        else:
            st.warning("No surface data available for prediction.")

# Streamlit app
if __name__ == "__main__":
    material_prediction()
