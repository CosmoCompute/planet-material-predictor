import streamlit as st
import numpy as np
from tensorflow import keras
import joblib
import math
from components import app_sidebar

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
    """
    Calculate all 19 engineered features from the 4 basic inputs
    """
    # Basic features
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
        for i, rock_type in enumerate(le.classes_):
            st.write(f"   {rock_type}: {pred_prob[0][i]:.3f}")

    return pred_label, confidence, pred_prob[0]

def material_prediction():
    app_sidebar.surface_material_page_sidebar()
    V_venus = 5.5       # km/s
    A_venus = 0.60       # amplitude
    D_venus = 300       # ms
    f_venus = 30        # Hz

    g_planet = 3.73      # m/s^2
    g_earth = 9.81      # m/s^2
    phi_planet = 0.18
    phi_earth = 0.10

# Convert all properties
    V_earth = convert_velocity_to_earth(V_venus, g_planet, g_earth, phi_planet, phi_earth)
    A_earth = convert_amplitude_to_earth(A_venus, g_planet, g_earth, phi_planet, phi_earth)
    D_earth = convert_duration_to_earth(D_venus, g_planet, g_earth, phi_planet, phi_earth)
    f_earth = convert_frequency_to_earth(f_venus, g_planet, g_earth, phi_planet, phi_earth)

    sample = np.array([V_earth, A_earth, D_earth])

    model = keras.models.load_model("models/model.h5")
    scaler = joblib.load("models/scaler.pkl")
    le = joblib.load("models/label_encoder.pkl")

    sample = np.array(sample).reshape(1, -1)  # FIXED: Reshape to 2D
    sample_scaled = scaler.transform(sample)

    raw_prediction = model.predict(sample_scaled)
    predicted_index = np.argmax(raw_prediction)
    predicted_label = le.inverse_transform([predicted_index])[0]

    st.write(predicted_label)

    sample = np.array([V_earth, A_earth, D_earth, f_earth])

    if predicted_label == "igneous_rocks":
        ig_model=keras.models.load_model("models/igneous_model.keras")
        ig_scaler=joblib.load("models/igneous_scaler.pkl")
        ig_le=joblib.load("models/igneous_label_encoder.pkl")
        rock_type, confidence, probabilities = predict_rock_type(
            ig_scaler, ig_model, ig_le,V_earth, A_earth, D_earth, f_earth
        )

        st.write(f"\nFinal Result: {rock_type} (Confidence: {confidence:.3f})")
    
    elif predicted_label == "metamorphic_rocks":
        ig_model=keras.models.load_model("models/metamorphic_model.keras")
        ig_scaler=joblib.load("models/metamorphic_scaler.pkl")
        ig_le=joblib.load("models/metamorphic_label_encoder.pkl")
        rock_type, confidence, probabilities = predict_rock_type(
            ig_scaler, ig_model, ig_le,V_earth, A_earth, D_earth, f_earth
        )

        st.write(f"\nFinal Result: {rock_type} (Confidence: {confidence:.3f})")