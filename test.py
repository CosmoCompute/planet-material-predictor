import numpy as np
from tensorflow.keras.models import load_model
import joblib

# ---- Load model, scaler, and label encoder ----
model = load_model("models/igneous_model.keras")
scaler = joblib.load("models/igneous_scaler.pkl")
le = joblib.load("models/igneous_label_encoder.pkl")

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

def predict_rock_type(velocity, amplitude, duration, frequency_hz, verbose=True):
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

# ---- Example Usage ----
if __name__ == "__main__":
    # Test with sample data
    # Format: velocity, amplitude, duration, frequency_hz
    velocity = 5.14
    amplitude = 0.53
    duration = 238
    frequency_hz = 29.75

    # Make prediction
    rock_type, confidence, probabilities = predict_rock_type(
        velocity, amplitude, duration, frequency_hz
    )

    print(f"\nFinal Result: {rock_type} (Confidence: {confidence:.3f})")