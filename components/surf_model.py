import streamlit as st
import numpy as np
from tensorflow import keras
import joblib
import math

def prediction(session_key, ):
    if session_key == "igneous_rocks":

    else:
        st.write("Unknown material identified.")

def material_prediction():
    model=keras.models.load_model("models/model.h5")
    scaler = joblib.load("models/scaler.pkl")
    le = joblib.load("models/label_encoder.pkl")
    sample = np.array([[5.5, 0.57, 240]])
    sample_scaled = scaler.transform(sample)
    raw_prediction = model.predict(sample_scaled)
    predicted_index = np.argmax(raw_prediction)
    predicted_label = le.inverse_transform([predicted_index])[0]

    

