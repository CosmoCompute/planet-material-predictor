import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

MODEL_DIR = 'models'
DATA_DIR = 'data'

# Add root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from utils import db_utils
from models.temp_model import load_model

def data_anly():
    file_name = "Earth.duckdb"
    model_path = "temp_model.pkl"  # Local path to temp model

    # Load data
    df = db_utils.load_db(file_name)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df.set_index('Date', inplace=True)
    y = df['Maximum Temperature']

    # Load model
    model = load_model()

    # Predict
    forecast = model.predict(start=0, end=len(y)-1)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(y, label='Actual')
    plt.plot(forecast, label='Forecast', linestyle='--')
    plt.title('SARIMAX Forecast vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Maximum Temperature')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

    # Delete the model file to keep environment clean
    if os.path.exists(model_path):
        os.remove(model_path)
        st.info("Temporary model file deleted after use.")
