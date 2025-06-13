import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import os
from utils import db_utils as du
from models import temp_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

def data_analysis():
    st.title("🪐 Universal Planet Temperature Forecast")

    # Dynamic Planet Selection
    planets = du.list_planet_dbs()
    if not planets:
        st.warning("No DuckDB databases found in 'data/'.")
        return

    planet = st.selectbox("🌌 Select a Planet Database", planets)

    # Load data
    df = du.load_planet_data(planet)
    st.success(f"✅ Loaded {len(df)} rows for {planet}")

    # Select target variable
    columns = ['min_temp', 'max_temp', 'wind_speed', 'pressure']
    columns = [col for col in columns if col in df.columns]
    if not columns:
        st.error("❌ No valid data columns found.")
        return

    target_col = st.selectbox("📊 Select variable to forecast", columns)

    # Forecast horizon
    forecast_days = st.slider("📆 Forecast Days", min_value=7, max_value=90, value=30, step=7)

    # Train SARIMAX
    model, forecast_df = temp_model(df, column=target_col, forecast_days=forecast_days)

    # Evaluation Metrics
    mae = mean_absolute_error(forecast_df['actual'], forecast_df['forecast'])
    rmse = mean_squared_error(forecast_df['actual'], forecast_df['forecast'], squared=False)
    st.subheader("📉 Evaluation Metrics")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")

    # Plot forecast
    st.subheader("📈 Forecast Visualization")
    fig, ax = plt.subplots(figsize=(10, 4))
    df[target_col].plot(ax=ax, label="Historical", color='blue')
    forecast_df['forecast'].plot(ax=ax, label="Forecast", color='orange')
    forecast_df['actual'].plot(ax=ax, label="Actual", color='green')
    ax.fill_between(forecast_df.index,
                    forecast_df.iloc[:, 0],  # lower bound
                    forecast_df.iloc[:, 1],  # upper bound
                    color='lightgrey', alpha=0.3)
    ax.set_title(f"{planet} - {target_col} Forecast ({forecast_days} Days)")
    ax.legend()
    st.pyplot(fig)

    # Save and Download Model
    st.subheader("💾 Save & Download Model")

    if st.button("✅ Save Model"):
        model_dir = "saved_models"
        os.makedirs(model_dir, exist_ok=True)
        filename = f"{model_dir}/{planet.lower()}_{target_col}_sarimax.pkl"
        joblib.dump(model, filename)
        st.success(f"Model saved as `{filename}`")

        with open(filename, "rb") as f:
            st.download_button(
                label="📥 Download Model (.pkl)",
                data=f,
                file_name=os.path.basename(filename),
                mime="application/octet-stream"
            )
