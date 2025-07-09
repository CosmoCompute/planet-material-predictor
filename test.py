import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate date range from 2017 to 2022
dates = pd.date_range(start='2017-01-01', end='2022-12-31', freq='D')
num_days = len(dates)

# Generate synthetic data
min_temp = np.random.uniform(450, 470, num_days).round(1)          # °C
max_temp = np.random.uniform(460, 480, num_days).round(1)          # °C
pressure = np.random.uniform(9100, 9500, num_days).round(1)        # hPa
wind_speed = np.random.randint(2, 11, num_days)                    # m/s

# Build DataFrame
venus_df = pd.DataFrame({
    "Date": dates,
    "Minimum": min_temp,
    "Maximum": max_temp,
    "Pressure": pressure,
    "Wind Speed": wind_speed
})

# Optional: Save to CSV
venus_df.to_csv("venus_2017_2022_synthetic_data.csv", index=False)

# Display first few rows
print(venus_df.head())
