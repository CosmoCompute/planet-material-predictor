from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import pickle
import sys
import os

# Get the path to the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the root directory (one level up)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from utils import db_utils

MODEL_DIR = os.path.join(project_root, 'models')

df=db_utils.load_db("Earth.duckdb")
df['Date']=pd.to_datetime(df['Date'], dayfirst=True)
df.set_index('Date', inplace=True)

y=df['Maximum Temperature']

model = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12))
results = model.fit(disp=False)

MODEL_PATH = os.path.join(MODEL_DIR, 'temp_model.pkl')
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(results, f)

print("Model trained and saved successfully.")
