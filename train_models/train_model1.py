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
 # if db_utils.py is directly in utils/


df=db_utils.load_db("Earth.duckdb")
df['Date']=pd.to_datetime(df['Date'], dayfirst=True)

print(df['Date'])