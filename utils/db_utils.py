import duckdb
import os
import pandas as pd

DATA_DIR="data"

def list_planet_dbs():
    #i want to return a list
    dbs=[f[:-7].capitalize() for f in os.listdir(DATA_DIR) if f.endswith(".duckdb")]
    return sorted(dbs)

def load_planet_data(planet):
    """
    Loads data from the planet's DuckDB database.
    Assumes table name is planet
    """
    db_path = os.path.join(DATA_DIR, f"{planet}.duckdb")
    table_name = f"{planet}"

    con = duckdb.connect(database=db_path, read_only=True)
    df = con.execute(f"SELECT * FROM {table_name} ORDER BY date").df()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df