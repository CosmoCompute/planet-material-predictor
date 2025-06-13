import duckdb
import os

def load_db(file_name):
    DATA_DIR = 'data'

    data_path=os.path.join(DATA_DIR, file_name)
    con=duckdb.connect(data_path)
    table_name, _=os.path.splitext(file_name)
    df=con.execute(f"SELECT * FROM {table_name}").fetchdf()

    return df