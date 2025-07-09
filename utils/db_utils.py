import duckdb
import os

def load_db(file_name, session_key):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if session_key == "temp":
        data_path = os.path.join(project_root, 'data/data_temp/', file_name)
    elif session_key == "surf":
        data_path = os.path.join(project_root, 'data/data_surf/', file_name)

    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Database file not found: {data_path}")

    con = duckdb.connect(data_path)
    table_name, _ = os.path.splitext(file_name)
    df = con.execute(f"SELECT * FROM {table_name}").fetchdf()
    return df
