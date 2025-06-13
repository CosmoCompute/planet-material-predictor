import duckdb
import os

def load_db(file_name):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(project_root, 'data', file_name)

    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Database file not found: {data_path}")

    con = duckdb.connect(data_path)
    table_name, _ = os.path.splitext(file_name)
    df = con.execute(f"SELECT * FROM {table_name}").fetchdf()
    return df
