# Import required libraries
import streamlit as st              # For building the web interface
import pandas as pd                # For handling dataframes (CSV/Excel)
import duckdb                      # Lightweight database for SQL on DataFrames
import os                          # For file path and directory handling

# Define the upload function
def upload():
    """
    Uploads a CSV or Excel file using Streamlit, saves it to a DuckDB database,
    and displays file management options. Returns a Pandas DataFrame or None.
    """
    
    # Title of the Streamlit page
    st.title("📊 Streamlit Data Uploader")

    # ✅ Check if 'uploaded_files' is present in session state; if not, initialize it
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    # File uploader widget that accepts .csv, .xlsx, and .xls formats
    uploaded_file = st.file_uploader("📤 Upload CSV or Excel File", type=["csv", "xlsx", "xls"])

    # If a file has been uploaded
    if uploaded_file is not None:
        try:
            # ✅ Read uploaded file into a DataFrame based on file type
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # ✅ Add the file to session state only if it's not already uploaded
            if uploaded_file.name not in [f.name for f in st.session_state.uploaded_files]:
                st.session_state.uploaded_files.append(uploaded_file)
                st.success("✅ File uploaded successfully!")
            else:
                st.warning("⚠️ This file is already uploaded.")

            # ✅ Prepare DuckDB database file and table name
            file_name_with_ext = uploaded_file.name                  # Original filename (with extension)
            file_name, _ = os.path.splitext(file_name_with_ext)      # Extract base name without extension
            db_name = f"{file_name}.duckdb"                          # Set DuckDB filename
            db_path = os.path.join("data", db_name)                  # Full path in "data" directory
            table_name = file_name.replace(" ", "_")                 # Replace spaces in table name for SQL compatibility
            os.makedirs("data", exist_ok=True)                       # Create "data" folder if not exists

            # ✅ Connect to DuckDB and save the DataFrame as a table
            con = duckdb.connect(db_path)
            con.execute(f"DROP TABLE IF EXISTS {table_name}")        # Drop existing table if it exists
            con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")  # Create new table with uploaded data
            con.close()

            st.success(f"✅ Saved to `{db_path}` under table `{table_name}`")

            # ✅ Display list of uploaded files with delete button
            st.subheader("📁 Uploaded Files")
            for i, file in enumerate(st.session_state.uploaded_files):
                col1, col2 = st.columns([4, 1])                      # Layout: filename + delete button
                with col1:
                    st.write(file.name)                              # Show file name
                with col2:
                    if st.button("🗑️ Delete", key=f"delete_{i}"):
                        del st.session_state.uploaded_files[i]       # Remove file from session
                        st.experimental_rerun()                      # Refresh the app to update the list

            return df  # Return the DataFrame after successful upload

        except Exception as e:
            # ✅ Handle file reading errors
            st.error(f"❌ Error reading file: {e}")
            return None

    else:
        # Message shown when no file is uploaded yet
        st.info("ℹ️ Please upload a file to begin.")
        return None
