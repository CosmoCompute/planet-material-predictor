import streamlit as st
import pandas as pd

def upload():
    """
    Uploads a CSV or Excel file using Streamlit and displays the DataFrame.
    
    Returns:
        pd.DataFrame or None
    """

    col1, col2 = st.columns([2,1])

    with col1:
        uploaded_file = st.file_uploader("📤 Upload CSV or Excel File", type=["csv", "xlsx", "xls"])

        if uploaded_file is not None:
            try:
            # Read the uploaded file into a DataFrame
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

            # Show success message
                st.success("✅ File uploaded successfully!")
                

                return df

            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
                return None
        else:
            st.info("Please upload a file to begin.")
            return None