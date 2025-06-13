import streamlit as st
import pandas as pd

def upload():
    """
    Uploads a CSV or Excel file using Streamlit and displays the DataFrame.
    Returns:
        pd.DataFrame or None
    """
    st.title("📊 Streamlit Data Uploader")

    # ✅ Initialize session state for uploaded_files
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    uploaded_file = st.file_uploader("📤 Upload CSV or Excel File", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            # ✅ Read uploaded file
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # ✅ Add to session state if not a duplicate
            if uploaded_file.name not in [f.name for f in st.session_state.uploaded_files]:
                st.session_state.uploaded_files.append(uploaded_file)
                st.success("✅ File uploaded successfully!")
            else:
                st.warning("⚠️ This file is already uploaded.")

            # ✅ Show uploaded files with delete option
            st.subheader("📁 Uploaded Files")
            for i, file in enumerate(st.session_state.uploaded_files):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(file.name)
                with col2:
                    if st.button("🗑️ Delete", key=f"delete_{i}"):
                        del st.session_state.uploaded_files[i]
                        st.experimental_rerun()

            return df

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            return None

    else:
        st.info("ℹ️ Please upload a file to begin.")
        return None