import streamlit as st

# Use Streamlit's cache_resource decorator to avoid re-downloading/loading the model each time
@st.cache_resource
def load_model():
    """
    Downloads and loads a pre-trained model from Google Drive using gdown,
    caches it using Streamlit's resource cache, and returns the loaded model.
    
    Returns:
        model (any): The deserialized model object loaded using pickle.
    """

    # Import required libraries inside the function (keeps dependencies isolated when cached)
    import gdown     # For downloading files from Google Drive
    import pickle    # For loading the serialized model
    import os        # For checking file existence

    # Google Drive file ID and direct download URL
    file_id = "1EW1y5ULRBOZ8FazZI_LB7UmK5XZjjzi9"
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = "temp_model.pkl"  # File will be saved locally with this name

    # Download the model file only if it doesn't already exist
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

    # Load the model from the downloaded file using pickle
    with open(output_path, "rb") as f:
        model = pickle.load(f)

    # Return the loaded model object
    return model

