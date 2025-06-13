import streamlit as st

@st.cache_resource
def load_model():
    import gdown, pickle, os
    file_id = "1nrItW1IYiuFGMJ4i3fDjhsjCqxctLZsW"
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = "temp_model.pkl"

    # Download only if not already downloaded
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

    with open(output_path, "rb") as f:
        model = pickle.load(f)

    return model
