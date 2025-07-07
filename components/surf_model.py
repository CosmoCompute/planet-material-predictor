"""from tensorflow import keras
import streamlit as st
import numpy as np
import gdown
import os

@st.cache_resource
def load_model():
    file_id = "1gGKATiHtrowZUCb-kPVsfu6jpBP95Lle"
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = "model.h5"

    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

    model = keras.models.load_model(output_path)
    return model

model = load_model()
print("✅ Model loaded successfully!")

# Example prediction
sample = np.array([[2.61, 0.32, 175]])
prediction = model.predict(sample)

# Decode prediction
material_labels = [
    "igneous_rocks", "metamorphic_rocks", "sedimentary_rocks",
    "ore_and_industrial_minerals", "silicate_minerals",
    "evaporites_and_soft_minerals", "gem_and_rare_minerals"
]

predicted_class = np.argmax(prediction, axis=1)[0]
predicted_material = material_labels[predicted_class]

print("🪨 Predicted Material:", predicted_material)
"""