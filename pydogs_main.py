import streamlit as st
import numpy as np
import keras
import tensorflow as tf
from PIL import Image
#load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('dog_model_saved')

model = load_model()
# Load mapping from text file
def load_class_mapping(path="class_mapping.txt"):
    mapping = {}
    with open(path, "r") as f:
        for line in f:
            key, val = line.strip().split(":")
            mapping[key] = val
    return mapping

label_to_name = load_class_mapping()
class_labels = list(label_to_name.keys())

st.title("🐶 Dog Breed Classifier")

uploaded_file = st.file_uploader("Upload a dog image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))  # adjust based on your model
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = class_labels[predicted_index]
    predicted_name = label_to_name[predicted_label]
    confidence = np.max(prediction)

    st.markdown(f"### 🐾 Predicted Breed: **{predicted_name}**")
    st.markdown(f"Confidence Score: `{confidence:.2f}`")
