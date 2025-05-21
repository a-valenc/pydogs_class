import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torchvision import models
import torch.nn as nn

# Load class mapping
def load_class_mapping(path="class_mapping.txt"):
    mapping = {}
    with open(path, "r") as f:
        for line in f:
            key, val = line.strip().split(":")
            mapping[key] = val
    return mapping

label_to_name = load_class_mapping()
class_labels = list(label_to_name.keys())

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# Load model
@st.cache_resource
def load_model():
    num_classes = len(class_labels)
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("model_weights.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Streamlit UI
st.title("🐶 Dog Breed Classifier")

uploaded_file = st.file_uploader("Upload a dog image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_index = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_index].item()

    predicted_label = class_labels[predicted_index]
    predicted_name = label_to_name[predicted_label]

    st.markdown(f"### 🐾 Predicted Breed: **{predicted_name}**")
    st.markdown(f"Confidence Score: `{confidence:.2f}`")
