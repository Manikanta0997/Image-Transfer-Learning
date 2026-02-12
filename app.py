import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# -----------------------------------
# Page Config
# -----------------------------------
st.set_page_config(page_title="Image Classifier", page_icon="🧠", layout="centered")

st.title("🧠 EfficientNetV2 Image Classifier")
st.markdown("### Drag & Drop an Image Below 👇")


# -----------------------------------
# Load Model (Cached)
# -----------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("efficientnetv2_model_Animal_Faces.keras")


model = load_model()

# -----------------------------------
# Load Class Names
# -----------------------------------
with open("class_names.json", "r") as f:
    class_names = json.load(f)


# -----------------------------------
# Prediction Function
# -----------------------------------
def predict_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = np.max(predictions[0])

    return predicted_class, confidence


# -----------------------------------
# Drag & Drop Upload Area
# -----------------------------------
uploaded_file = st.file_uploader(
    "Drag and drop image here or click to browse", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):

        with st.spinner("Analyzing Image..."):
            predicted_class, confidence = predict_image(image)

        st.success("Prediction Complete ✅")

        st.markdown(f"## 🎯 Prediction: **{predicted_class}**")
        st.markdown(f"### 📊 Confidence: `{confidence:.4f}`")
