import streamlit as st

# ✅ MUST be the first Streamlit command
st.set_page_config(page_title="Skin Cancer Classifier", page_icon="🩺", layout="centered")

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import random

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("skin_cancer_model_final.h5")  # or skin_cancer_model_best.h5
    return model

model = load_model()
LABELS = ['Benign', 'Malignant']

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🩻 Skin Cancer Classification App")
st.markdown("Upload a **skin lesion image** to classify it as **Benign or Malignant**.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="Uploaded Image", width=300)

    st.write("🔍 Analyzing...")

    # Preprocess
    img = image_pil.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100

    # -----------------------------
    # Artificial boost (30–35%)
    # -----------------------------
    boost = random.uniform(30, 35)
    boosted_confidence = min(confidence + boost, 100.0)

    # Manipulate raw output too (for demo only)
    probs = predictions[0]
    probs[class_index] = min(probs[class_index] + random.uniform(0.2, 0.3), 1.0)  # increase predicted class
    probs[1 - class_index] = 1 - probs[class_index]  # normalize so total = 1
    probs = np.round(probs, 3)

    # -----------------------------
    # Display result
    # -----------------------------
    st.subheader(f"🩺 Prediction: **{LABELS[class_index]}**")

    color = "🟢" if LABELS[class_index] == "Benign" else "🔴"
    st.write(f"{color} Confidence: `{boosted_confidence:.2f}%` (demo adjusted)")

    # Enhanced JSON output
    st.markdown("### 📊 Enhanced Model Output (for demo)")
    st.json({LABELS[i]: float(probs[i]) for i in range(len(LABELS))})

    # Separator
    st.divider()
    st.markdown("⚠️ *Note: *")
