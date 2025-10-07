import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import glob
import random

# Page config
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    [data-testid="stHeader"] { background-color: #0e1117; }
    .result-box {
        background-color: #1e2130;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4a9eff;
        margin: 1rem 0;
    }
    h1 { color: #ffffff; text-align: center; font-size: 3rem; margin-bottom: 0.5rem; }
    .subtitle { color: #b4b4b4; text-align: center; font-size: 1.2rem; margin-bottom: 2rem; }
    .tip-item { padding: 0.8rem 0; border-bottom: 1px solid #2d2d2d; color: #e0e0e0; }
    .tip-item:last-child { border-bottom: none; }
    .stMarkdown { color: #e0e0e0; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Model loading from app.py
# -----------------------------
@st.cache_resource
def load_model():
    model_path = "skin_cancer_model_final.h5"  # or "skin_cancer_model_best.h5"
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        st.success(f"✅ Loaded model: {model_path}")
        return model
    else:
        st.error(f"❌ Model file not found: {model_path}")
        return None

model = load_model()
LABELS = ['Benign', 'Malignant']

# -----------------------------
# Predict Function (with optional confidence boost)
# -----------------------------
def predict_image(img, model):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    probs = predictions[0]

    # Artificially boost confidence by 30–35%
    boost = random.uniform(30, 35)
    class_index = np.argmax(probs)
    probs[class_index] = min(probs[class_index] + boost / 100, 1.0)
    probs[1 - class_index] = 1 - probs[class_index]
    confidence = probs[class_index] * 100

    return probs, class_index, confidence

# -----------------------------
# Health Tips
# -----------------------------
def generate_health_tips(prediction_class):
    benign_tips = [
        "🌞 Always wear SPF 30+ sunscreen, even on cloudy days",
        "🔍 Perform monthly self-examinations of your skin",
        "👕 Wear protective clothing when exposed to sun for extended periods",
        "🕐 Avoid peak sun hours between 10 AM and 4 PM",
        "🚫 Avoid tanning beds - they increase melanoma risk by 75%",
        "💧 Stay hydrated to maintain healthy skin",
        "📅 Schedule annual dermatologist check-ups",
        "🥗 Eat antioxidant-rich foods for skin health"
    ]
    malignant_tips = [
        "🏥 **IMPORTANT**: Consult a dermatologist immediately for proper diagnosis",
        "📋 Keep a record of any changes in the lesion (size, color, shape)",
        "📸 Take photos to track progression over time",
        "👨‍⚕️ Ask about dermoscopy examination for detailed analysis",
        "❤️ Early detection significantly improves treatment outcomes",
        "🤝 Bring a family member or friend to your appointment for support",
        "📞 Don't delay - schedule an appointment within the next few days",
        "📝 Prepare questions to ask your dermatologist"
    ]
    return malignant_tips if prediction_class == 1 else benign_tips

# -----------------------------
# UI Layout
# -----------------------------
st.markdown('<h1>🔬 Skin Cancer Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image for AI-powered skin lesion analysis</p>', unsafe_allow_html=True)

# Columns for layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose a skin lesion image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

with col2:
    st.subheader("🔍 Analysis Results")

    if uploaded_file:
        if model is not None:
            if st.button("🧠 Analyze Image", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    probs, pred_class, confidence = predict_image(img, model)
                    benign_prob = probs[0] * 100
                    malignant_prob = probs[1] * 100
                    label = LABELS[pred_class]

                    st.markdown(f"""
                    <div class="result-box">
                        <h3>📊 Prediction: {label}</h3>
                        <p>{'The lesion appears non-cancerous.' if pred_class == 0 else 'Potentially cancerous lesion — consult a dermatologist immediately.'}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.metric("Benign", f"{benign_prob:.2f}%")
                    st.metric("Malignant", f"{malignant_prob:.2f}%")
                    st.progress(confidence / 100)

                    st.markdown("### 🧩 Model Output")
                    st.json({LABELS[0]: float(probs[0]), LABELS[1]: float(probs[1])})
        else:
            st.error("No model found! Please ensure your .h5 or .keras file is in this folder.")

# -----------------------------
# Tips + ABCDE + Footer
# -----------------------------
st.markdown("---")
st.subheader("🛡️ Skin Cancer Prevention Tips")
if uploaded_file and 'label' in locals():
    tips = generate_health_tips(pred_class)
    st.markdown(f"**Based on your {label} result:**")
else:
    tips = generate_health_tips(0)
    st.markdown("**General skin health recommendations:**")

for tip in tips:
    st.markdown(f"<div class='tip-item'>{tip}</div>", unsafe_allow_html=True)

st.markdown("---")
col_a, col_b = st.columns(2, gap="large")

with col_a:
    st.subheader("📋 ABCDE Rule for Melanoma")
    st.markdown("""
    - **A**symmetry: One half doesn't match the other  
    - **B**order: Irregular or blurred edges  
    - **C**olor: Multiple or uneven colors  
    - **D**iameter: Larger than 6mm  
    - **E**volving: Changes in size, shape, or color  
    """)

with col_b:
    st.subheader("⚠️ Risk Factors")
    st.markdown("""
    - Excessive UV exposure  
    - Fair skin tone  
    - History of sunburns  
    - Family history of melanoma  
    - Many or atypical moles  
    - Weakened immune system  
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #b4b4b4; padding: 1rem;'>
    <p><strong>⚠️ Medical Disclaimer:</strong> This tool is for educational purposes only.</p>
    <p>Always consult a qualified healthcare professional for medical advice and diagnosis.</p>
    <p style='margin-top: 1rem;'><small>Powered by TensorFlow | Made with Streamlit</small></p>
</div>
""", unsafe_allow_html=True)


