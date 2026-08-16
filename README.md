# 🔬 Skin Cancer Detection App

A Streamlit web application that classifies skin lesion images as **Benign** or **Malignant** using a deep learning model built on EfficientNetB0. Includes a full training pipeline and an interactive demo UI for uploading and analyzing skin lesion images.

> ⚠️ **Medical Disclaimer**: This project is for **educational and research purposes only**. It is **not** a certified medical diagnostic tool and should never be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified dermatologist or healthcare provider for any concerns about skin lesions.

---

## 📁 Project Structure

```
.
├── app.py                          # Main Streamlit app (loads trained .h5 model)
├── app1.py                         # Demo/mock version of the app (no model required)
├── train.py                        # Training script (EfficientNetB0 transfer learning)
├── requirements.txt                # Python dependencies
├── melanoma_cancer_dataset/        # Training dataset (train/benign, train/malignant)
├── skin_cancer_model.keras         # Saved model (Keras format)
├── skin_cancer_model_best.h5       # Best checkpoint from training
└── skin_cancer_model_final.h5      # Final trained model
```

---

## ✨ Features

- 📤 Upload a skin lesion image (`.jpg`, `.jpeg`, `.png`)
- 🧠 Classifies the lesion as **Benign** or **Malignant**
- 📊 Displays prediction confidence and class probabilities
- 🛡️ Shows prevention tips based on the result
- 📋 Includes the **ABCDE rule** for melanoma self-checks
- ⚠️ Lists common melanoma risk factors

---

## 🧠 Model

The model is built via transfer learning on **EfficientNetB0** (pretrained on ImageNet):

- Input size: `224x224x3`
- Base: EfficientNetB0 (top layers fine-tuned, earlier layers frozen)
- Head: `GlobalAveragePooling2D → Dense(256, relu) → Dropout(0.5) → Dense(2, softmax)`
- Loss: Categorical Crossentropy
- Optimizer: Adam
- Callbacks: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`
- Class imbalance handled via `class_weight='balanced'`

Training data is expected in the following structure:

```
melanoma_cancer_dataset/
└── train/
    ├── benign/
    └── malignant/
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Train the model

If you want to retrain the model on your own dataset:

```bash
python train.py
```

This will save `skin_cancer_model_best.h5` and `skin_cancer_model_final.h5`.

### 4. Run the app

**Main app** (uses the trained model):

```bash
streamlit run app.py
```

**Demo app** :

```bash
streamlit run app1.py
```

---

## 📦 Requirements

```
streamlit
tensorflow
pillow
numpy
```

Install with:

```bash
pip install -r requirements.txt
```

## 🗺️ Roadmap Ideas

- [ ] Remove artificial confidence boosting from `app.py`
- [ ] Add model evaluation metrics (precision, recall, ROC-AUC) to the README
- [ ] Add Grad-CAM visualization for model interpretability
- [ ] Deploy as a hosted demo (Streamlit Community Cloud / Hugging Face Spaces)
- [ ] Expand dataset and add cross-validation

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open a pull request or issue.
