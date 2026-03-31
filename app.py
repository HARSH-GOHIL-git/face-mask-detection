import streamlit as st
import numpy as np
from PIL import Image
import pickle

# Set page configuration for a wider, cleaner layout
st.set_page_config(page_title="Face Mask Detector", page_icon="😷", layout="centered")

# Custom CSS for a beautiful, modern look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .title-text {
        color: #2c3e50;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .subtitle-text {
        text-align: center;
        color: #7f8c8d;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .result-mask {
        color: #155724;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
    }
    .result-no-mask {
        color: #721c24;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("<h1 class='title-text'>😷 Face Mask Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Upload an image to verify if the person is wearing a face mask.</p>", unsafe_allow_html=True)

# Load the trained model gracefully
@st.cache_resource
def load_model():
    try:
        with open("face_mask_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file 'face_mask_model.pkl' not found. Please ensure it is in the same directory as this script.")
        st.stop()

model = load_model()

# Image uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the uploaded image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    with st.spinner("Analyzing image..."):
        # --- Preprocessing (Matching your Colab Notebook's logic) ---
        # 1. Convert to RGB (in case of RGBA/Grayscale images)
        image = image.convert('RGB')
        
        # 2. Resize to 128x128
        image_resized = image.resize((128, 128))
        
        # 3. Convert to numpy array
        img_array = np.array(image_resized)
        
        # 4. Scale pixel values to 0-1
        img_scaled = img_array / 255.0
        
        # 5. Expand dimensions to match model input (1, 128, 128, 3)
        img_reshaped = np.expand_dims(img_scaled, axis=0)
        
        # --- Prediction ---
        prediction = model.predict(img_reshaped)
        pred_label = np.argmax(prediction) # Get the index of the highest probability
        # Get confidence score
        confidence = np.max(prediction)
        
        # --- Results ---
        # Based on your notebook: 1 = with_mask, 0 = without_mask
        if pred_label == 1:
            st.markdown("<div class='result-mask'>✅ Safe: Person is wearing a Mask</div>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>Confidence: {confidence*100:.2f}%</p>", unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown("<div class='result-no-mask'>❌ Alert: Person is NOT wearing a Mask</div>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>Confidence: {confidence*100:.2f}%</p>", unsafe_allow_html=True)