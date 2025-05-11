import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile

# Load the trained model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()


# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f9f9f9;
        }
        .title {
            font-size: 2.5em;
            font-weight: bold;
            color: #d62828;
        }
        .subtitle {
            font-size: 1.2em;
            color: #6c757d;
        }
        .footer {
            margin-top: 50px;
            font-size: 0.9em;
            color: #999;
        }
        .uploader {
            margin: 20px 0;
        }
        .result-box {
            border-radius: 10px;
            padding: 20px;
            background-color: white;
            border: 1px solid #e0e0e0; /* Thin light gray border */
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05); /* Softer shadow */
            margin: 20px 0;
}

    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="title">🍜 Soup Can Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect Campbell soup cans using a YOLOv8x model trained on synthetic data.</div>', unsafe_allow_html=True)

# File uploader
st.markdown('<div class="uploader">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📷 Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display original and result side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)  # Updated parameter

    # Save to temp file for YOLOv8
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name)
        pred = model(tmp_file.name)[0]

    # Display result
    result_image = Image.fromarray(pred.plot()[:, :, ::-1])  # Convert BGR to RGB
    with col2:
        st.image(result_image, caption="Detection Result", use_container_width=True)  # Updated parameter

    # Prediction details box
    st.markdown('<div class="result-box" style="padding: 1px;">', unsafe_allow_html=True)
    st.subheader("📋 Prediction Details")
    
    if len(pred.boxes) == 0:
        st.write("No soup cans detected")
    else:
        for i, box in enumerate(pred.boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xywh = [round(x, 2) for x in box.xywh[0].tolist()]

            st.markdown(f"""
                 **Detection {i+1}**
                - **Class**: `{cls}`
                - **Confidence**: `{conf:.2f}`
                - **Box coordinates (x,y,width,height)**: `{xywh}`
                <div style="background-color: #f0f0f0; border-radius: 5px; height: 4px; width: 100%; margin-top: 5px; position: relative;">
                    <div style="width: {conf * 100}%; background-color: #d62828; height: 100%; border-radius: 5px;"></div>
                    <span style="position: absolute; top: -25px; right: 0; font-size: 0.9em; color: #0;">{conf * 100:.1f}%</span>
                </div>
                <div style="margin-bottom: 10px;"></div>
                """, unsafe_allow_html=True)

    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Made with ❤️ for the Synthetic 2 Real Challenge</div>', unsafe_allow_html=True)