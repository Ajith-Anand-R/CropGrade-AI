import os
import sys
import tempfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Add root folder to path so we can import Models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Models.inference import predict_batch

# Set Streamlit Page Config
st.set_page_config(
    page_title="CropGrade AI",
    page_icon="🍅",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Premium Styling
st.markdown("""
    <style>
    /* Dark Theme Overlay */
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    
    /* Header design */
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-size: 38px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #ff4e50, #f9d423);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    .main-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        color: #9ca3af;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 24px;
    }
    
    /* Glassmorphism panels */
    .glass-card {
        background: rgba(22, 30, 49, 0.7);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Grade Badges */
    .grade-badge-a {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 6px 16px;
        border-radius: 12px;
        font-weight: 700;
        text-transform: uppercase;
        display: inline-block;
    }
    .grade-badge-b {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid #f59e0b;
        padding: 6px 16px;
        border-radius: 12px;
        font-weight: 700;
        text-transform: uppercase;
        display: inline-block;
    }
    .grade-badge-c {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 6px 16px;
        border-radius: 12px;
        font-weight: 700;
        text-transform: uppercase;
        display: inline-block;
    }
    
    /* Box recommendation */
    .rec-box {
        background-color: rgba(255, 255, 255, 0.04);
        border-left: 4px solid #ff5e62;
        border-radius: 4px 12px 12px 4px;
        padding: 16px;
        margin-top: 16px;
        font-size: 14px;
        line-height: 1.5;
    }
    </style>
""", unsafe_allow_html=True)

# Layout Title
st.markdown('<div class="main-title">CropGrade AI</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Tomato Quality Assessment & Freshness</div>', unsafe_allow_html=True)

# Camera input container
st.write("### 📸 Scan Tomato Batch")
photo = st.camera_input("Position the box containing a ₹10 coin reference in the frame:")

# Upload fallback if no camera
if not photo:
    st.write("---")
    st.write("##### Or upload a photo manually:")
    photo = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if photo is not None:
    # Save file temporarily
    suffix = ".jpg"
    if hasattr(photo, "name"):
        suffix = os.path.splitext(photo.name)[1]
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(photo.read())
        temp_path = temp_file.name

    # Display processing status
    with st.spinner("AI evaluating tomato quality and size..."):
        try:
            results = predict_batch(temp_path)
        except Exception as e:
            st.error(f"Error executing inference: {e}")
            results = None
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    if results:
        # Load raw image in OpenCV to draw boundaries
        # We read from BytesIO directly to avoid locking or file access issues
        photo.seek(0)
        file_bytes = np.asarray(bytearray(photo.read()), dtype=np.uint8)
        cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Color codes (BGR format)
        COLOR_MAP = {
            "grade_a": (0, 185, 16),      # Green
            "grade_b": (11, 158, 245),    # Yellow
            "grade_c": (68, 68, 239)      # Red
        }
        
        # Draw Tomato Bounding Boxes
        for det in results["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            grade = det["grade"]
            color = COLOR_MAP.get(grade, (255, 255, 255))
            
            # Bbox
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, 4)
            # Label background
            label_text = f"#{det['id']} Grade {grade[-1].upper()}"
            (w_txt, h_txt), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(cv_img, (x1, y1 - h_txt - 10), (x1 + w_txt + 10, y1), color, -1)
            # Label text
            cv2.putText(cv_img, label_text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
        # Draw Coin Circle (Blue)
        coin_circle = results.get("coin_circle")
        if coin_circle:
            cx, cy, cr = coin_circle
            cv2.circle(cv_img, (cx, cy), cr, (255, 80, 50), 4) # blue outline
            # Label
            cv2.putText(cv_img, "Reference Coin (27mm)", (cx - cr, cy - cr - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 50), 2, cv2.LINE_AA)
            
        # Convert BGR to RGB for Streamlit displaying
        annotated_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        # Render image
        st.write("### 🖼️ Detection Preview")
        st.image(annotated_rgb, use_container_width=True)
        
        # Render Batch Report
        st.write("### 📊 Batch Report Summary")
        summary = results["summary"]
        
        # Select badge CSS class
        badge_class = "grade-badge-b"
        if summary["batch_grade"] == "Grade A":
            badge_class = "grade-badge-a"
        elif summary["batch_grade"] == "Grade C":
            badge_class = "grade-badge-c"
            
        st.markdown(f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                    <span style="font-family: 'Outfit'; font-size: 20px; font-weight:700;">Evaluation Result</span>
                    <span class="{badge_class}">{summary['batch_grade']}</span>
                </div>
                <div style="display: flex; justify-content: space-around; text-align: center; margin-bottom: 20px;">
                    <div>
                        <div style="font-size: 24px; font-weight: 700;">{summary['total_count']}</div>
                        <div style="font-size: 11px; color:#9ca3af; text-transform: uppercase;">Total Detected</div>
                    </div>
                    <div style="border-left: 1px solid rgba(255,255,255,0.1); height: 40px;"></div>
                    <div>
                        <div style="font-size: 24px; font-weight: 700; text-transform: capitalize;">{summary['dominant_ripeness'].replace('_', ' ')}</div>
                        <div style="font-size: 11px; color:#9ca3af; text-transform: uppercase;">Dominant Ripeness</div>
                    </div>
                </div>
                <div class="rec-box">
                    <strong>💡 Recommendation:</strong><br/>
                    {summary['recommendation']}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Render detailed cards
        st.write("### 🔍 Individual Tomato Analysis")
        
        for det in results["detections"]:
            grade_char = det['grade'][-1].upper()
            ripeness_val = det['ripeness'].replace('_', ' ').capitalize()
            defect_val = det['defect'].capitalize() if det['defect'] != "none" else "None"
            
            # Badge styles
            sub_badge = "color: #10b981;" if det['grade'] == 'grade_a' else ("color: #f59e0b;" if det['grade'] == 'grade_b' else "color: #ef4444;")
            
            st.markdown(f"""
                <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 12px 18px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 600; font-size: 14px;">🍅 Tomato #{det['id']}</div>
                        <div style="font-size: 12px; color: #9ca3af; margin-top: 4px; display: flex; gap: 12px;">
                            <span>📏 {det['size_mm']} mm ({det['size_category']})</span>
                            <span>🍂 Ripeness: {ripeness_val}</span>
                            <span>⚠️ Defect: {defect_val}</span>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-weight: 700; font-size: 14px; {sub_badge}">Grade {grade_char}</div>
                        <div style="font-size: 11px; color: #9ca3af;">{det['shelf_life']} shelf life</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
else:
    # Instructions panel
    st.markdown("""
        <div class="glass-card" style="text-align: center; display: flex; flex-direction: column; align-items: center; gap: 16px;">
            <div style="font-size: 48px;">🍅</div>
            <h3>How to Grade Tomatoes</h3>
            <p style="font-size: 14px; color: #9ca3af; line-height: 1.5; max-width: 320px;">
                Position your mobile device directly above your tomato box. Ensure the ₹10 coin reference is placed on flat ground inside the frame. Tap the camera input above to snap and assess!
            </p>
        </div>
    """, unsafe_allow_html=True)
