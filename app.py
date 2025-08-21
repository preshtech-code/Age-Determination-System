import streamlit as st
import numpy as np
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# -------------------------------
# Fake Age Prediction Function
# (Replace with real ML model later)
# -------------------------------
def predict_age(image):
    # Dummy model: random age group
    import random
    age = random.randint(18, 60)
    confidence = round(random.uniform(70, 99), 2)
    return f"{age} years", confidence

# -------------------------------
# Face Detector
# -------------------------------
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return [(x, y, x+w, y+h) for (x, y, w, h) in faces]

# -------------------------------
# Webcam Transformer for Streamlit
# -------------------------------
class AgePredictionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detect_faces(rgb)

        for (x1, y1, x2, y2) in faces:
            face_img = rgb[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            label, conf = predict_age(face_img)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return img

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Age Determination System", layout="centered")

st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #fff;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #00f5d4;
        text-shadow: 0px 0px 20px #00f5d4;
    }
    .report-box {
        padding: 20px;
        background: rgba(0,0,0,0.4);
        border-radius: 15px;
        box-shadow: 0px 0px 20px #00f5d4;
        animation: fadeIn 1s ease-in-out;
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">🧑 Futuristic Age Determination System</div>', unsafe_allow_html=True)

# -------------------------------
# Questionnaire
# -------------------------------
with st.form("user_form"):
    name = st.text_input("Your Name")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    country = st.text_input("Country")
    hobbies = st.text_area("Hobbies / Lifestyle")
    submitted = st.form_submit_button("Save Info")

user_data = {
    "Name": name,
    "Gender": gender,
    "Country": country,
    "Hobbies": hobbies
} if submitted else {}

# -------------------------------
# Image Upload Mode
# -------------------------------
st.subheader("📸 Upload an Image")
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces = detect_faces(image)
    if faces:
        for (x1, y1, x2, y2) in faces:
            face_img = image[y1:y2, x1:x2]
            label, conf = predict_age(face_img)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} ({conf}%)", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        st.image(image, caption="Processed Image", use_column_width=True)

        # Show report
        st.markdown('<div class="report-box">', unsafe_allow_html=True)
        st.write("### 📝 Prediction Report")
        st.write(f"**Predicted Age:** {label}")
        st.write(f"**Confidence:** {conf}%")
        if user_data:
            st.write("**Your Info:**")
            for k, v in user_data.items():
                st.write(f"- {k}: {v}")
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.error("No face detected. Please upload a clearer photo.")

# -------------------------------
# Webcam Mode
# -------------------------------
st.subheader("🎥 Use Webcam")
webrtc_streamer(key="age-detection", video_transformer_factory=AgePredictionTransformer)
