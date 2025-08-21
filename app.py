import streamlit as st
import cv2
import numpy as np
from PIL import Image
import gdown
import os
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Age Detector", page_icon="🧑", layout="wide")

# --- CUSTOM CSS for FUTURISTIC STYLE ---
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #1f1c2c, #928DAB);
            color: white;
        }
        .stButton>button {
            background: linear-gradient(90deg, #00C9FF, #92FE9D);
            color: black;
            border-radius: 12px;
            border: none;
            font-weight: bold;
            box-shadow: 0px 0px 15px rgba(0,255,255,0.6);
            transition: 0.3s;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0px 0px 25px rgba(0,255,255,0.9);
        }
        .report-box {
            padding: 20px;
            border-radius: 15px;
            background: rgba(255,255,255,0.1);
            box-shadow: 0px 0px 25px rgba(0,255,255,0.3);
            margin-top: 20px;
        }
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
        .fade-in {
            animation: fadeIn 1.5s ease-in;
        }
    </style>
""", unsafe_allow_html=True)

# --- DOWNLOAD MODEL ---
file_id = "1R8GCtxzP2KaDxGp6pa885-piHcIXRslt"
url = f"https://drive.google.com/uc?id={file_id}"
output = "age_net.caffemodel"

if not os.path.exists(output):
    with st.spinner("⚡ Downloading Age Model... Please wait"):
        gdown.download(url, output, quiet=False)

# --- LOAD MODELS ---
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = output

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)

AGE_GROUPS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
              '(21-32)', '(33-43)', '(44-53)', '(60-100)']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)


def detect_faces(frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    h, w = frame.shape[:2]
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w-1, x2), min(h-1, y2)
            faces.append((x1, y1, x2, y2))
    return faces


def predict_age(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    i = agePreds[0].argmax()
    return AGE_GROUPS[i], agePreds[0][i]


# --- APP HEADER ---
st.title("✨ AI Age Determination System")
st.write("Upload an image or use your webcam. Answer a few questions, and get a **personalized report** with age prediction.")

# --- QUESTIONNAIRE ---
with st.expander("📝 Fill Quick Questionnaire (Optional)"):
    name = st.text_input("Your Name")
    gender = st.radio("Gender", ["Male", "Female", "Prefer not to say"])
    occupation = st.text_input("Occupation")
    mood = st.selectbox("How do you feel today?", ["😊 Happy", "😔 Sad", "😎 Confident", "😴 Tired", "🤔 Curious"])

mode = st.radio("Choose mode:", ["Upload Image", "Use Webcam"])

# --- IMAGE UPLOAD MODE ---
if mode == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        frame = np.array(image)

        with st.spinner("🔍 Detecting faces and predicting age..."):
            time.sleep(2)  # simulate animation delay
            faces = detect_faces(frame)

        predictions = []
        for (x1, y1, x2, y2) in faces:
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            label, confidence = predict_age(face_img)
            predictions.append((label, confidence))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2, cv2.LINE_AA)

        st.image(frame, channels="RGB")

        # --- REPORT SECTION ---
        if predictions:
            st.markdown('<div class="report-box fade-in">', unsafe_allow_html=True)
            st.subheader("📊 Personalized Report")
            if name:
                st.write(f"👤 **Name:** {name}")
            st.write(f"⚧ **Gender:** {gender}")
            if occupation:
                st.write(f"💼 **Occupation:** {occupation}")
            st.write(f"🧠 **Mood:** {mood}")
            for i, (label, conf) in enumerate(predictions):
                st.write(f"🧑 **Detected Age Range (Face {i+1}):** {label}  \n🎯 Confidence: {conf*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)

# --- WEBCAM MODE ---
elif mode == "Use Webcam":
    st.write("Click below to start webcam feed.")
    run = st.checkbox("Start Webcam")
    cap = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])

    while run:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detect_faces(frame)

        for (x1, y1, x2, y2) in faces:
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            label, confidence = predict_age(face_img)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2, cv2.LINE_AA)

        FRAME_WINDOW.image(frame)
    cap.release()
