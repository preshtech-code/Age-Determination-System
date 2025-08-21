import streamlit as st
import cv2
import numpy as np
from PIL import Image
import gdown
import os
import time

# ----------------- Custom CSS for Styling -----------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
    .report-box {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 0 15px rgba(0, 255, 200, 0.5);
        animation: fadeIn 2s ease-in-out;
    }
    .glow-text {
        font-size: 22px;
        font-weight: bold;
        color: #00ffcc;
        text-shadow: 0 0 10px #00ffcc, 0 0 20px #00ffcc, 0 0 30px #00ffcc;
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- Download Model -----------------
file_id = "1R8GCtxzP2KaDxGp6pa885-piHcIXRslt"
url = f"https://drive.google.com/uc?id={file_id}"
output = "age_net.caffemodel"

if not os.path.exists(output):
    with st.spinner("🚀 Downloading Age Model..."):
        gdown.download(url, output, quiet=False)

# ----------------- Load Models -----------------
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = output

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)

AGE_GROUPS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
              '(21-32)', '(33-43)', '(44-53)', '(60-100)']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# ----------------- Functions -----------------
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

# ----------------- Streamlit UI -----------------
st.title("🌌 Futuristic Age Determination System")
st.write("Upload an image or use your webcam to detect age. Afterwards, fill in a quick questionnaire for your **Personalized Report** 🚀.")

# Questionnaire
st.subheader("📝 Quick Questionnaire")
name = st.text_input("What’s your name?")
gender = st.radio("Gender:", ["Male", "Female", "Other"])
hobby = st.text_input("What’s your favorite hobby?")
goal = st.text_input("What’s one goal you’re working on this year?")

st.markdown("---")

mode = st.radio("Choose mode:", ["Upload Image", "Use Webcam"])

final_age_label = None
final_confidence = None

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        with st.spinner("⚡ Processing Image..."):
            image = Image.open(uploaded_file).convert("RGB")
            frame = np.array(image)
            faces = detect_faces(frame)
            for (x1, y1, x2, y2) in faces:
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue
                label, confidence = predict_age(face_img)
                final_age_label, final_confidence = label, confidence
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}, {confidence*100:.1f}%",
                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2, cv2.LINE_AA)
            time.sleep(1.5)
        st.image(frame, channels="RGB")

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
            final_age_label, final_confidence = label, confidence
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2, cv2.LINE_AA)
        FRAME_WINDOW.image(frame)
    cap.release()

# ----------------- Report Section -----------------
if final_age_label:
    st.markdown("<div class='report-box'>", unsafe_allow_html=True)
    st.markdown(f"<p class='glow-text'>🧑 Predicted Age Group: {final_age_label}</p>", unsafe_allow_html=True)
    st.markdown(f"<p>Confidence: {final_confidence*100:.1f}%</p>", unsafe_allow_html=True)
    st.markdown(f"<p><b>Name:</b> {name if name else 'N/A'}</p>", unsafe_allow_html=True)
    st.markdown(f"<p><b>Gender:</b> {gender}</p>", unsafe_allow_html=True)
    st.markdown(f"<p><b>Hobby:</b> {hobby if hobby else 'N/A'}</p>", unsafe_allow_html=True)
    st.markdown(f"<p><b>Goal for the Year:</b> {goal if goal else 'N/A'}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
