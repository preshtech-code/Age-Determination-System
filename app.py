import streamlit as st
import cv2
import numpy as np
from PIL import Image
import gdown
import os

# Inject custom futuristic CSS
st.markdown("""
    <style>
    body {
        background: radial-gradient(circle at top left, #0f0f0f, #1a1a2e, #0f3460);
        color: #e0e0e0;
        font-family: 'Orbitron', sans-serif;
    }
    .stApp {
        background: transparent;
    }
    h1 {
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        color: #00f5d4;
        text-shadow: 0px 0px 10px #00f5d4, 0px 0px 20px #00f5d4;
    }
    .block-container {
        padding: 2rem 3rem;
        border-radius: 20px;
        background: rgba(20, 20, 40, 0.7);
        box-shadow: 0 0 25px rgba(0, 255, 255, 0.3);
    }
    .stRadio > label {
        font-size: 1.2rem;
        color: #9ef01a;
    }
    .stButton>button {
        background: linear-gradient(135deg, #00f5d4, #00bbf9, #9b5de5);
        color: white;
        border-radius: 15px;
        padding: 0.7rem 1.5rem;
        font-size: 1rem;
        font-weight: bold;
        border: none;
        box-shadow: 0px 0px 15px #00f5d4;
        transition: 0.3s ease-in-out;
    }
    .stButton>button:hover {
        transform: scale(1.08);
        box-shadow: 0px 0px 25px #9b5de5;
    }
    .stFileUploader label {
        font-weight: bold;
        color: #f72585;
    }
    </style>
""", unsafe_allow_html=True)


# Google Drive file link
file_id = "1R8GCtxzP2KaDxGp6pa885-piHcIXRslt"
url = f"https://drive.google.com/uc?id={file_id}"
output = "age_net.caffemodel"

# Download the model if not already present
if not os.path.exists(output):
    with st.spinner("🔽 Downloading AI Age Model..."):
        gdown.download(url, output, quiet=False)

# Load models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = output  # downloaded file

# Load networks
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)

# Age categories
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


# Streamlit Futuristic UI
st.title("👾 Futuristic Age Determination AI")
st.markdown("<h4 style='text-align: center; color:#9ef01a;'>Upload an image or activate your webcam to detect age with Artificial Intelligence ⚡</h4>", unsafe_allow_html=True)

mode = st.radio("Choose mode:", ["Upload Image", "Use Webcam"])

if mode == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        frame = np.array(image)
        faces = detect_faces(frame)

        for (x1, y1, x2, y2) in faces:
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            label, confidence = predict_age(face_img)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 180), 2)
            cv2.putText(frame, f"{label}, {confidence*100:.1f}%",
                        (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX,
                        0.8, (0, 255, 180), 2, cv2.LINE_AA)

        st.image(frame, channels="RGB", caption="🧑 AI Age Detection Result")

elif mode == "Use Webcam":
    st.write("🎥 Click below to start webcam feed.")
    run = st.checkbox("🚀 Activate Futuristic Webcam")

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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 180), 2)
            cv2.putText(frame, f"{label}", (x1, y1-10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8,
                        (0, 255, 180), 2, cv2.LINE_AA)

        FRAME_WINDOW.image(frame)
    cap.release()
