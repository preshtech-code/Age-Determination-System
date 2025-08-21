import streamlit as st
import cv2
import numpy as np
from PIL import Image
import gdown

# ========================
# Download age model if not found
# ========================
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

file_id = "1R8GCtxzP2KaDxGp6pa885-piHcIXRslt"  # Google Drive file ID
url = f"https://drive.google.com/uc?id={file_id}"
output = "age_net.caffemodel"

import os
if not os.path.exists(ageModel):
    gdown.download(url, output, quiet=False)

# Face model
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

# Load networks
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)

# Age groups
AGE_GROUPS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
              '(21-32)', '(33-43)', '(44-53)', '(60-100)']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)


# ========================
# Helper functions
# ========================
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
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
            faces.append((x1, y1, x2, y2))
    return faces


def predict_age(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                 MODEL_MEAN_VALUES, swapRB=False)
    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    i = agePreds[0].argmax()
    return AGE_GROUPS[i], agePreds[0][i]


# ========================
# Multi-Step Form
# ========================
if "step" not in st.session_state:
    st.session_state.step = 1

st.markdown("<h2 style='text-align: center;'>📝 Smart Age Prediction Survey</h2>", unsafe_allow_html=True)
st.progress(st.session_state.step / 4)

# Step 1: Education History
if st.session_state.step == 1:
    st.header("Education History")
    st.text_input("What year did you graduate from primary school?", key="primary_school")
    st.text_input("What year did you graduate from secondary school?", key="secondary_school")
    st.text_input("What year did you write WAEC?", key="waec")
    st.text_input("What year did you do your JAMB?", key="jamb")
    st.text_input("What year did you finish university?", key="university")

    if st.button("Next ➡️"):
        st.session_state.step = 2
        st.rerun()

# Step 2: Personal Life
elif st.session_state.step == 2:
    st.header("Personal Life")
    st.radio("Are you married?", ["Yes", "No"], key="married")
    st.radio("Do you have children?", ["Yes", "No"], key="children")

    col1, col2 = st.columns(2)
    if col1.button("⬅️ Previous"):
        st.session_state.step = 1
        st.rerun()
    if col2.button("Next ➡️"):
        st.session_state.step = 3
        st.rerun()

# Step 3: Career
elif st.session_state.step == 3:
    st.header("Career & Milestones")
    st.text_input("What is your current job/profession?", key="job")
    st.text_input("What year did you start your current job?", key="job_year")
    st.radio("Do you have a driver's license?", ["Yes", "No"], key="license")
    st.radio("Do you own any property?", ["Yes", "No"], key="property")

    col1, col2 = st.columns(2)
    if col1.button("⬅️ Previous"):
        st.session_state.step = 2
        st.rerun()
    if col2.button("Continue to Facial Scan 🎥"):
        st.session_state.step = 4
        st.rerun()

# Step 4: Facial Age Scan
elif st.session_state.step == 4:
    st.header("🎥 Facial Age Detection")

    mode = st.radio("Choose mode:", ["Upload Image", "Use Webcam"])

    if mode == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            frame = np.array(image)
            faces = detect_faces(frame)

            for (x1, y1, x2, y2) in faces:
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue
                label, confidence = predict_age(face_img)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}, {confidence*100:.1f}%",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2, cv2.LINE_AA)

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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2, cv2.LINE_AA)

            FRAME_WINDOW.image(frame)
        cap.release()
