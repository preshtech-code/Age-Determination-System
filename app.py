import streamlit as st
import cv2
import numpy as np
from PIL import Image
import gdown
import os

# -------------------------------
# Download Model
# -------------------------------
file_id = "1R8GCtxzP2KaDxGp6pa885-piHcIXRslt"
url = f"https://drive.google.com/uc?id={file_id}"
output = "age_net.caffemodel"

if not os.path.exists(output):
    with st.spinner("⏳ Downloading Age Model... Please wait"):
        gdown.download(url, output, quiet=False)

# -------------------------------
# Load Models
# -------------------------------
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = output

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)

AGE_GROUPS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
              '(21-32)', '(33-43)', '(44-53)', '(60-100)']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)


# -------------------------------
# Helper Functions
# -------------------------------
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


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Age Determination", page_icon="🧑", layout="centered")

st.title("🧑 Age Determination System")
st.markdown(
    """
    This app uses **Deep Learning** (OpenCV DNN) to detect faces 
    and estimate the **age group** of a person.  
    Upload an image or use your webcam 👇
    """
)

mode = st.radio("📷 Choose mode:", ["Upload Image", "Use Webcam"])


# -------------------------------
# Upload Mode
# -------------------------------
if mode == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        frame = np.array(image)
        faces = detect_faces(frame)

        if not faces:
            st.warning("⚠️ No face detected in the image. Try another one.")
        else:
            for (x1, y1, x2, y2) in faces:
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue
                label, confidence = predict_age(face_img)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}, {confidence*100:.1f}%",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2, cv2.LINE_AA)

            st.image(frame, channels="RGB", caption="🖼️ Processed Image")


# -------------------------------
# Webcam Mode
# -------------------------------
elif mode == "Use Webcam":
    st.info("🎥 Turn on your webcam to start detection")
    run = st.checkbox("✅ Start Webcam")

    cap = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Could not access webcam.")
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
