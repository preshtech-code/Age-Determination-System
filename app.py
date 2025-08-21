import streamlit as st
import cv2
import numpy as np
from PIL import Image
import gdown
import os
import io
import datetime

# =========================================================
# Page Config
# =========================================================
st.set_page_config(
    page_title="AI Age Determination Suite",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# State-of-the-art Styling (Gradient + Glassmorphism)
# =========================================================
st.markdown("""
<style>
/* App background */
.stApp {
  background: radial-gradient(1200px 600px at 10% 10%, rgba(0, 255, 163, 0.08) 0%, rgba(0,0,0,0) 60%),
              radial-gradient(900px 500px at 90% 20%, rgba(0, 153, 255, 0.10) 0%, rgba(0,0,0,0) 60%),
              linear-gradient(135deg, #0f172a 0%, #111827 40%, #0b1020 100%);
  color: #E5E7EB;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica Neue, Arial;
}

/* Titles with gradient text */
h1, h2, h3 {
  font-weight: 800 !important;
  background: linear-gradient(90deg, #22d3ee, #a78bfa, #22c55e);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  letter-spacing: .5px;
}

/* Cards / panels */
.glass {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: 0 10px 35px rgba(0,0,0,0.35);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  border-radius: 18px;
  padding: 22px 24px;
}

/* Buttons */
.stButton > button {
  background: linear-gradient(90deg, #22d3ee, #a78bfa);
  color: white;
  border: 0;
  border-radius: 12px;
  padding: .6rem 1rem;
  font-weight: 700;
  transition: transform .08s ease-in-out, box-shadow .2s ease-in-out;
}
.stButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 24px rgba(167,139,250,0.35);
}

/* Radios / select styles */
[data-baseweb="radio"] > div, .stSelectbox, .stTextInput, .stSlider, .stFileUploader {
  background: rgba(255,255,255,0.04);
  border-radius: 14px;
  padding: 8px 12px;
  border: 1px solid rgba(255,255,255,0.08);
}

/* Labels */
.st-emotion-cache-16idsys p, label, .st-emotion-cache-115gedg {
  color: #e5e7eb !important;
}

/* Small helper text */
.helper {
  color: #9CA3AF;
  font-size: 0.9rem;
}

/* Result badge */
.badge {
  display: inline-block;
  background: linear-gradient(90deg, #22c55e, #16a34a);
  color: white;
  padding: 6px 12px;
  border-radius: 9999px;
  font-weight: 800;
  letter-spacing: .3px;
}

/* Divider */
.hr {
  width: 100%;
  height: 1px;
  background: linear-gradient(90deg, rgba(255,255,255,0), rgba(255,255,255,.15), rgba(255,255,255,0));
  margin: 16px 0 8px 0;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# Model Files & Auto-Download (age model via Google Drive)
# =========================================================
# Your provided Google Drive file (age_net.caffemodel)
AGE_DRIVE_FILE_ID = "1R8GCtxzP2KaDxGp6pa885-piHcIXRslt"
AGE_MODEL_URL = f"https://drive.google.com/uc?id={AGE_DRIVE_FILE_ID}"
AGE_PROTO = "age_deploy.prototxt"
AGE_MODEL = "age_net.caffemodel"

# Face detector (TensorFlow graph). Keep these 2 files in your repo.
FACE_PROTO = "opencv_face_detector.pbtxt"
FACE_MODEL = "opencv_face_detector_uint8.pb"

# Download age model if missing
if not os.path.exists(AGE_MODEL):
    with st.spinner("Downloading age model… (first run only)"):
        gdown.download(AGE_MODEL_URL, AGE_MODEL, quiet=False)

# Quick file existence check for face model / protos
missing = []
for f in [FACE_PROTO, FACE_MODEL, AGE_PROTO]:
    if not os.path.exists(f):
        missing.append(f)

if missing:
    st.error(
        "❌ Missing required model/config files:\n\n" +
        "\n".join([f"- `{m}`" for m in missing]) +
        "\n\nPlease add them to your repository."
    )
    st.stop()

# =========================================================
# Load Networks
# =========================================================
try:
    faceNet = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
    ageNet = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# Age buckets trained for this model
AGE_GROUPS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-32)', '(33-43)', '(44-53)', '(60-100)']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# =========================================================
# Helpers
# =========================================================
def detect_faces_bboxes(rgb_image, conf_thresh=0.7):
    """Detect faces and return bounding boxes on the given RGB image."""
    h, w = rgb_image.shape[:2]
    blob = cv2.dnn.blobFromImage(rgb_image, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf >= conf_thresh:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
            if x2 > x1 and y2 > y1:
                bboxes.append((x1, y1, x2, y2, conf))
    return bboxes

def predict_age_on_face(rgb_face):
    """Return (age_label, confidence_score) for a cropped RGB face."""
    if rgb_face.size == 0:
        return None, None
    blob = cv2.dnn.blobFromImage(rgb_face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    ageNet.setInput(blob)
    preds = ageNet.forward()
    idx = int(np.argmax(preds[0]))
    return AGE_GROUPS[idx], float(preds[0][idx])

def draw_annotations(rgb_image, results):
    """Draw bounding boxes and labels on RGB image and return it."""
    img = rgb_image.copy()
    for (x1, y1, x2, y2, conf), age_label, age_conf in results:
        cv2.rectangle(img, (x1, y1), (x2, y2), (34, 197, 94), 2)
        text = f"{age_label}  {age_conf*100:.1f}%"
        cv2.putText(img, text, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (34, 197, 94), 2, cv2.LINE_AA)
    return img

def make_report_html(name, survey, detections):
    """Create a lightweight HTML report."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    det_text = "No face detected."
    if detections:
        labels = [f"{age} ({conf*100:.1f}%)" for (_, age, conf) in detections]
        det_text = ", ".join(labels)

    rows = "".join(
        f"<tr><td style='padding:6px 10px;border-bottom:1px solid #eee;'>{k}</td>"
        f"<td style='padding:6px 10px;border-bottom:1px solid #eee;color:#0ea5e9'>{v}</td></tr>"
        for k, v in survey.items()
    )

    html = f"""
    <html>
    <head>
        <meta charset="utf-8"/>
        <title>AI Age Determination Report</title>
    </head>
    <body style="font-family:Segoe UI, Roboto, Helvetica, Arial; background:#0b1020; color:#e5e7eb; padding:24px;">
        <div style="max-width:900px;margin:0 auto;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:24px;">
            <h2 style="margin-top:0;background:linear-gradient(90deg,#22d3ee,#a78bfa,#22c55e);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">AI Age Determination Report</h2>
            <p class="helper" style="color:#9ca3af;">Generated: {ts}</p>
            <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,.2),transparent);margin:12px 0 18px;"></div>

            <h3 style="margin:10px 0;">Subject</h3>
            <div style="background:rgba(255,255,255,0.04);padding:12px 14px;border-radius:12px;border:1px solid rgba(255,255,255,0.08);">
                <strong>Name:</strong> {name if name else "—"}
            </div>

            <h3 style="margin:18px 0 8px;">Questionnaire Summary</h3>
            <table style="width:100%;border-collapse:collapse;background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.08);border-radius:12px;">
                {rows}
            </table>

            <h3 style="margin:18px 0 8px;">Age Detection</h3>
            <div style="background:rgba(34,197,94,0.12);padding:12px 14px;border-radius:12px;border:1px solid rgba(34,197,94,0.35);">
                <strong>Result(s):</strong> {det_text}
            </div>

            <p style="color:#9ca3af;margin-top:18px;">Note: This is an AI-based estimation. For health or official verification, consult a professional.</p>
        </div>
    </body>
    </html>
    """
    return html

# =========================================================
# Session State
# =========================================================
if "step" not in st.session_state:
    st.session_state.step = 1
if "survey" not in st.session_state:
    st.session_state.survey = {}
if "detections" not in st.session_state:
    st.session_state.detections = []
if "name" not in st.session_state:
    st.session_state.name = ""

# =========================================================
# Sidebar: Navigation / Progress
# =========================================================
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    st.progress(st.session_state.step / 4)
    st.markdown("---")
    st.caption("Complete the steps to generate your report.")
    if st.button("Restart flow ↺"):
        st.session_state.step = 1
        st.session_state.survey = {}
        st.session_state.detections = []
        st.session_state.name = ""
        st.experimental_rerun()

# =========================================================
# Header
# =========================================================
st.markdown("<h1>👤 AI Age Determination Suite</h1>", unsafe_allow_html=True)
st.markdown('<p class="helper">Answer a short questionnaire, then upload a photo or use your camera. We estimate age and generate a neat report.</p>', unsafe_allow_html=True)

# =========================================================
# STEP 1 — Education
# =========================================================
if st.session_state.step == 1:
    st.markdown("### 🏫 Education", unsafe_allow_html=True)
    with st.container():
        with st.form("edu_form", clear_on_submit=False):
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            name = st.text_input("Full Name (for your report)")
            p1 = st.text_input("Year graduated from Primary School")
            p2 = st.text_input("Year graduated from Secondary School")
            p3 = st.text_input("Year you wrote WAEC")
            p4 = st.text_input("Year you did JAMB")
            p5 = st.text_input("Year you finished University")
            submitted = st.form_submit_button("Next ➡️")
            st.markdown('</div>', unsafe_allow_html=True)

        if submitted:
            st.session_state.name = name.strip()
            st.session_state.survey.update({
                "Primary School Graduation": p1,
                "Secondary School Graduation": p2,
                "WAEC Year": p3,
                "JAMB Year": p4,
                "University Graduation": p5
            })
            st.session_state.step = 2
            st.experimental_rerun()

# =========================================================
# STEP 2 — Personal Life
# =========================================================
elif st.session_state.step == 2:
    st.markdown("### 🧩 Personal Life", unsafe_allow_html=True)
    with st.form("personal_form"):
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        married = st.radio("Are you married?", ["Yes", "No"], horizontal=True)
        children = st.radio("Do you have children?", ["Yes", "No"], horizontal=True)
        sleep = st.slider("Average hours of sleep per day", 3, 12, 7)
        alcohol = st.radio("Do you drink alcohol?", ["Yes", "No"], horizontal=True)
        smoking = st.radio("Do you smoke?", ["Yes", "No"], horizontal=True)

        colA, colB = st.columns(2)
        prev_btn = colA.button("⬅️ Previous")
        next_btn = colB.form_submit_button("Next ➡️")
        st.markdown('</div>', unsafe_allow_html=True)

    if prev_btn:
        st.session_state.step = 1
        st.experimental_rerun()
    if next_btn:
        st.session_state.survey.update({
            "Married": married,
            "Children": children,
            "Sleep (hrs/day)": sleep,
            "Alcohol": alcohol,
            "Smoking": smoking
        })
        st.session_state.step = 3
        st.experimental_rerun()

# =========================================================
# STEP 3 — Career & Milestones
# =========================================================
elif st.session_state.step == 3:
    st.markdown("### 💼 Career & Milestones", unsafe_allow_html=True)
    with st.form("career_form"):
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        job = st.text_input("Current job / profession")
        job_year = st.text_input("Year you started current job")
        license_ = st.radio("Do you have a driver's license?", ["Yes", "No"], horizontal=True)
        property_ = st.radio("Do you own any property?", ["Yes", "No"], horizontal=True)

        colA, colB = st.columns(2)
        prev_btn = colA.button("⬅️ Previous")
        next_btn = colB.form_submit_button("Continue to Facial Scan 🎥")
        st.markdown('</div>', unsafe_allow_html=True)

    if prev_btn:
        st.session_state.step = 2
        st.experimental_rerun()
    if next_btn:
        st.session_state.survey.update({
            "Job/Profession": job,
            "Job Start Year": job_year,
            "Driver's License": license_,
            "Owns Property": property_
        })
        st.session_state.step = 4
        st.experimental_rerun()

# =========================================================
# STEP 4 — Facial Scan (Upload or Camera) + Report
# =========================================================
elif st.session_state.step == 4:
    st.markdown("### 🎥 Facial Age Detection", unsafe_allow_html=True)
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    mode = st.radio("Choose input mode:", ["Upload Image", "Use Camera"], horizontal=True)
    detections_for_report = []

    col1, col2 = st.columns([1, 1])

    # ---------- Upload Image ----------
    if mode == "Upload Image":
        with col1:
            uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
            st.caption("Tip: Use a front-facing, well-lit photo for best results.")
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            frame_rgb = np.array(image)
            faces = detect_faces_bboxes(frame_rgb)

            results = []
            for (x1, y1, x2, y2, conf) in faces:
                face_crop = frame_rgb[y1:y2, x1:x2]
                label, score = predict_age_on_face(face_crop)
                if label is not None:
                    results.append(((x1, y1, x2, y2, conf), label, score))

            frame_annot = draw_annotations(frame_rgb, results) if results else frame_rgb

            with col2:
                st.image(frame_annot, caption="Detection Result", use_container_width=True)
                if not results:
                    st.warning("No face detected. Try a clearer photo.")
                else:
                    st.success(f"Detected {len(results)} face(s).")
                    # Save for report (no boxes needed, just labels)
                    detections_for_report = [(r[0], r[1], r[2]) for r in results]

    # ---------- Camera Snapshot ----------
    else:
        with col1:
            snap = st.camera_input("Capture a snapshot")
        if snap is not None:
            image = Image.open(snap).convert("RGB")
            frame_rgb = np.array(image)
            faces = detect_faces_bboxes(frame_rgb)

            results = []
            for (x1, y1, x2, y2, conf) in faces:
                face_crop = frame_rgb[y1:y2, x1:x2]
                label, score = predict_age_on_face(face_crop)
                if label is not None:
                    results.append(((x1, y1, x2, y2, conf), label, score))

            frame_annot = draw_annotations(frame_rgb, results) if results else frame_rgb

            with col2:
                st.image(frame_annot, caption="Detection Result", use_container_width=True)
                if not results:
                    st.warning("No face detected. Try better lighting or framing.")
                else:
                    st.success(f"Detected {len(results)} face(s).")
                    detections_for_report = [(r[0], r[1], r[2]) for r in results]

    st.markdown('</div>', unsafe_allow_html=True)

    # Store detections for report in session
    if detections_for_report:
        st.session_state.detections = detections_for_report

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # ---------- Report Section ----------
    st.markdown("### 📑 Final Report", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        colL, colR = st.columns([1, 1])

        with colL:
            st.markdown("#### Subject")
            st.write(f"**Name:** {st.session_state.name or '—'}")

            st.markdown("#### Questionnaire Summary")
            for k, v in st.session_state.survey.items():
                st.write(f"• **{k}:** {v if v else '—'}")

        with colR:
            st.markdown("#### Age Detection Result")
            if not st.session_state.detections:
                st.warning("No detection results yet. Upload/capture a face image above.")
            else:
                for _, age_label, score in st.session_state.detections:
                    st.markdown(f"<span class='badge'>Estimated Age: {age_label} &nbsp;|&nbsp; Confidence: {score*100:.1f}%</span>",
                                unsafe_allow_html=True)

        # Downloadable HTML report
        if st.session_state.detections:
            report_html = make_report_html(
                st.session_state.name,
                st.session_state.survey,
                st.session_state.detections
            )
            b = io.BytesIO(report_html.encode("utf-8"))
            st.download_button(
                label="⬇️ Download Report (HTML)",
                data=b,
                file_name="age_report.html",
                mime="text/html"
            )

        st.markdown('</div>', unsafe_allow_html=True)

    # Navigation
    nav_prev, nav_reset = st.columns([1, 1])
    if nav_prev.button("⬅️ Back to Career"):
        st.session_state.step = 3
        st.experimental_rerun()
    if nav_reset.button("Start Over ↺"):
        st.session_state.step = 1
        st.session_state.survey = {}
        st.session_state.detections = []
        st.session_state.name = ""
        st.experimental_rerun()
