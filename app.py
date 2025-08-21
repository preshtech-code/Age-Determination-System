import streamlit as st
from deepface import DeepFace
from PIL import Image
import tempfile

# -------------------
# Streamlit Page Config
# -------------------
st.set_page_config(page_title="AI Age Estimator", page_icon="🧑‍⚕️", layout="centered")

# -------------------
# Custom CSS for Styling
# -------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #d9a7c7, #fffcdc);
        font-family: 'Segoe UI', sans-serif;
        color: #333;
    }
    .report-card {
        background-color: #ffffffdd;
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .report-title {
        font-size: 28px;
        font-weight: 700;
        color: #4a148c;
        text-align: center;
        margin-bottom: 15px;
    }
    .age-result {
        font-size: 22px;
        font-weight: bold;
        color: #1565c0;
        margin-bottom: 20px;
        text-align: center;
    }
    .question {
        font-weight: 600;
        color: #333;
    }
    .answer {
        color: #1565c0;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------
# App Title
# -------------------
st.title("🧑‍⚕️ AI Age Estimator")

st.write("This app estimates your age using AI and summarizes your lifestyle questionnaire in a neat report.")

# -------------------
# Questionnaire Section
# -------------------
st.subheader("📝 Quick Questionnaire")

q1 = st.radio("Do you smoke?", ["Yes", "No"])
q2 = st.radio("Do you exercise regularly?", ["Yes", "No"])
q3 = st.slider("How many hours of sleep do you get daily?", 3, 12, 7)
q4 = st.radio("Do you consume alcohol?", ["Yes", "No"])
q5 = st.selectbox("What is your diet type?", ["Balanced", "Vegetarian", "Vegan", "Fast Food Heavy", "Other"])

# Save responses in session state
st.session_state["answers"] = {
    "Do you smoke?": q1,
    "Exercise regularly?": q2,
    "Daily sleep hours": q3,
    "Alcohol consumption": q4,
    "Diet type": q5
}

# -------------------
# Upload Image
# -------------------
st.subheader("📤 Upload Your Image")
uploaded_file = st.file_uploader("Upload an image for age estimation", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    if st.button("🔎 Predict Age"):
        try:
            result = DeepFace.analyze(img_path=temp_path, actions=['age'], enforce_detection=False)
            age = result[0]['age']

            # -------------------
            # Report Page
            # -------------------
            st.markdown("<div class='report-card'>", unsafe_allow_html=True)

            st.markdown("<div class='report-title'>📑 Final Report</div>", unsafe_allow_html=True)

            st.markdown(f"<div class='age-result'>🧑 Detected Age: <strong>{age} years old</strong></div>", unsafe_allow_html=True)

            st.markdown("### 📝 Questionnaire Summary")
            for question, answer in st.session_state["answers"].items():
                st.markdown(f"<p class='question'>• {question}: <span class='answer'>{answer}</span></p>", unsafe_allow_html=True)

            st.markdown("<br><hr><p style='text-align:center; color:gray;'>💡 This is an AI-based estimation. For accurate health/age assessments, please consult a medical professional.</p>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")
