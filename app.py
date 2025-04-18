import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import random

st.set_page_config(page_title="HealthScan.AI", page_icon="🩺")
st.title("🩺 HealthScan.AI – Face-Based Wellness Scanner")
st.markdown("Upload a selfie and I'll analyze your facial clues to give a wellness tip!")

uploaded_file = st.file_uploader("📷 Upload Your Selfie", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption="🖼️ Uploaded Image", use_container_width=True)


    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detection:
        results = face_detection.process(image_rgb)

        if results.detections:
            st.success("✅ Face detected!")

            # Simulated Emotion Detection
            emotion = random.choice(["Fatigued", "Happy", "Stressed", "Neutral", "Sleepy"])
            score = random.randint(60, 90)

            st.markdown(f"### 😐 Detected Emotion: **{emotion}**")
            st.markdown(f"### 📊 Wellness Score: **{score}/100**")

            # Simple local logic for tips
            if emotion == "Fatigued":
                tip = "Try taking a 15-minute power nap and drink water."
            elif emotion == "Stressed":
                tip = "Practice 5 minutes of deep breathing. You’ve got this 💪."
            elif emotion == "Sleepy":
                tip = "Your body needs rest. Aim for 7–8 hours tonight 💤."
            elif emotion == "Happy":
                tip = "You're doing great! Keep the positivity going 😊."
            else:
                tip = "Maintain your daily balance. Hydration, food, and rest count."

            st.info(f"💡 **Wellness Tip**: {tip}")
        else:
            st.error("😕 No face detected. Try again with a clearer image.")
