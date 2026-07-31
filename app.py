import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ----------------------------
# Load Model & Scaler
# ----------------------------
model = load_model("heart_attack_model.keras")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Heart Attack Risk Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Attack Risk Prediction")
st.write("Enter the patient's health information to predict the risk level.")

# ----------------------------
# User Inputs
# ----------------------------

heart_rate = st.number_input("Heart Rate", 30.0, 200.0, 75.0)
resp_rate = st.number_input("Respiratory Rate", 5.0, 40.0, 18.0)
body_temp = st.number_input("Body Temperature (°C)", 34.0, 42.0, 37.0)
oxygen = st.number_input("Oxygen Saturation (%)", 50.0, 100.0, 98.0)

sys_bp = st.number_input("Systolic Blood Pressure", 70.0, 250.0, 120.0)
dia_bp = st.number_input("Diastolic Blood Pressure", 40.0, 150.0, 80.0)

weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
height = st.number_input("Height (m)", 1.2, 2.2, 1.70)

age = st.number_input("Age", 1, 120, 40)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

gender_num = 1 if gender == "Male" else 0

# ----------------------------
# Derived Features
# ----------------------------

derived_bmi = weight / (height ** 2)
derived_map = (sys_bp + 2 * dia_bp) / 3
derived_pulse_pressure = sys_bp - dia_bp

# لا يوجد HRV من المستخدم لذلك نستخدم قيمة افتراضية
derived_hrv = 60

features = np.array([[
    heart_rate,
    resp_rate,
    body_temp,
    oxygen,
    sys_bp,
    dia_bp,
    derived_map,
    derived_hrv,
    derived_pulse_pressure,
    derived_bmi,
    age,
    gender_num
]])

# ----------------------------
# Prediction
# ----------------------------

if st.button("Predict"):

    import pandas as pd

    features_df = pd.DataFrame(features, columns=[
        "Heart Rate",
        "Respiratory Rate",
        "Body Temperature",
        "Oxygen Saturation",
        "Systolic Blood Pressure",
        "Diastolic Blood Pressure",
        "Derived_MAP",
        "Derived_HRV",
        "Derived_Pulse_Pressure",
        "Derived_BMI",
        "Age",
        "Gender_num"
    ])

    scaled = scaler.transform(features_df)

    sequence = np.repeat(scaled, 5, axis=0)
    sequence = sequence.reshape(1, 5, 12)

    probability = model.predict(sequence, verbose=0)[0][0]

    if probability >= 0.5:
        st.error(f"⚠️ High Risk ({probability:.2%})")
    else:
        st.success(f"✅ Low Risk ({1-probability:.2%})")

    st.progress(float(probability))