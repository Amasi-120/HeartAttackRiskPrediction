import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model


# ----------------------------
# Page Configuration
# ----------------------------

st.set_page_config(
    page_title="Heart Attack Risk AI",
    page_icon="❤️",
    layout="wide"
)


# ----------------------------
# Custom CSS
# ----------------------------

st.markdown("""
<style>

.main {
    background-color: #f8fbff;
}

h1 {
    color: #0b3d91;
    text-align: center;
}

.subtitle {
    text-align:center;
    color:#555;
    font-size:18px;
}

.card {
    background-color:white;
    padding:25px;
    border-radius:15px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom:20px;
}

.section-title {
    color:#0b3d91;
    font-size:22px;
    font-weight:bold;
}

.result-card {
    padding:30px;
    border-radius:20px;
    text-align:center;
    font-size:25px;
    font-weight:bold;
}

.stButton button {
    width:100%;
    height:50px;
    border-radius:12px;
    background-color:#0b3d91;
    color:white;
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)



# ----------------------------
# Load Model
# ----------------------------

@st.cache_resource
def load_resources():

    model = load_model("heart_attack_model.keras")
    scaler = joblib.load("scaler.pkl")

    return model, scaler


model, scaler = load_resources()



# ----------------------------
# Header
# ----------------------------

st.markdown(
"""
<h1>❤️ Heart Attack Risk Prediction System</h1>

<p class="subtitle">
AI-powered cardiovascular risk assessment using Deep Learning (LSTM)
</p>
""",
unsafe_allow_html=True
)



# Model Information Cards

col1, col2, col3 = st.columns(3)

with col1:
    st.info("🧠 Model\n\nLSTM Neural Network")

with col2:
    st.info("📊 Features\n\n12 Clinical Features")

with col3:
    st.info("⚡ Status\n\nOnline Prediction")



st.divider()



# ----------------------------
# Patient Information
# ----------------------------


st.markdown(
"""
<div class="section-title">
👤 Patient Information
</div>
""",
unsafe_allow_html=True
)


col1, col2 = st.columns(2)


with col1:

    age = st.number_input(
        "Age",
        1,
        120,
        40
    )

    gender = st.selectbox(
        "Gender",
        ["Female","Male"]
    )


with col2:

    weight = st.number_input(
        "Weight (kg)",
        30.0,
        200.0,
        70.0
    )

    height = st.number_input(
        "Height (m)",
        1.2,
        2.2,
        1.70
    )



bmi = weight/(height**2)


st.success(
    f"⚖️ Calculated BMI: {bmi:.2f}"
)



st.divider()



# ----------------------------
# Vital Signs
# ----------------------------


st.markdown(
"""
<div class="section-title">
❤️ Vital Signs
</div>
""",
unsafe_allow_html=True
)


col1,col2,col3,col4 = st.columns(4)


with col1:
    heart_rate = st.number_input(
        "Heart Rate",
        30.0,
        200.0,
        75.0
    )

with col2:
    resp_rate = st.number_input(
        "Respiratory Rate",
        5.0,
        40.0,
        18.0
    )

with col3:
    body_temp = st.number_input(
        "Temperature °C",
        34.0,
        42.0,
        37.0
    )

with col4:
    oxygen = st.number_input(
        "Oxygen Saturation %",
        50.0,
        100.0,
        98.0
    )



st.divider()



# ----------------------------
# Blood Pressure
# ----------------------------


st.markdown(
"""
<div class="section-title">
🩺 Blood Pressure
</div>
""",
unsafe_allow_html=True
)


col1,col2 = st.columns(2)


with col1:

    sys_bp = st.number_input(
        "Systolic Blood Pressure",
        70.0,
        250.0,
        120.0
    )


with col2:

    dia_bp = st.number_input(
        "Diastolic Blood Pressure",
        40.0,
        150.0,
        80.0
    )



# Derived Features

derived_map = (sys_bp + 2*dia_bp)/3
pulse_pressure = sys_bp-dia_bp
hrv = 60

gender_num = 1 if gender=="Male" else 0



features = np.array([[
    heart_rate,
    resp_rate,
    body_temp,
    oxygen,
    sys_bp,
    dia_bp,
    derived_map,
    hrv,
    pulse_pressure,
    bmi,
    age,
    gender_num
]])



st.divider()



# ----------------------------
# Prediction
# ----------------------------


if st.button("🚀 Predict Heart Attack Risk"):


    columns=[
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
    ]


    df = pd.DataFrame(
        features,
        columns=columns
    )


    scaled = scaler.transform(df)


    sequence=np.repeat(
        scaled,
        5,
        axis=0
    )


    sequence=sequence.reshape(
        1,
        5,
        12
    )


    prediction=model.predict(
        sequence,
        verbose=0
    )[0][0]



    st.subheader("Prediction Result")



    if prediction >=0.5:

        st.markdown(
        f"""
        <div class="result-card"
        style="background:#ffe6e6;color:#b30000">

        ⚠️ HIGH RISK

        <br>

        Confidence:
        {prediction:.2%}

        </div>
        """,
        unsafe_allow_html=True
        )


    else:

        confidence=1-prediction

        st.markdown(
        f"""
        <div class="result-card"
        style="background:#e6fff0;color:#008000">

        ✅ LOW RISK

        <br>

        Confidence:
        {confidence:.2%}

        </div>
        """,
        unsafe_allow_html=True
        )



    st.progress(
        float(prediction)
    )
