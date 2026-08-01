import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model


# ==============================
# Page Configuration
# ==============================

st.set_page_config(
    page_title="Heart Attack Risk Prediction",
    page_icon="❤️",
    layout="wide"
)

st.markdown(
"""
<center>

### 🩺 AI-Based Clinical Decision Support System

This application predicts the probability of heart attack risk using an LSTM Deep Learning model trained on clinical vital signs.

</center>
""",
unsafe_allow_html=True
)


# ==============================
# Custom Style
# ==============================

st.markdown("""
<style>

body {
    background-color:#f5f9ff;
}

.main-title {
    text-align:center;
    font-size:45px;
    font-weight:800;
    color:#0B3D91;
    margin-bottom:5px;
}

.sub-title {
    text-align:center;
    color:#64748b;
    font-size:20px;
    margin-bottom:30px;
}


.card {
    background:white;
    padding:25px;
    border-radius:20px;
    box-shadow:0px 8px 25px rgba(0,0,0,0.08);
    margin-bottom:20px;
}


.card-title {
    color:#0B3D91;
    font-size:22px;
    font-weight:bold;
}


.result-high {
    background:#ffe4e6;
    padding:35px;
    border-radius:25px;
    text-align:center;
    color:#b91c1c;
    font-size:30px;
    font-weight:bold;
}


.result-low {
    background:#dcfce7;
    padding:35px;
    border-radius:25px;
    text-align:center;
    color:#15803d;
    font-size:30px;
    font-weight:bold;
}


.stButton button {

    width:100%;
    height:55px;
    border-radius:15px;
    background:#0B3D91;
    color:white;
    font-size:20px;
    font-weight:bold;

}


</style>

""", unsafe_allow_html=True)



# ==============================
# Load Resources
# ==============================

@st.cache_resource
def load_resources():

    model = load_model(
        "heart_attack_model.keras"
    )

    scaler = joblib.load(
        "scaler.pkl"
    )

    return model, scaler


model, scaler = load_resources()



# ==============================
# Sidebar
# ==============================

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/2966/2966327.png",
        width=100
    )

    st.title("❤️ Heart AI")
    st.success("🟢 Model Loaded Successfully")
    st.metric("Model Type", "LSTM")
    st.metric("Clinical Features", "12")

    st.write(
        """
        **Heart Attack Risk Prediction**

        Deep Learning healthcare system
        using LSTM Neural Network.

        ---
        
        Model:
        LSTM

        Framework:
        TensorFlow

        Features:
        12 Clinical Variables
        """
    )



# ==============================
# Header
# ==============================

st.markdown(
"""
<div class="main-title">
❤️ Heart Attack Risk Prediction
</div>

<div class="sub-title">
AI-powered cardiovascular risk assessment system using Deep Learning
</div>
""",
unsafe_allow_html=True
)



# ==============================
# Info Cards
# ==============================


c1,c2,c3 = st.columns(3)


with c1:
    st.info(
        "🧠\n\n"
        "Deep Learning\n\n"
        "LSTM Model"
    )


with c2:
    st.info(
        "📊\n\n"
        "Input Features\n\n"
        "12 Clinical Features"
    )


with c3:
    st.info(
        "⚡\n\n"
        "Prediction\n\n"
        "Real-time AI"
    )



st.divider()



# ==============================
# Patient Information
# ==============================


st.markdown(
"""
<div class="card-title">
👤 Patient Information
</div>
""",
unsafe_allow_html=True
)


col1,col2,col3,col4 = st.columns(4)


with col1:
    age = st.number_input(
        "Age",
        1,
        120,
        40
    )


with col2:
    gender = st.selectbox(
        "Gender",
        ["Female","Male"]
    )


with col3:
    weight = st.number_input(
        "Weight (kg)",
        30.0,
        200.0,
        70.0
    )


with col4:
    height = st.number_input(
        "Height (m)",
        1.2,
        2.2,
        1.70
    )


bmi = weight/(height**2)


st.success(
    f"⚖️ BMI: {bmi:.2f}"
)



st.divider()



# ==============================
# Vital Signs
# ==============================


st.markdown(
"""
<div class="card-title">
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



# ==============================
# Blood Pressure
# ==============================


st.markdown(
"""
<div class="card-title">
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



# ==============================
# Prediction
# ==============================


if st.button(
    "🚀 Analyze Heart Risk"
):


    gender_num = 1 if gender=="Male" else 0


    map_value = (
        sys_bp + 2*dia_bp
    )/3


    pulse_pressure = (
        sys_bp-dia_bp
    )


    hrv = 60



    data = np.array([[

        heart_rate,
        resp_rate,
        body_temp,
        oxygen,
        sys_bp,
        dia_bp,
        map_value,
        hrv,
        pulse_pressure,
        bmi,
        age,
        gender_num

    ]])


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
        data,
        columns=columns
    )


    scaled = scaler.transform(df)



    sequence = np.repeat(
        scaled,
        5,
        axis=0
    )


    sequence = sequence.reshape(
        1,
        5,
        12
    )


    prediction = model.predict(
        sequence,
        verbose=0
    )[0][0]



    st.divider()

    st.subheader(
        "Prediction Result"
    )


    if prediction >=0.5:

        st.markdown(
        f"""
        <div class="result-high">

        ⚠️ HIGH RISK

        <br><br>

        Confidence:
        {prediction:.2%}

        </div>
        """,
        unsafe_allow_html=True
        )


        value=prediction



    else:

        confidence=1-prediction

        st.markdown(
        f"""
        <div class="result-low">

        ✅ LOW RISK

        <br><br>

        Confidence:
        {confidence:.2%}

        </div>
        """,
        unsafe_allow_html=True
        )


        value=confidence



    st.markdown("### 📊 Risk Level")

st.progress(float(value))

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Risk Score", f"{value:.2%}")

with col2:
    st.metric("BMI", f"{bmi:.2f}")

with col3:
    st.metric("MAP", f"{map_value:.1f}")

st.markdown("---")

if prediction >= 0.5:
        st.markdown("### 📊 Risk Level")

        st.markdown("### 📊 Risk Level")

    st.progress(float(value))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Risk Score", f"{value:.2%}")

    with col2:
        st.metric("BMI", f"{bmi:.2f}")

    with col3:
        st.metric("MAP", f"{map_value:.1f}")

    st.markdown("---")

    if prediction >= 0.5:
        st.warning(
            "⚠️ The patient may be at high risk of heart attack. Immediate medical evaluation is recommended."
        )
    else:
        st.success(
            "✅ The patient's vital signs indicate a low predicted risk based on the model."
        )
