import streamlit as st
import pandas as pd
import pickle


# --------------------------------------------------
# Load model and encoders
# --------------------------------------------------

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("weather_encoder.pkl", "rb") as f:
    weather_encoder = pickle.load(f)

with open("traffic_encoder.pkl", "rb") as f:
    traffic_encoder = pickle.load(f)

with open("time_encoder.pkl", "rb") as f:
    time_encoder = pickle.load(f)

with open("vehicle_encoder.pkl", "rb") as f:
    vehicle_encoder = pickle.load(f)


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🍔",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🍔 Food Delivery Time Predictor")

st.write(
    "Enter the delivery details below to estimate the delivery time."
)


# --------------------------------------------------
# User inputs
# --------------------------------------------------

distance = st.number_input(
    "Distance (km)",
    min_value=0.1,
    max_value=100.0,
    value=5.0,
    step=0.1
)

weather = st.selectbox(
    "Weather",
    weather_encoder.classes_
)

traffic = st.selectbox(
    "Traffic Level",
    traffic_encoder.classes_
)

time_of_day = st.selectbox(
    "Time of Day",
    time_encoder.classes_
)

vehicle = st.selectbox(
    "Vehicle Type",
    vehicle_encoder.classes_
)

preparation_time = st.number_input(
    "Preparation Time (minutes)",
    min_value=1.0,
    max_value=120.0,
    value=20.0,
    step=1.0
)

courier_experience = st.number_input(
    "Courier Experience (years)",
    min_value=0.0,
    max_value=30.0,
    value=2.0,
    step=0.1
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Delivery Time"):

    # Convert categorical values to encoded values
    weather_encoded = weather_encoder.transform([weather])[0]

    traffic_encoded = traffic_encoder.transform([traffic])[0]

    time_encoded = time_encoder.transform([time_of_day])[0]

    vehicle_encoded = vehicle_encoder.transform([vehicle])[0]


    # Create input dataframe
    input_data = pd.DataFrame({
        "Distance_km": [distance],
        "Weather": [weather_encoded],
        "Traffic_Level": [traffic_encoded],
        "Time_of_Day": [time_encoded],
        "Vehicle_Type": [vehicle_encoded],
        "Preparation_Time_min": [preparation_time],
        "Courier_Experience_yrs": [courier_experience]
    })


    # Make prediction
    prediction = model.predict(input_data)[0]


    # Display result
    st.success(
        f"Estimated Delivery Time: {prediction:.0f} minutes"
    )