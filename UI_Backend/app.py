import streamlit as st
import requests

url = "http://127.0.0.1:8000/model"

st.set_page_config(
    page_title="Diabetes Prediction",
    layout="centered"
)

st.title("Diabetes Prediction App")
st.write("Enter patient information to predict diabetes.")

st.divider()

st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0,
        max_value=200,
        value=120
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        max_value=150,
        value=70
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=20
    )

with col2:
    insulin = st.number_input(
        "Insulin",
        min_value=0,
        max_value=300,
        value=80
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=67.1,
        value=25.0
    )

    diabetes_pedigree_function = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=2.5,
        value=0.5
    )

    age = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=30
    )

st.divider()

# Prediction Button
if st.button("Predict Diabetes", use_container_width=True):

    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree_function,
        "Age": age
    }

    try:
        response = requests.post(url, params=data)

        if response.status_code == 200:

            result = response.json()
            prediction = result["prediction"]

            st.subheader("Prediction Result")

            if prediction == 1:
                st.error("Prediction: Diabetic")
            else:
                st.success("Prediction: Non-Diabetic")

        else:
            st.error("Error in prediction.")

    except requests.exceptions.ConnectionError:
        st.error("Backend is not running. Please start FastAPI.")

st.divider()

st.caption("Powered by Logistic Regression + FastAPI + Streamlit")