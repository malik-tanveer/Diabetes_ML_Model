import streamlit as st
import requests

url = "http://127.0.0.1:8000/ml_model"

st.title("Diabetes Prediction App")

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20)
glucose = st.number_input("Glucose", min_value=0, max_value=200)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100)
insulin = st.number_input("Insulin", min_value=0, max_value=300)
bmi = st.number_input("BMI", min_value=0.0, max_value=67.1)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5)
age = st.number_input("Age", min_value=0, max_value=120)

if st.button("Predict"):
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

    response = requests.post(url,  params=data)

    if response.status_code == 200:
        result = response.json()
        st.success(result["message"])
    else:
        st.error("Error in prediction. Please try again.")