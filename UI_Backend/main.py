import pandas as pd
import numpy as np
import pickle
from fastapi import FastAPI

app = FastAPI()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/model")
def Model(Pregnancies: int,
    Glucose: int, BloodPressure: int, SkinThickness: int, Insulin: int, BMI: float, DiabetesPedigreeFunction: float, Age: int):

    data = pd.DataFrame({
        "Pregnancies": [Pregnancies],
        "Glucose": [Glucose],
        "BloodPressure": [BloodPressure],
        "SkinThickness": [SkinThickness],
        "Insulin": [Insulin],
        "BMI": [BMI],
        "DiabetesPedigreeFunction": [DiabetesPedigreeFunction],
        "Age": [Age]
    })

    print(data)

    prediction = model.predict(data)

    print("Prediction:", prediction)

    return {
        "prediction": prediction.tolist()
    }
