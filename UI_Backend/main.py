import pandas as pd
import pickle
from fastapi import FastAPI

app = FastAPI(
    title="Diabetes Prediction API",
    description="API for predicting diabetes using a trained ML model.",
    version="1.0.0"
)


# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def read_root():
    df = pd.read_csv("../diabetes.csv")

    data = df.head(10)

    return {
        "message": "Diabetes Prediction API is running",
        "data": data.to_dict(orient="records")
    }

@app.post("/model")
def predict_diabetes(
    Pregnancies: int,
    Glucose: int,
    BloodPressure: int,
    SkinThickness: int,
    Insulin: int,
    BMI: float,
    DiabetesPedigreeFunction: float,
    Age: int
):

    # Create input DataFrame
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

    # Make prediction
    prediction = model.predict(data)[0]

    # Print for backend testing
    print("Input Data:")
    print(data)

    print("Prediction:", prediction)

    # Return response
    return {
        "prediction": int(prediction)
    }
