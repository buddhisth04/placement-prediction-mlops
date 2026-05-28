from fastapi import FastAPI
import joblib
import numpy as np

# Create FastAPI app
app = FastAPI()

# Load trained model
model = joblib.load("model/model.pkl")

# Home route
@app.get("/")
def home():
    return {"message": "Placement Prediction API Running"}

# Prediction route
@app.post("/predict")
def predict(
    ssc_p: float,
    hsc_p: float,
    degree_p: float,
    etest_p: float,
    mba_p: float
):

    data = np.array([[ssc_p, hsc_p, degree_p, etest_p, mba_p]])

    prediction = model.predict(data)

    result = "Placed" if prediction[0] == 1 else "Not Placed"

    return {"prediction": result}