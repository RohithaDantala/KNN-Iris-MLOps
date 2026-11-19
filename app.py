
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
pca = joblib.load("models/pca.pkl")
encoder = joblib.load("models/encoder.pkl")

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "KNN Iris Prediction API is running"}
    
@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/model-info")
def model_info():
    return {"model": "KNN", "dataset": "Iris"}

@app.get("/metrics")
def metrics():
    return {"model": "KNN", "status": "running"}

@app.post("/predict")
def predict(data: IrisData):

    X = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
        data.petal_length * data.petal_width,
        data.sepal_length * data.sepal_width
    ]])

    X = scaler.transform(X)
    X = pca.transform(X)

    prediction = model.predict(X)

    species = encoder.inverse_transform(prediction)

    return {"prediction": species[0]}