
from fastapi import FastAPI
import uvicorn
import joblib
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI(
    title="PCOS predictor",
    version="0.1",
)

# Defining path operation for root endpoint
@app.get('/')
def main():
	return {'message': 'This is a model for ascertaining wether someone has PCOS or not'}

class RequestBody(BaseModel):
    Weight_kg: float
    Age: str
    Hyperandrogenism: str
    Hirsutism: str
    Conception_Difficulty: str
    Insulin_Resistance: str
    Exercise_Frequency: str
    Exercise_Type: str
    Exercise_Duration: str
    Sleep_Hours: str
    Exercise_Benefit: str
    Hormonal_Imbalance: str

@app.on_event('startup')
def load_model():
    global model
    model_uri = "../mlruns/1/fa712ae5b5a94169b4251a2c77c79c92/artifacts/model"
    model = mlflow.pyfunc.load_model(model_uri)

# Defining path operation for /predict endpoint
@app.post('/predict')
def predict(data : RequestBody):
    X = pd.DataFrame([data.dict()])
    predictions = model.predict(X)
    preds = ["No PCOS", "PCOS"]
    return {'Prediction': preds[predictions.tolist()[0]]}