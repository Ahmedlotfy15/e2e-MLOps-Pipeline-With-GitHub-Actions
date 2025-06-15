from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from infer import predict_survival

app = FastAPI(title="Survival Prediction API")

class PredictionInput(BaseModel):
    features: List[float]

@app.post("/predict")
async def predict(input_data: PredictionInput):
    
    prediction, probability = predict_survival(input_data.features)

    return {
        "prediction": prediction,
        "probability": float(probability)
    }

    
