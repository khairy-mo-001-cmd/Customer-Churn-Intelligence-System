# src/api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
from src.models.predict import load_model_bundle, run_batch_inference
from src.utils.config import MODELS_DIR

app = FastAPI(title="Customer Churn Intelligence API", version="1.0.0")

MODEL_PATH = MODELS_DIR / 'model_v1.pkl'
model_artifact = None

@app.on_event("startup")
def load_artifacts():
    global model_artifact
    if MODEL_PATH.is_file():
        model_artifact = load_model_bundle(MODEL_PATH)

class CustomerPayload(BaseModel):
    Customer_ID: str
    Frequency_Orders: int
    Monetary_Spend: float
    Total_Units: int
    Unique_Products: int
    Avg_Order_Value: float
    Avg_Units_Per_Order: float

@app.get("/health")
def health_check():
    return {"status": "online", "model_loaded": model_artifact is not None}

@app.post("/predict")
def predict_churn(customer: CustomerPayload):
    if model_artifact is None:
        raise HTTPException(status_code=500, detail="Model artifact not found in server memory.")
    
    df_input = pd.DataFrame([customer.dict()])
    scored_df = run_batch_inference(df_input, model_artifact)
    return {"status": "success", "prediction": scored_df.iloc[0].to_dict()}