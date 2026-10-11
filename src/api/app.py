import json
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from src.models.predict import load_model_bundle, run_batch_inference
from src.utils.config import CHAMPION_MODEL_PATH

MODEL_PATH = CHAMPION_MODEL_PATH


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model_bundle(MODEL_PATH) if MODEL_PATH.is_file() else None
    yield


app = FastAPI(title='Customer Churn Intelligence API', version='2.0.0', lifespan=lifespan)


class CustomerPayload(BaseModel):
    Customer_ID: str
    Recency_Days: int = Field(ge=0)
    Frequency_Orders: int = Field(ge=1)
    Monetary_Spend: float = Field(ge=0)
    Total_Units: int = Field(ge=1)
    Unique_Products: int = Field(ge=1)
    Avg_Order_Value: float = Field(ge=0)
    Avg_Units_Per_Order: float = Field(ge=0)
    Tenure_Days: int = Field(ge=0)

    @model_validator(mode='after')
    def tenure_not_shorter_than_recency(self):
        if self.Tenure_Days < self.Recency_Days:
            raise ValueError('Tenure_Days cannot be smaller than Recency_Days')
        return self


@app.get('/health')
def health_check(request: Request):
    bundle = request.app.state.model
    return {
        'status': 'online',
        'model_loaded': bundle is not None,
        'trained_at': bundle.get('trained_at') if bundle else None,
    }


@app.post('/predict')
def predict_churn(customer: CustomerPayload, request: Request):
    bundle = request.app.state.model
    if bundle is None:
        raise HTTPException(status_code=503, detail='Model is not loaded')
    scored = run_batch_inference(pd.DataFrame([customer.model_dump()]), bundle)
    prediction = json.loads(scored.to_json(orient='records'))[0]  # numpy-safe
    return {'status': 'success', 'prediction': prediction}
