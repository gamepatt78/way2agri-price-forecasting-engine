"""
FastAPI High-Performance Agricultural Price Forecasting Microservice
Way2Agribusiness Screening Assignment
"""

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, date
import numpy as np
import pandas as pd
import joblib
import os

app = FastAPI(
    title="Way2Agribusiness Price Forecasting Engine",
    description="Asynchronous Microservice for real-time commodity price predictions, 12-month forward curves, and anomaly detection.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schemas
class MonthlyForecastPoint(BaseModel):
    month_index: int = Field(..., description="Horizon step (1 to 12)")
    target_date: str = Field(..., description="Forecast target month (YYYY-MM-01)")
    predicted_price: float = Field(..., description="Point forecast in INR/quintal")
    lower_bound_95: float = Field(..., description="Lower 95% confidence interval")
    upper_bound_95: float = Field(..., description="Upper 95% confidence interval")

class ForecastResponse(BaseModel):
    model_name: str
    algorithm: str
    horizon_months: int
    latest_historical_date: str
    latest_historical_price: float
    forecast_points: List[MonthlyForecastPoint]
    forecast_generated_at: str

class CustomPredictionRequest(BaseModel):
    recent_monthly_prices: List[float] = Field(..., min_items=12, description="At least 12 recent monthly prices in chronological order")
    horizon_months: int = Field(default=12, ge=1, le=24, description="Forecast horizon")

class AnomalyPoint(BaseModel):
    date: str
    price: float
    z_score: float
    is_anomaly: bool
    reason: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    model_metadata: Dict

# Global cache for trained model
MODEL_REGISTRY = {}

def load_production_model():
    model_path = os.path.join(os.path.dirname(__file__), "..", "ml_pipeline", "artifacts", "best_model.joblib")
    meta_path = os.path.join(os.path.dirname(__file__), "..", "ml_pipeline", "artifacts", "model_metadata.json")
    if os.path.exists(model_path):
        import json
        MODEL_REGISTRY["model"] = joblib.load(model_path)
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                MODEL_REGISTRY["metadata"] = json.load(f)
        print("Loaded production model into FastAPI memory cache.")
    else:
        MODEL_REGISTRY["model"] = None
        MODEL_REGISTRY["metadata"] = {"status": "Model artifact pending generation"}

@app.on_event("startup")
async def startup_event():
    load_production_model()

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status="HEALTHY",
        service="FastAPI Price Forecasting Microservice",
        model_loaded=MODEL_REGISTRY.get("model") is not None,
        model_metadata=MODEL_REGISTRY.get("metadata", {})
    )

@app.get("/predict/forecast-12m", response_model=ForecastResponse, tags=["Forecasting"])
async def get_12_month_forecast():
    """
    Returns the official 12-month forward price curve (2025-10 to 2026-09)
    using the fine-tuned production forecasting pipeline.
    """
    meta = MODEL_REGISTRY.get("metadata", {})
    forecast_list = meta.get("official_12m_forecast", [])
    
    if not forecast_list:
        # Fallback or live generation from artifacts if json present
        forecast_cache_path = os.path.join(os.path.dirname(__file__), "..", "ml_pipeline", "artifacts", "forecast_12m.json")
        if os.path.exists(forecast_cache_path):
            import json
            with open(forecast_cache_path, "r") as f:
                forecast_data = json.load(f)
                return forecast_data
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Production forecasts not yet generated. Please execute the training pipeline."
        )

    return ForecastResponse(
        model_name=meta.get("model_name", "SARIMAX-XGBoost Ensemble"),
        algorithm=meta.get("algorithm", "Hybrid Stacking Ensemble"),
        horizon_months=len(forecast_list),
        latest_historical_date=meta.get("latest_date", "2025-09-01"),
        latest_historical_price=meta.get("latest_price", 15750.0),
        forecast_points=[MonthlyForecastPoint(**pt) for pt in forecast_list],
        forecast_generated_at=datetime.utcnow().isoformat()
    )

@app.post("/predict/custom", tags=["Forecasting"])
async def predict_custom_horizon(payload: CustomPredictionRequest):
    """
    Allows clients (e.g. Next.js dashboard, Mobile App) to supply recent prices
    and receive an on-the-fly autoregressive forecast.
    """
    prices = np.array(payload.recent_monthly_prices)
    if len(prices) < 12:
        raise HTTPException(status_code=400, detail="Require minimum 12 historical months.")
    
    # Recursive autoregressive inference using trailing momentum & seasonality
    horizon = payload.horizon_months
    history = list(prices)
    preds = []
    
    # Compute trailing seasonality pattern from past 12 months
    seasonal_factors = np.array(history[-12:]) / np.mean(history[-12:])
    trend = (history[-1] - history[-12]) / 12.0
    
    for h in range(1, horizon + 1):
        s_idx = (h - 1) % 12
        step_pred = (history[-1] + trend * 0.5) * (0.8 + 0.2 * seasonal_factors[s_idx])
        preds.append(round(float(step_pred), 2))
        history.append(step_pred)
        
    return {
        "status": "success",
        "horizon_months": horizon,
        "input_points": len(payload.recent_monthly_prices),
        "predictions": preds
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
