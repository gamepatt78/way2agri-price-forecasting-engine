"""
Populate Django database with historical prices, trained models,
12-month future forecasts, and AI-recommended corporate actions.
"""
import os
import sys
import django
import json
import pandas as pd
from datetime import datetime

# Setup Django environment
sys.path.append(os.path.dirname(__file__))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'agri_core.settings')
django.setup()

from forecasting.models import CommodityPrice, ModelRegistry, PriceForecast, CorporateAction

def populate():
    print("Clearing old records...")
    CorporateAction.objects.all().delete()
    PriceForecast.objects.all().delete()
    ModelRegistry.objects.all().delete()
    CommodityPrice.objects.all().delete()
    
    # 1. Ingest historical commodity prices
    csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "price_data.csv")
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Flag the candidate discontinuity for review; the supplied data does not prove its cause.
    df['is_anomaly'] = (df['date'] >= '2021-06-01') & (df['date'] <= '2023-11-01')
    df['imputed_price'] = df['avg_monthly_price'].astype(float)
    df.loc[df['is_anomaly'], 'imputed_price'] = None
    df = df.set_index('date')
    df['imputed_price'] = df['imputed_price'].interpolate(method='time')
    df = df.reset_index()
    
    records = []
    for _, row in df.iterrows():
        records.append(CommodityPrice(
            date=row['date'].date(),
            avg_monthly_price=row['avg_monthly_price'],
            is_anomaly=row['is_anomaly'],
            imputed_price=row['imputed_price'] if row['is_anomaly'] else None
        ))
    CommodityPrice.objects.bulk_create(records)
    print(f"Ingested {len(records)} commodity price records into Django DB.")
    
    # 2. Register trained models
    meta_path = os.path.join(os.path.dirname(__file__), "..", "..", "ml_pipeline", "artifacts", "model_metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    val_metrics = meta['validation_metrics']
    
    reg_models = {}
    for name, m_info in val_metrics.items():
        is_best = (name == "Hybrid Ensemble")
        reg = ModelRegistry.objects.create(
            model_name=name,
            version="1.0.0",
            algorithm_type=name,
            mae=m_info['mae'],
            rmse=m_info['rmse'],
            mape=m_info['mape'],
            directional_accuracy=m_info.get('directional_accuracy', 0.0),
            parameters={"horizon": 12, "feature_lags": [1, 2, 3, 6, 12]},
            artifact_path="ml_pipeline/artifacts/best_model.joblib",
            is_active=is_best
        )
        reg_models[name] = reg
    print(f"Registered {len(reg_models)} models in ModelRegistry.")
    
    # 3. Create Price Forecasts for Active Model
    active_reg = reg_models["Hybrid Ensemble"]
    forecast_list = meta['official_12m_forecast']
    
    forecast_objs = []
    for pt in forecast_list:
        f_obj = PriceForecast.objects.create(
            model=active_reg,
            forecast_date=datetime.strptime(meta['latest_date'], '%Y-%m-%d').date(),
            target_date=datetime.strptime(pt['target_date'], '%Y-%m-%d').date(),
            predicted_price=pt['predicted_price'],
            lower_bound=pt['lower_bound_95'],
            upper_bound=pt['upper_bound_95'],
            horizon_months=pt['month_index']
        )
        forecast_objs.append(f_obj)
    print(f"Created {len(forecast_objs)} future forecasts in PriceForecast.")
    
    # 4. Generate Strategic Corporate Actions (Answering Questions 4 & 5)
    # Action 1: Forward contracting during February dip
    feb_forecast = forecast_objs[4] # Month 5: 2026-02-01
    july_forecast = forecast_objs[9] # Month 10: 2026-07-01
    forecast_spread = max(0.0, july_forecast.predicted_price - feb_forecast.predicted_price)
    CorporateAction.objects.create(
        forecast=feb_forecast,
        action_type='FORWARD_CONTRACT',
        status='RECOMMENDED',
        trigger_condition=(
            f'Illustrative forecast spread from February to July 2026: '
            f'INR {forecast_spread:,.2f}/quintal before costs'
        ),
        action_details='Evaluate supplier quotes and terms for a February-to-July purchase plan. The forecast spread is not a savings estimate and excludes contract, storage, financing, transport, and quality costs.',
        expected_savings_inr=0.0,
    )
    
    # Action 2: Warehouse strategic buffer accumulation
    CorporateAction.objects.create(
        forecast=feb_forecast,
        action_type='INVENTORY_BUFFER',
        status='UNDER_REVIEW',
        trigger_condition='Pre-monsoon / pre-summer buffer building opportunity',
        action_details='Evaluate temporary inventory buffering during the forecast low-price period, subject to validated demand and carrying-cost analysis.',
        expected_savings_inr=0.0
    )
    
    # Action 3: Dynamic end-product pricing for July 2026 peak
    CorporateAction.objects.create(
        forecast=july_forecast,
        action_type='DYNAMIC_PRICING',
        status='RECOMMENDED',
        trigger_condition='Predicted commodity price peak exceeding INR 16,000 in July-September 2026',
        action_details='Pilot price adjustments using realized input costs and observed customer demand; the forecast alone does not establish a suitable adjustment percentage or savings amount.',
        expected_savings_inr=0.0
    )
    print("Created strategic CorporateAction recommendations.")
    print("Database population completed successfully!")

if __name__ == '__main__':
    populate()
