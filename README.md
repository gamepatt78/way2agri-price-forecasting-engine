# Way2Agribusiness India Pvt Ltd — AI/ML Developer Screening Task
## 12-Month Commodity Price Forecasting Engine & Production Architecture

### Candidate Submission & Project Overview
This repository contains the complete end-to-end technical screening solution for the **AI/ML Developer** position at **Way2Agribusiness India Pvt Ltd** (Yeshwanthpur Office, Bengaluru, Karnataka).

The project adheres to all requirements:
1. Historical dataset ingestion (2005–2025, 249 monthly observations).
2. Exploratory data analysis, stationarity testing, and critical anomaly discovery & imputation.
3. Multi-paradigm time series forecasting (SARIMAX, Holt-Winters, Random Forest, XGBoost, Stacking Hybrid Ensemble).
4. Walk-forward 12-month holdout test validation (MAPE, MAE, RMSE).
5. 12-Month out-of-sample forward curve projection (October 2025 – September 2026) with 95% confidence intervals.
6. Corporate action formulation & effectiveness determination (Difference-in-Differences, PCV).
7. Full-stack production engineering spanning **Django 5**, **FastAPI**, **Next.js 15**, and **React Native**.
8. Enterprise MLOps production monitoring, Population Stability Index (PSI), and CI/CD automated retraining.

---

## Directory Structure
```
d:\Agri\
├── notebooks\
│   └── AgriPrice_Forecasting_Way2Agri.ipynb   <-- PRIMARY SUBMISSION FILE (Executed with all outputs & plots)
├── docs\
│   ├── Way2Agribusiness_Assignment_Report.docx  <-- Official formatted Word Document Report
│   └── AgriPrice_Forecasting_Way2Agri.html     <-- Standalone HTML export of the notebook
├── backend\
│   └── django_agri\                            <-- Production Django 5 Backend
│       ├── agri_core\                          <-- Project settings, ASGI/WSGI, root URLs
│       ├── forecasting\                        <-- Django app (Models, Views, Serializers, Admin)
│       ├── populate_db.py                      <-- Automated database seeding script
│       └── db.sqlite3                          <-- Seeded database (249 prices, models, forecasts, actions)
├── fastapi_service\
│   └── main.py                                 <-- Low-latency async FastAPI microservice (<10ms)
├── frontend_examples\
│   ├── NextJs_AgriDashboard.tsx               <-- Next.js 15 Web Dashboard with interactive Hedging Simulator
│   └── ReactNative_AgriApp.tsx                <-- React Native offline-first mobile app for field procurement
├── ml_pipeline\
│   ├── train_and_forecast.py                  <-- Automated model training & forecasting script
│   └── artifacts\                             <-- Serialized models (.joblib), metadata (.json), plots (.png)
├── data\
│   ├── price_data.csv                         <-- Historical monthly commodity prices (2005-2025)
│   └── price_data_dictionary.txt              <-- Dataset dictionary
├── SUBMISSION_GUIDE.md                         <-- Complete submission template for WhatsApp & Email
└── README.md
```

---

## Quickstart & Execution Guide

### 1. Run the Jupyter Notebook
```bash
jupyter notebook notebooks/AgriPrice_Forecasting_Way2Agri.ipynb
```
*(The notebook is pre-executed and ready to review immediately).*

### 2. Run the Django REST Backend
```bash
cd backend/django_agri
python manage.py runserver 0.0.0.0:8000
```
- API Base: `http://localhost:8000/api/`
- Forward Curve Endpoint: `http://localhost:8000/api/forecasts/forward-curve/`
- Dashboard Stats: `http://localhost:8000/api/dashboard/stats/`
- Recommended Corporate Actions: `http://localhost:8000/api/actions/`
- Admin Dashboard: `http://localhost:8000/admin/`

### 3. Run the FastAPI High-Performance Microservice
```bash
cd fastapi_service
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```
- Interactive Swagger UI: `http://localhost:8001/docs`
- 12-Month Forecast Endpoint: `http://localhost:8001/predict/forecast-12m`
- Health Check: `http://localhost:8001/health`

---

## Model Benchmark Summary (12-Month Holdout Validation)

| Model Name | Paradigm | MAE (INR) | RMSE (INR) | MAPE (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Seasonal Naive** | Baseline (Lag 12) | 1,723.51 | 1,956.70 | 13.14% |
| **Random Forest** | Tree-based ML | 1,506.10 | 1,970.89 | 12.05% |
| **SARIMAX(1,1,1)x(1,1,1)12** | Statistical Autoregressive | 1,223.50 | 1,606.19 | 9.73% |
| **Holt-Winters** | Triple Exponential Smoothing | 1,223.06 | 1,580.40 | 9.65% |
| **XGBoost Regressor** | Gradient Boosted Trees | 1,186.72 | 1,568.62 | 9.33% |
| **Hybrid Stacking Ensemble** | **Champion (SARIMAX + XGB)** | **1,190.16** | **1,572.02** | **9.43%** |

---

## Official 12-Month Future Price Forecast (Oct 2025 – Sep 2026)

| Horizon | Target Month | Predicted Price | 95% Confidence Interval | Strategic Action Signal |
| :--- | :--- | :--- | :--- | :--- |
| **M+1** | 2025-10-01 | **INR 15,771.46** | INR 13,770.77 – INR 17,772.15 | Stable |
| **M+2** | 2025-11-01 | **INR 15,786.11** | INR 13,706.94 – INR 17,865.28 | Stable |
| **M+3** | 2025-12-01 | **INR 15,430.61** | INR 13,275.81 – INR 17,585.41 | Stable |
| **M+4** | 2026-01-01 | **INR 15,271.50** | INR 13,043.63 – INR 17,499.37 | Begin Pre-Buying Negotiations |
| **M+5** | **2026-02-01** | **INR 14,442.23** | **INR 12,143.62 – INR 16,740.84** | **SEASONAL DIP: Lock FPO Forward Contracts** |
| **M+6** | 2026-03-01 | **INR 15,299.42** | INR 12,932.18 – INR 17,666.66 | Post-Harvest Recovery |
| **M+7** | 2026-04-01 | **INR 15,442.30** | INR 13,008.36 – INR 17,876.24 | Upward Inflation |
| **M+8** | 2026-05-01 | **INR 15,885.80** | INR 13,386.95 – INR 18,384.65 | Implement Dynamic Catalog Price Hikes |
| **M+9** | 2026-06-01 | **INR 15,858.49** | INR 13,296.36 – INR 18,420.62 | Pre-Monsoon Elevated Level |
| **M+10** | **2026-07-01** | **INR 16,140.03** | **INR 13,516.16 – INR 18,763.90** | **SEASONAL PEAK: Gross Margin Defense** |
| **M+11** | 2026-08-01 | **INR 15,357.19** | INR 12,672.99 – INR 18,041.39 | Interim Moderation |
| **M+12** | 2026-09-01 | **INR 16,159.63** | INR 13,416.43 – INR 18,902.83 | Late Kharif Peak |

---

## Submission Contacts
- **WhatsApp:** `+91 9449004956` (Dr. Prasanna)
- **Email:** `way2agriintelai@gmail.com`
- **Station:** Way2Agribusiness India Pvt Ltd, Yeshwanthpur Office, Bengaluru, Karnataka
