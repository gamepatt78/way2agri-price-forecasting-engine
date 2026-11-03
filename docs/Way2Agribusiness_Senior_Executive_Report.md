# Way2Agribusiness India Pvt Ltd — Senior Executive Technical Report
## Comprehensive Time Series Price Forecasting, Econometric Diagnostics & Production Architecture

---

### Document Metadata
- **Prepared For:** Dr. Prasanna & The Technical Evaluation Board, Way2Agribusiness India Pvt Ltd
- **Role:** AI/ML Developer (Senior Screening Evaluation)
- **Codebase Repository:** `https://github.com/gamepatt78/way2agri-price-forecasting-engine`

---

## 1. Executive Summary & Problem Formulation

### 1.1 Business Context
Way2Agribusiness India Pvt Ltd is at the forefront of modernizing Karnataka's agricultural ecosystem through direct-to-retail agribusiness services, farmer producer organization (FPO) linkages, and tech-enabled commodity trade. In agricultural supply chains, procurement cost variance represents the single largest determinant of gross processing margins. 

Mandi prices for agricultural staples fluctuate heavily under the influence of:
1. **Seasonal Harvest Influx:** Annual supply gluts during Kharif and Rabi harvesting periods depress spot prices.
2. **Pre-Monsoon & Summer Lean Seasons:** Diminished market arrivals cause price surges.
3. **Macroeconomic Shocks:** Revisions to Minimum Support Prices (MSP), fuel inflation, and trade export bans cause non-linear level shifts.

### 1.2 Mandate of the Assignment
The objective of this screening project is to forecast **average monthly prices for the next 12 operating months (October 2025 to September 2026)** using the supplied historical price records (2005–2025) and describe a possible production architecture.

Beyond statistical prediction, this assignment calls for a proposed architecture using **Django 5**, **FastAPI**, **Next.js 15**, and **React Native**, along with illustrative business actions, an effectiveness evaluation framework, and an MLOps monitoring design.

---

## 2. Forensic Data Audit & Anomaly Treatment (Question 1)

### 2.1 Dataset Profile
The dataset encompasses 249 continuous monthly price observations from **January 01, 2005 to September 01, 2025**.
- **Data Frequency:** Strictly continuous monthly observations ($\text{freq} = \text{MS}$). Zero missing dates or gaps exist in the index.
- **Summary Metrics:**
  - Sample Size: $N = 249$ months
  - Mean Price: ₹7,918.89/quintal
  - Median Price: ₹7,250.00/quintal
  - Standard Deviation: ₹2,804.99/quintal
  - Minimum Price: ₹3,500.00/quintal (April 2006)
  - Maximum Price: ₹16,163.00/quintal (December 2023)

### 2.2 Suspected 30-Month Discontinuity
An audit of month-over-month price differences ($\Delta P_t = P_t - P_{t-1}$) identified two sharp changes that require source verification:
1. In June 2021, the reported price plummeted from **₹10,863 to ₹5,070 (-53.3%)** in a single month.
2. In December 2023, the reported price surged from **₹8,400 to ₹16,163 (+92.4%)**.

The 30 prices from June 2021 through November 2023 were compared with the same-length historical window from October 2007 through March 2010. **Eleven of the 30 values differ**, and a scan of the supplied series found no exact matching 30-month window. This comparison does not establish that the observations are erroneous or explain their cause.

### 2.3 Root-Cause Analysis & Statistical Treatment
- **Cause:** Unknown from this dataset; verify against primary source records before calling the observations corrupt.
- **Treatment scenario:** For sensitivity analysis, the notebook flags the 30-month period and interpolates between May 2021 (₹10,863) and December 2023 (₹16,163), retaining the raw prices. This is a provisional modeling assumption, not a confirmed correction:
  $$\hat{P}_t = P_{t_{\text{start}}} + \frac{t - t_{\text{start}}}{t_{\text{end}} - t_{\text{start}}} \cdot \left( P_{t_{\text{end}}} - P_{t_{\text{start}}} \right)$$
  Compare forecasts trained on raw and interpolated values, and validate against the source before production use.

---

## 3. Econometric Diagnostics & Stationarity Analysis

Time series models require strict verification of stationarity. We performed both the Augmented Dickey-Fuller (ADF) test (testing for a unit root) and the KPSS test (testing for level stationarity).

### 3.1 Unit Root & Stationarity Tests

| Test | Null Hypothesis ($H_0$) | Series Evaluated | Test Statistic | p-value | Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ADF Test** | Unit Root exists (Non-stationary) | Raw Cleaned Series | $-0.0914$ | $0.9912$ | Fail to reject $H_0$ (Non-stationary) |
| **ADF Test** | Unit Root exists (Non-stationary) | 1st Differenced Series ($d=1$) | $-6.4218$ | $0.0001$ | Reject $H_0$ at 99% confidence (Stationary) |
| **KPSS Test** | Series is level stationary | Raw Cleaned Series | $1.4120$ | $0.0100$ | Reject $H_0$ (Non-stationary) |
| **KPSS Test** | Series is level stationary | 1st Differenced Series ($d=1$) | $0.1840$ | $0.1000$ | Fail to reject $H_0$ (Stationary) |

**Diagnostic Takeaway:** The commodity price series is integrated of order one, $I(1)$. Therefore, any linear statistical model must incorporate first-order integration ($d=1$) and seasonal differencing ($D=1$).

---

## 4. Model Architecture & Selection Justification (Question 2)

### 4.1 Candidate Architectures Evaluated
We formulated and evaluated models across three distinct computational paradigms:

1. **Statistical State-Space Paradigm: SARIMAX$(1,1,1) \times (1,1,1)_{12}$**
   $$\Phi_P(B^{12}) \phi_p(B) (1 - B)^d (1 - B^{12})^D y_t = \Theta_Q(B^{12}) \theta_q(B) \epsilon_t$$
   - *Strengths:* Mathematically rigorous modeling of annual crop seasonality; produces analytical Gaussian prediction intervals.
   - *Limitations:* Constrained to linear relationships; sensitive to abrupt structural level shifts.

2. **Exponential Smoothing Paradigm: Holt-Winters Triple Additive**
   - Incorporates level ($\ell_t$), linear trend ($b_t$), and additive seasonal components ($s_t$) with 12-month periodicity.
   - *Strengths:* Simple, reliable seasonal extrapolation without parameter explosion.

3. **Machine Learning Paradigm: XGBoost Regressor with Recursive Multi-Step**
   - Fits gradient boosted decision trees over 15 engineered features:
     - Lags: $y_{t-1}, y_{t-2}, y_{t-3}, y_{t-6}, y_{t-12}$
     - Rolling Statistics: $\mu_3, \mu_6, \mu_{12}, \sigma_3, \sigma_6$
     - Cyclical Calendar Encodings: $\sin\left(\frac{2\pi m}{12}\right), \cos\left(\frac{2\pi m}{12}\right)$, quarter, trend index.
   - *Strengths:* Captures complex non-linear feature interactions and momentum dynamics.
   - *Limitations:* Tree algorithms cannot extrapolate linear trends beyond maximum observed historical training values.

4. **Hybrid Forecast Candidate**
   $$\hat{y}_t^{\text{Production}} = w_1 \cdot \hat{y}_t^{\text{SARIMAX}} + w_2 \cdot \hat{y}_t^{\text{XGBoost}}, \quad w_1 = 0.50, \; w_2 = 0.50$$
   - *Rationale:* The 50/50 blend combines SARIMAX extrapolation with XGBoost lag features. The single holdout below does not establish that the blend outperforms XGBoost; additional rolling-origin validation is needed before production promotion.

---

## 5. Single Holdout Evaluation & Performance (Question 3)

### 5.1 Validation Methodology
We evaluated one **12-month out-of-sample holdout window** (not repeated rolling-origin validation):
- **Training Window:** 237 months (January 2005 to September 2024).
- **Holdout Test Window:** 12 months (October 2024 to September 2025).

Evaluation metrics:
$$\text{MAE} = \frac{1}{n}\sum_{t=1}^n |y_t - \hat{y}_t|, \quad \text{RMSE} = \sqrt{\frac{1}{n}\sum_{t=1}^n (y_t - \hat{y}_t)^2}, \quad \text{MAPE} = \frac{100\%}{n}\sum_{t=1}^n \left|\frac{y_t - \hat{y}_t}{y_t}\right|$$

### 5.2 Performance Benchmark Leaderboard

| Model Architecture | MAE (INR) | RMSE (INR) | MAPE (%) | Directional Accuracy (%) | Operational Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Seasonal Naive (Lag 12)** | ₹1,723.51 | ₹1,956.70 | 13.14% | 33.3% | Baseline |
| **Random Forest Regressor** | ₹1,502.83 | ₹1,967.29 | 12.02% | 41.7% | Tree ensemble |
| **SARIMAX$(1,1,1) \times (1,1,1)_{12}$** | ₹1,223.49 | ₹1,606.19 | 9.73% | 41.7% | Statistical candidate |
| **Holt-Winters Exponential** | ₹1,243.12 | ₹1,610.39 | 9.87% | 41.7% | Benchmark |
| **XGBoost Regressor** | **₹1,186.72** | **₹1,568.62** | **9.33%** | 33.3% | Holdout error leader |
| **50/50 Hybrid Blend** | ₹1,190.16 | ₹1,572.02 | 9.43% | 33.3% | Forecast candidate |

XGBoost had the lowest error on this holdout. The hybrid blend is used for the saved forward forecast to combine statistical extrapolation and lag-based predictions, but it was not the holdout winner. Repeated rolling-origin validation is required before claiming a production champion.

---

## 6. Official 12-Month Out-of-Sample Future Price Forecast

Trained on the complete 2005–2025 timeline, the Hybrid Ensemble projected the following forward curve (October 2025 to September 2026):

| Month Index | Target Date | Point Forecast (₹/Qtl) | Approx. 95% Prediction Interval | Market Phase & Action Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **M+1** | 2025-10-01 | **₹15,771.46** | ₹13,770.77 – ₹17,772.15 | Stable |
| **M+2** | 2025-11-01 | **₹15,786.11** | ₹13,706.94 – ₹17,865.28 | Stable |
| **M+3** | 2025-12-01 | **₹15,430.61** | ₹13,275.81 – ₹17,585.41 | Stable |
| **M+4** | 2026-01-01 | **₹15,271.50** | ₹13,043.63 – ₹17,499.37 | Stable |
| **M+5** | **2026-02-01** | **₹14,442.23** | **₹12,143.62 – ₹16,740.84** | **Forecast trough; evaluate procurement options** |
| **M+6** | 2026-03-01 | **₹15,299.42** | ₹12,932.18 – ₹17,666.66 | Stable |
| **M+7** | 2026-04-01 | **₹15,442.30** | ₹13,008.36 – ₹17,876.24 | Stable |
| **M+8** | 2026-05-01 | **₹15,885.80** | ₹13,386.95 – ₹18,384.65 | Stable |
| **M+9** | 2026-06-01 | **₹15,858.49** | ₹13,296.36 – ₹18,420.62 | Stable |
| **M+10** | **2026-07-01** | **₹16,140.03** | **₹13,516.16 – ₹18,763.90** | **Forecast high; review margin exposure** |
| **M+11** | 2026-08-01 | **₹15,357.19** | ₹12,672.99 – ₹18,041.39 | Stable |
| **M+12** | 2026-09-01 | **₹16,159.63** | ₹13,416.43 – ₹18,902.83 | Forecast high |

---

## 7. Algorithmic Corporate Strategic Actions (Question 4)

Based on the forecasted forward curve, Way2Agribusiness should execute four coordinated strategic actions:

### Action 1: Evaluate FPO Forward Contracting (Timing: January–February 2026)
- **Market Signal:** The forecast has a February price of **₹14,442.23/Qtl** and a July price of **₹16,140.03/Qtl** ($\Delta = ₹1,697.80/\text{Qtl}$).
- **Implementation:** Compare supplier quotes, delivery terms, and quality requirements for potential six-month FPO agreements.
- **Illustrative gross spread:** The forecast difference is **₹16.98 lakh per 1,000 Qtl** before contract premiums, transport, quality, financing, or storage costs. This is not a guaranteed saving.

### Action 2: Evaluate Strategic Inventory Buffering (Timing: February 2026)
- **Implementation:** Compare the forecast spread with documented warehousing, insurance, handling, spoilage, and financing costs before changing safety-stock targets.
- **Financial impact:** No net savings are claimed until costs and realized prices are measured.

### Action 3: Review B2B / B2C Pricing (Timing: May–June 2026)
- **Implementation:** Evaluate catalog price changes against realized input costs, customer demand, and contractual constraints.
- **Financial impact:** The forecast alone cannot establish margin compression or quantify profit protection.

### Action 4: Cross-Mandi Arbitrage & Geographic Routing
- **Implementation:** Establish automated daily price scrapers tracking APMC mandis in Karnataka, Andhra Pradesh, and Maharashtra. Re-route purchase orders to secondary mandis exhibiting a 2-to-3-week price transmission lag relative to Bengaluru benchmark rates.

---

## 8. Effectiveness Determination & Causal Inference (Question 5)

To evaluate whether corporate performance improvements are causally driven by the forecasting system rather than broad macroeconomic market conditions, we establish a **3-Tiered Evaluation Framework**:

### 8.1 Quantitative KPI Metrics
1. **Procurement Cost Variance (PCV):**
   $$\text{PCV} = \sum_{t=1}^T \left( P_t^{\text{spot actual}} - P_t^{\text{contracted}} \right) \cdot Q_t^{\text{procured}} - \text{Storage \& Financing Costs}$$
   - *Success Benchmark:* Set a positive net-benefit threshold with business stakeholders before a pilot; no savings target has been validated.
2. **Gross Margin Stability Index:**
   - Standard deviation of monthly gross processing margin:
     $$\sigma_{\text{margin}} = \sqrt{\frac{1}{12}\sum_{t=1}^{12} (M_t - \bar{M})^2}$$
   - *Success Benchmark:* Set a margin-stability threshold from a documented business baseline before a pilot.

### 8.2 Econometric Difference-in-Differences (DiD) Causal Estimator
To isolate causality, we compare treated commodity lines (procured using AI forward contracts) against control commodity lines (procured via traditional spot purchases):

$$\hat{\delta}_{\text{DiD}} = \left( \bar{Y}_{\text{treatment, post}} - \bar{Y}_{\text{treatment, pre}} \right) - \left( \bar{Y}_{\text{control, post}} - \bar{Y}_{\text{control, pre}} \right)$$

Where $Y$ is the effective procurement cost per quintal. A negative estimate is consistent with lower treated costs only if the design assumptions hold. Statistical significance alone does not prove causality; assess parallel trends, confounders, and uncertainty.

---

## 9. Production Architecture: Django & FastAPI (Questions 6 & 7)

```
+----------------------------------------------------------------------------------------------------+
|                                PRODUCTION RUNTIME TOPOLOGY                                         |
+----------------------------------------------------------------------------------------------------+

   [ Next.js 15 Web / SSR ]                                   [ React Native Field App ]
              |                                                           |
              +----------------------------+------------------------------+
                                           |
                                           v
                             [ NGINX API Gateway & SSL ]
                                           |
                 +-------------------------+-------------------------+
                 |                                                   |
                 v                                                   v
   +----------------------------+                      +----------------------------+
   |      Django 5 REST Core    |                      |   FastAPI Microservice     |
   |      (Port 8000)           |                      |   (Port 8001)              |
   | - Models: CommodityPrice,  |                      | - Model pre-loaded in RAM  |
   |   ModelRegistry, Forecasts |                         | - Latency to be benchmarked |
   | - Business Action Workflows|                      | - Auto Swagger /docs       |
   | - Admin Audit Dashboard    |                      | - Pydantic Request Parsing |
   +----------------------------+                      +----------------------------+
                 |                                                   |
                 +-------------------------+-------------------------+
                                           |
                                           v
                           [ Shared Redis 7.0 Memory Tier ]
                           - Response Caching (TTL = 24h)
                           - Celery Distributed Task Broker
                                           |
                                           v
                           [ Relational Time Series Store ]
                           - PostgreSQL 16 / SQLite 3
```

### 9.1 Django 5 Core Implementation (`backend/django_agri/`)
- **`CommodityPrice` Model:** Stores continuous time series data, anomaly flags (`is_anomaly`), and clean imputed series.
- **`ModelRegistry` Model:** Enterprise registry recording algorithm types, hyperparameter JSONs, validation metrics (MAE, RMSE, MAPE), and active status flags.
- **`PriceForecast` Model:** Stores 12-month forward points, confidence intervals, and realized prices for drift auditing.
- **`CorporateAction` Model:** Tracks actionable recommendations, projected vs. realized savings, and execution approval status.
- **Celery Beat Retraining:** Automated task scheduled on the 1st of every month to ingest new mandi data, check for drift, and trigger retraining.

### 9.2 FastAPI Inference Engine (`fastapi_service/main.py`)
- **Asynchronous Execution:** Built on Starlette with zero-blocking I/O.
- **In-Memory Cache:** Loads serialized `.joblib` model weights into memory at startup.
- **Performance:** Endpoint latency has not been benchmarked; measure it under representative deployment conditions before making service-level claims.
- **Interactive Documentation:** Automatic OpenAPI documentation accessible at `http://localhost:8001/docs`.

---

## 10. Frontend Client Integration: Next.js & React Native (Question 8)

### 10.1 Next.js 15 Enterprise Dashboard (`frontend_examples/NextJs_AgriDashboard.tsx`)
- **Real-Time Integration:** Fetches live 12-month forward curves from FastAPI (`/predict/forecast-12m`) and aggregated business actions from Django (`/api/dashboard/stats/`).
- **Hedging & Procurement Simulator:** Interactive volume slider (100–5,000 Quintals) that computes projected cost variance against predicted seasonal troughs.
- **Action Approval Workflow:** Allows procurement executives to approve or reject AI-recommended actions.

### 10.2 React Native Offline Field Mobile App (`frontend_examples/ReactNative_AgriApp.tsx`)
- **Field Officer Utility:** Built for procurement agents visiting rural Karnataka mandis.
- **Offline Resiliency:** Implements `AsyncStorage` caching to store the latest forward curves locally, allowing full operation in areas with intermittent connectivity.
- **Mandi Alert Badges:** Automatically displays color-coded *BUY DIP* alert badges when visiting mandis during predicted trough months.

---

## 11. Production Monitoring, Drift Handling & MLOps (Question 9)

### 11.1 Continuous Drift Metrics
1. **Population Stability Index (PSI):**
   $$\text{PSI} = \sum_{b=1}^B \left( \text{Actual}_b - \text{Expected}_b \right) \cdot \ln\left( \frac{\text{Actual}_b}{\text{Expected}_b} \right)$$
   - $\text{PSI} < 0.10$: Model stable; no intervention required.
   - $0.10 \le \text{PSI} < 0.25$: Moderate drift; warning logged to Prometheus/Grafana.
   - $\text{PSI} \ge 0.25$: Significant drift; automated retraining triggered.
2. **Kolmogorov-Smirnov (KS) Two-Sample Test:**
   - Tests whether incoming 6-month mandi price distributions diverge from baseline training data ($p < 0.05$ indicates distribution divergence).

### 11.2 Automated Retraining & Promotion Pipeline
```
New Mandi Data Ingestion 
  ---> Anomaly Validation Filter
  ---> Drift Metrics (PSI & KS Test)
  ---> Trigger Celery Retraining Task (if PSI >= 0.20 or monthly schedule)
  ---> Train Candidate Model
  ---> Walk-Forward Holdout Validation Gate (MAPE_candidate < MAPE_champion)
  ---> Canary Deployment (10% Traffic Shadow Scoring)
  ---> Automated Promotion in Django ModelRegistry
```

---

## 12. Conclusion & Verification Checklist

This submission provides an assignment-scale forecasting prototype and architectural examples for the AI/ML Developer screening task.

### Verified Deliverables:
- [x] **Primary Jupyter Notebook:** `AgriPrice_Forecasting_Way2Agri.ipynb` (Executed with all outputs and figures).
- [x] **Technical Word Report:** `Way2Agribusiness_Assignment_Report.docx` (Executive formatted document).
- [x] **Interactive HTML Export:** `AgriPrice_Forecasting_Way2Agri.html`.
- [x] **Django 5 REST Backend:** Seeded database with 249 historical prices, models, and endpoints.
- [x] **FastAPI Example:** Async inference service with Swagger UI; latency requires deployment benchmarking.
- [x] **Next.js 15 Example:** Web dashboard concept with hedging simulator and curve visualization.
- [x] **React Native Example:** Offline-first procurement app concept with alert triggers.
- [x] **GitHub Repository:** `https://github.com/gamepatt78/way2agri-price-forecasting-engine`

---

*Prepared as an assignment-scale prototype and architecture proposal.*
