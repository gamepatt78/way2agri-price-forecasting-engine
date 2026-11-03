"""
Generate senior-level executive Word document (.docx) report for Way2Agribusiness screening submission.
"""
import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

doc = docx.Document()

# Page Margins
for s in doc.sections:
    s.top_margin = Inches(0.9)
    s.bottom_margin = Inches(0.9)
    s.left_margin = Inches(0.9)
    s.right_margin = Inches(0.9)

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def add_callout(text, title="KEY ARCHITECTURAL INSIGHT"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    c = tbl.rows[0].cells[0]
    c.width = Inches(6.7)
    set_cell_background(c, "F0FDF4") # soft emerald
    p = c.paragraphs[0]
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    r_t = p.add_run(f"[{title}]\n")
    r_t.bold = True
    r_t.font.color.rgb = RGBColor(5, 150, 105)
    r_b = p.add_run(text)
    r_b.font.size = Pt(10)
    r_b.font.color.rgb = RGBColor(15, 23, 42)
    doc.add_paragraph()

# Title Header
p_title = doc.add_paragraph()
r_sub = p_title.add_run("WAY2AGRIBUSINESS INDIA PVT LTD — TECHNICAL SCREENING\n")
r_sub.font.size = Pt(11)
r_sub.font.bold = True
r_sub.font.color.rgb = RGBColor(5, 150, 105)

r_main = p_title.add_run("Commodity Price Forecasting Assignment Prototype & Architecture Examples\n")
r_main.font.size = Pt(22)
r_main.font.bold = True
r_main.font.color.rgb = RGBColor(15, 23, 42)

r_tag = p_title.add_run("Technical Report: Price Forecasting, Django, FastAPI, Next.js & React Native")
r_tag.font.size = Pt(11)
r_tag.font.italic = True
r_tag.font.color.rgb = RGBColor(100, 116, 139)

doc.add_paragraph()

# Executive Metadata Table
meta_table = doc.add_table(rows=4, cols=2)
meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
meta_data = [
    ("Assessment", "AI/ML Developer screening assignment"),
    ("Target Organization", "Way2Agribusiness India Pvt Ltd"),
    ("Tech Stack Assessed", "AI/ML (Time Series), Django 5, FastAPI, Next.js 15, React Native"),
    ("Project Type", "Assignment-scale prototype and architecture examples")
]

for i, (k, v) in enumerate(meta_data):
    row = meta_table.rows[i]
    c0 = row.cells[0]
    c1 = row.cells[1]
    c0.width = Inches(2.2)
    c1.width = Inches(4.5)
    c0.paragraphs[0].add_run(k).bold = True
    c1.paragraphs[0].add_run(v)
    set_cell_background(c0, "F1F5F9")
    set_cell_background(c1, "FFFFFF")

doc.add_paragraph()

# Section 1: Executive Summary
doc.add_heading("1. Executive Summary & Problem Formulation", level=1)
doc.add_paragraph(
    "In agricultural supply chains across South India—particularly Karnataka's APMC mandi ecosystem—procurement cost "
    "volatility is the primary determinant of operating margins. Traditional agribusiness procurement operates reactively "
    "on the spot market, exposing processing facilities to severe supply-side shocks.\n\n"
    "This report presents an assignment-scale forecasting prototype and microservice architecture example developed "
    "to forecast monthly commodity prices for the next 12 operating months (October 2025 to September 2026). "
    "The solution combines econometric diagnostics, a 50/50 SARIMAX-XGBoost forecast blend, a Django 5 "
    "backend example, a FastAPI inference microservice example, a Next.js 15 hedging dashboard, "
    "and an offline-first React Native mobile client for mandi field officers."
)

add_callout(
    "On one 12-month holdout, XGBoost achieved 9.33% MAPE and the 50/50 hybrid blend achieved 9.43%; the blend is used for the saved "
    "forward forecast but did not win the holdout. The forecast's February 2026 point is INR 14,442.23/Qtl and July 2026 is "
    "INR 16,140.03/Qtl. Financial benefits from procurement actions are illustrative and depend on realized prices and costs.",
    title="EXECUTIVE BOTTOM LINE"
)

# Section 2: Forensic Data Audit
doc.add_heading("2. Question 1: Forensic Data Audit & Anomaly Treatment", level=1)
doc.add_paragraph(
    "The historical dataset spans 249 continuous monthly price observations from January 01, 2005 to September 01, 2025. "
    "Temporal analysis verified zero missing dates in sequence.\n\n"
    "However, diagnostic profiling of month-over-month differences revealed two severe disruptions:\n"
    "• In June 2021, reported prices plummeted from INR 10,863 to INR 5,070 (-53.3%) in 30 days.\n"
    "• In December 2023, reported prices leaped from INR 8,400 to INR 16,163 (+92.4%).\n\n"
    "Comparison with the same-length historical window (October 2007 through March 2010) found that 11 of the 30 values differ; "
    "a scan found no exact matching 30-month window in the supplied series. The cause is unknown and should be checked against primary "
    "source records. For sensitivity analysis, the notebook flags the period and interpolates between May 2021 (INR 10,863) and "
    "December 2023 (INR 16,163), retaining the raw values. This is a provisional modeling scenario, not a confirmed correction."
)

# Section 3: Econometric Diagnostics
doc.add_heading("3. Econometric Diagnostics & Stationarity Verification", level=1)
doc.add_paragraph(
    "Rigorous stationarity testing confirmed that the raw commodity series contains a unit root (ADF p = 0.9912, KPSS p = 0.0100), "
    "confirming non-stationarity in levels. Applying first-order differencing (d=1) resulted in strict stationarity (ADF p = 0.0001, "
    "KPSS p = 0.1000). Consequently, all autoregressive linear specifications incorporate first differencing (d=1) and annual seasonal differencing (D=1)."
)

# Section 4: Model Benchmarking
doc.add_heading("4. Questions 2 & 3: Model Architecture & Single-Holdout Benchmark", level=1)
doc.add_paragraph(
    "To capture both 12-month annual crop seasonality and non-linear macroeconomic regime shifts, we engineered 15 features "
    "(lags t-1, t-2, t-3, t-6, t-12; rolling statistics 3, 6, 12; and cyclical sine/cosine calendar encodings). Models were "
    "evaluated on one 12-month holdout test window (October 2024 to September 2025), not repeated rolling-origin validation:"
)

res_table = doc.add_table(rows=7, cols=5)
res_table.alignment = WD_TABLE_ALIGNMENT.CENTER
headers = ["Model Paradigm", "Model Architecture", "MAE (INR)", "RMSE (INR)", "MAPE (%)"]
for j, h in enumerate(headers):
    c = res_table.rows[0].cells[j]
    c.paragraphs[0].add_run(h).bold = True
    set_cell_background(c, "E2E8F0")

results_data = [
    ("Baseline", "Seasonal Naive (Lag 12)", "1,723.51", "1,956.70", "13.14%"),
    ("Tree Ensemble", "Random Forest Regressor", "1,502.83", "1,967.29", "12.02%"),
    ("State-Space", "SARIMAX(1,1,1)x(1,1,1)[12]", "1,223.49", "1,606.19", "9.73%"),
    ("Exponential Smoothing", "Holt-Winters (Additive)", "1,243.12", "1,610.39", "9.87%"),
    ("Gradient Boosting", "XGBoost Regressor", "1,186.72", "1,568.62", "9.33%"),
    ("Forecast Candidate", "50/50 Blend (SARIMAX + XGB)", "1,190.16", "1,572.02", "9.43%")
]

for i, row in enumerate(results_data):
    for j, val in enumerate(row):
        c = res_table.rows[i+1].cells[j]
        r = c.paragraphs[0].add_run(val)
        if "Ensemble" in row[0]:
            r.bold = True
            set_cell_background(c, "DCFCE7")
        elif i % 2 == 1:
            set_cell_background(c, "F8FAFC")

doc.add_paragraph()
doc.add_paragraph(
    "XGBoost had the lowest error on this holdout. The hybrid blend is used for the saved forecast but did not outperform XGBoost; "
    "additional rolling-origin validation is needed before production model promotion."
)

# Section 5: Official 12-Month Out-of-Sample Forward Curve
doc.add_heading("5. Official 12-Month Out-of-Sample Price Forecast (Oct 2025 – Sep 2026)", level=1)
doc.add_paragraph(
    "Using full historical retraining, the 50/50 SARIMAX-XGBoost forecast blend projected the following curve with approximate 95% prediction intervals:"
)

fc_table = doc.add_table(rows=13, cols=6)
fc_table.alignment = WD_TABLE_ALIGNMENT.CENTER
fc_headers = ["Horizon", "Target Month", "Predicted Price", "95% Lower Bound", "95% Upper Bound", "Strategic Action Signal"]
for j, h in enumerate(fc_headers):
    c = fc_table.rows[0].cells[j]
    c.paragraphs[0].add_run(h).bold = True
    set_cell_background(c, "E2E8F0")

forecast_data = [
    ("M+1", "2025-10-01", "INR 15,771.46", "INR 13,770.77", "INR 17,772.15", "Stable Plateau"),
    ("M+2", "2025-11-01", "INR 15,786.11", "INR 13,706.94", "INR 17,865.28", "Stable Plateau"),
    ("M+3", "2025-12-01", "INR 15,430.61", "INR 13,275.81", "INR 17,585.41", "Harvest Softening"),
    ("M+4", "2026-01-01", "INR 15,271.50", "INR 13,043.63", "INR 17,499.37", "Pre-Contracting Opens"),
    ("M+5", "2026-02-01", "INR 14,442.23", "INR 12,143.62", "INR 16,740.84", "SEASONAL DIP (BUY SIGNAL)"),
    ("M+6", "2026-03-01", "INR 15,299.42", "INR 12,932.18", "INR 17,666.66", "Post-Harvest Recovery"),
    ("M+7", "2026-04-01", "INR 15,442.30", "INR 13,008.36", "INR 17,876.24", "Rising Trend"),
    ("M+8", "2026-05-01", "INR 15,885.80", "INR 13,386.95", "INR 18,384.65", "Catalog Price Hike Window"),
    ("M+9", "2026-06-01", "INR 15,858.49", "INR 13,296.36", "INR 18,420.62", "Pre-Monsoon Consolidation"),
    ("M+10", "2026-07-01", "INR 16,140.03", "INR 13,516.16", "INR 18,763.90", "SEASONAL PEAK (HEDGE)"),
    ("M+11", "2026-08-01", "INR 15,357.19", "INR 12,672.99", "INR 18,041.39", "Interim Moderation"),
    ("M+12", "2026-09-01", "INR 16,159.63", "INR 13,416.43", "INR 18,902.83", "Late Kharif Peak")
]

for i, row in enumerate(forecast_data):
    for j, val in enumerate(row):
        c = fc_table.rows[i+1].cells[j]
        r = c.paragraphs[0].add_run(val)
        if "BUY SIGNAL" in row[5]:
            r.bold = True
            set_cell_background(c, "FEF08A")
        elif "PEAK" in row[5]:
            r.bold = True
            set_cell_background(c, "FEE2E2")
        elif i % 2 == 1:
            set_cell_background(c, "F8FAFC")

doc.add_paragraph()

# Section 6: Corporate Strategy & Effectiveness
doc.add_heading("6. Questions 4 & 5: Corporate Strategic Responses & Effectiveness Measurement", level=1)
doc.add_paragraph(
    "Algorithmic Corporate Interventions:\n"
    "1. Evaluate FPO forward contracts against supplier quotes, delivery terms, and quality requirements. The February-to-July forecast spread is INR 1,697.80/Qtl (INR 16.98 lakh per 1,000 Qtl) before contract and carrying costs; this is not guaranteed savings.\n"
    "2. Evaluate warehouse buffering against documented insurance, handling, spoilage, and financing costs before changing stock targets.\n"
    "3. Review catalog pricing against realized input costs, customer demand, and contractual constraints; the forecast alone does not quantify margin protection.\n\n"
    "Causal Effectiveness Determination:\n"
    "• Procurement Cost Variance (PCV): Net monetary difference between actual spot outlay and contracted execution.\n"
    "• Difference-in-Differences (DiD) Estimator: Compares treated commodity lines against unhedged spot control lines. "
    "A negative estimate is not proof by itself; assess parallel trends, confounders, and uncertainty."
)

# Section 7: Architecture Proposal
doc.add_heading("7. Questions 6 & 7: Architecture Proposal (Django 5 & FastAPI)", level=1)
doc.add_paragraph(
    "Proposed decoupled service architecture (deployment behavior and performance are unvalidated):\n"
    "• Django example: ORM models, REST Framework viewsets, and admin configuration.\n"
    "• FastAPI example: In-memory model loading, Pydantic validation, and OpenAPI documentation; response latency requires deployment benchmarking.\n"
    "• Redis caching and scheduled retraining are design options, not verified production services."
)

# Section 8: Frontend Client Integration
doc.add_heading("8. Question 8: Frontend Integration (Next.js 15 & React Native)", level=1)
doc.add_paragraph(
    "• Next.js 15 Web Dashboard (NextJs_AgriDashboard.tsx): Features real-time forward curves, KPI metric cards, an interactive "
    "Hedging Volume Simulator (100–5,000 Quintals) estimating forecast-price scenarios, and corporate action approval buttons.\n"
    "• React Native Offline Mobile App (ReactNative_AgriApp.tsx): Designed for field procurement officers in rural Karnataka mandis; "
    "implements local AsyncStorage caching for zero-downtime offline access and automated BUY DIP alert banners."
)

# Section 9: Monitoring & MLOps Proposal
doc.add_heading("9. Question 9: Monitoring, Drift Handling & Retraining Proposal", level=1)
doc.add_paragraph(
    "MLOps Governance Pipeline:\n"
    "1. Population Stability Index (PSI): Computed monthly. PSI < 0.1 indicates stability; PSI >= 0.20 triggers automatic retraining.\n"
    "2. Two-Sample Kolmogorov-Smirnov (KS) Test: Evaluates input feature distribution drift against baseline (p < 0.05 indicates drift).\n"
    "3. CI/CD Retraining Pipeline: Mandi Ingestion -> Anomaly Filter -> Drift Evaluation -> Celery Retrain -> Walk-Forward Holdout Check -> "
    "Canary Shadow Deployment (10% traffic) -> Automated Promotion in ModelRegistry."
)

# Section 10: Submission Verification
doc.add_heading("10. Supporting Artifacts", level=1)
doc.add_paragraph(
    "Project artifacts:\n"
    "• Executed Jupyter Notebook: notebooks/AgriPrice_Forecasting_Way2Agri.ipynb\n"
    "• Official Word Document: docs/Way2Agribusiness_Assignment_Report.docx\n"
    "• HTML Report: docs/AgriPrice_Forecasting_Way2Agri.html"
)

doc_path = os.path.join(os.path.dirname(__file__), "..", "docs", "Way2Agribusiness_Assignment_Report.docx")
doc.save(doc_path)
print(f"Senior Word Document Report successfully regenerated at: {doc_path}")
