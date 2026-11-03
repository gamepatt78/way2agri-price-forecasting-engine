"""
Generate executive-grade Word document (.docx) report for Way2Agribusiness screening submission.
"""
import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

doc = docx.Document()

# Set page margins
sections = doc.sections
for s in sections:
    s.top_margin = Inches(1)
    s.bottom_margin = Inches(1)
    s.left_margin = Inches(1)
    s.right_margin = Inches(1)

# Helper for cell shading
def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

# Title Header
p_title = doc.add_paragraph()
r_sub = p_title.add_run("WAY2AGRIBUSINESS INDIA PVT LTD\n")
r_sub.font.size = Pt(12)
r_sub.font.bold = True
r_sub.font.color.rgb = RGBColor(16, 124, 65)

r_main = p_title.add_run("AI/ML Developer Screening Assignment Report\n")
r_main.font.size = Pt(22)
r_main.font.bold = True
r_main.font.color.rgb = RGBColor(26, 43, 76)

r_tag = p_title.add_run("12-Month Commodity Price Forecasting Assignment Prototype & Architecture Examples")
r_tag.font.size = Pt(12)
r_tag.font.italic = True
r_tag.font.color.rgb = RGBColor(100, 116, 139)

doc.add_paragraph() # spacer

# Metadata Table
meta_table = doc.add_table(rows=3, cols=2)
meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
meta_table.autofit = False

data_meta = [
    ("Assessment", "Way2Agribusiness AI/ML Developer screening assignment"),
    ("Evaluation Focus", "Django, AI/ML, FastAPI, React Native, Next.js"),
    ("Project Type", "Assignment-scale prototype and architecture examples")
]

for i, (k, v) in enumerate(data_meta):
    row = meta_table.rows[i]
    c0 = row.cells[0]
    c1 = row.cells[1]
    c0.width = Inches(2.2)
    c1.width = Inches(4.3)
    c0.paragraphs[0].add_run(k).bold = True
    c1.paragraphs[0].add_run(v)
    set_cell_background(c0, "F1F5F9")
    set_cell_background(c1, "FFFFFF")

doc.add_paragraph()

# Executive Summary
doc.add_heading("1. Executive Summary", level=1)
p_exec = doc.add_paragraph(
    "This technical report presents an assignment-scale solution for the AI/ML Developer screening assignment "
    "at Way2Agribusiness India Pvt Ltd. The objective is to forecast average monthly commodity prices for the next 12 months "
    "(October 2025 to September 2026) using historical prices from 2005 to 2025 (249 monthly observations), with architecture "
    "examples for Django, FastAPI, Next.js, and React Native.\n\n"
    "Key Technical Achievements:\n"
    "• Investigated a 30-month price discontinuity; the supplied values do not prove an exact duplicate or establish the cause. Interpolation is presented as a provisional modeling scenario.\n"
    "• Benchmarked 6 models on one 12-month holdout; XGBoost achieved 9.33% MAPE and the 50/50 hybrid blend achieved 9.43%.\n"
    "• The saved forecast uses a SARIMAX-XGBoost blend; the holdout does not show that it outperforms XGBoost.\n"
    "• Projected the 12-month forward curve: identifying a seasonal dip in February 2026 (INR 14,442) and summer peak in July 2026 (INR 16,140).\n"
    "• Outlined procurement actions and a measurement framework; financial impact depends on realized prices and verified costs.\n"
    "• Included a Django 5 backend example with ORM models, REST Framework, and admin configuration.\n"
    "• Included a FastAPI inference example with Pydantic schemas and automatic OpenAPI documentation.\n"
    "• Included Next.js 15 and React Native frontend examples; deployment and performance remain unvalidated."
)

# Question 1: Data Cleaning & Anomaly Detection
doc.add_heading("2. Question 1: Data Cleaning & Anomaly Detection", level=1)
doc.add_paragraph(
    "Exploratory data analysis of the 249 monthly records (2005-01-01 to 2025-09-01) confirmed zero missing dates in sequence. "
    "However, diagnostic profiling revealed an abrupt 53.3% price collapse in June 2021 (dropping from INR 10,863 to INR 5,070).\n\n"
    "The 30 prices from June 2021 through November 2023 were compared with the same-length historical window from October 2007 through "
    "March 2010. Eleven of 30 values differ, and a scan of the supplied series found no exact matching 30-month window. The cause is "
    "unknown and should be checked against primary source records.\n\n"
    "For sensitivity analysis, the notebook flags the 30-month period and interpolates between May 2021 (INR 10,863) and December 2023 "
    "(INR 16,163), retaining the raw prices. This is a provisional modeling scenario, not a confirmed correction."
)

# Question 2 & 3: Model Architecture & Evaluation
doc.add_heading("3. Questions 2 & 3: Model Architecture & Performance Evaluation", level=1)
doc.add_paragraph(
    "To capture both autoregressive temporal dependency and annual crop seasonality, we engineered 15 features (lags t-1, t-2, t-3, "
    "t-6, t-12; rolling statistics 3, 6, 12; and cyclical sine/cosine calendar encodings). We benchmarked our models on a 12-month "
    "single 12-month holdout set (October 2024 to September 2025):"
)

# Results Table
res_table = doc.add_table(rows=7, cols=5)
res_table.alignment = WD_TABLE_ALIGNMENT.CENTER
headers = ["Model Paradigm", "Model Architecture", "MAE (INR)", "RMSE (INR)", "MAPE (%)"]
for j, h in enumerate(headers):
    cell = res_table.rows[0].cells[j]
    cell.paragraphs[0].add_run(h).bold = True
    set_cell_background(cell, "E2E8F0")

results_data = [
    ("Baseline", "Seasonal Naive (Lag 12)", "1,723.51", "1,956.70", "13.14%"),
    ("Machine Learning", "Random Forest Regressor", "1,502.83", "1,967.29", "12.02%"),
    ("Statistical", "SARIMAX(1,1,1)x(1,1,1)[12]", "1,223.49", "1,606.19", "9.73%"),
    ("Exponential Smoothing", "Holt-Winters (Additive)", "1,243.12", "1,610.39", "9.87%"),
    ("Gradient Boosting", "XGBoost Regressor", "1,186.72", "1,568.62", "9.33%"),
    ("Forecast Candidate", "50/50 Hybrid (SARIMAX + XGB)", "1,190.16", "1,572.02", "9.43%")
]

for i, row in enumerate(results_data):
    for j, val in enumerate(row):
        c = res_table.rows[i+1].cells[j]
        run = c.paragraphs[0].add_run(val)
        if "Ensemble" in row[0]:
            run.bold = True
            set_cell_background(c, "DCFCE7")
        elif i % 2 == 1:
            set_cell_background(c, "F8FAFC")

doc.add_paragraph()

# 12-Month Out-of-Sample Price Forecast Table
doc.add_heading("4. Official 12-Month Out-of-Sample Future Price Forecast", level=1)
doc.add_paragraph(
    "Using the full 2005-2025 historical dataset, the 50/50 SARIMAX-XGBoost forecast blend generated the following 12-month "
    "forward curve (October 2025 to September 2026) with approximate 95% prediction intervals:"
)

fc_table = doc.add_table(rows=13, cols=6)
fc_table.alignment = WD_TABLE_ALIGNMENT.CENTER
fc_headers = ["Horizon", "Target Month", "Predicted Price", "95% Lower Bound", "95% Upper Bound", "Strategic Signal"]
for j, h in enumerate(fc_headers):
    cell = fc_table.rows[0].cells[j]
    cell.paragraphs[0].add_run(h).bold = True
    set_cell_background(cell, "E2E8F0")

forecast_data = [
    ("M+1", "2025-10-01", "INR 15,771.46", "INR 13,770.77", "INR 17,772.15", "Stable"),
    ("M+2", "2025-11-01", "INR 15,786.11", "INR 13,706.94", "INR 17,865.28", "Stable"),
    ("M+3", "2025-12-01", "INR 15,430.61", "INR 13,275.81", "INR 17,585.41", "Stable"),
    ("M+4", "2026-01-01", "INR 15,271.50", "INR 13,043.63", "INR 17,499.37", "Begin Pre-Buy"),
    ("M+5", "2026-02-01", "INR 14,442.23", "INR 12,143.62", "INR 16,740.84", "Seasonal Dip (BUY/LOCK)"),
    ("M+6", "2026-03-01", "INR 15,299.42", "INR 12,932.18", "INR 17,666.66", "Recovery Phase"),
    ("M+7", "2026-04-01", "INR 15,442.30", "INR 13,008.36", "INR 17,876.24", "Rising Trend"),
    ("M+8", "2026-05-01", "INR 15,885.80", "INR 13,386.95", "INR 18,384.65", "Catalog Price Hike"),
    ("M+9", "2026-06-01", "INR 15,858.49", "INR 13,296.36", "INR 18,420.62", "Elevated Plateaus"),
    ("M+10", "2026-07-01", "INR 16,140.03", "INR 13,516.16", "INR 18,763.90", "Seasonal Peak (HEDGE)"),
    ("M+11", "2026-08-01", "INR 15,357.19", "INR 12,672.99", "INR 18,041.39", "Interim Moderation"),
    ("M+12", "2026-09-01", "INR 16,159.63", "INR 13,416.43", "INR 18,902.83", "Late Kharif Peak")
]

for i, row in enumerate(forecast_data):
    for j, val in enumerate(row):
        c = fc_table.rows[i+1].cells[j]
        run = c.paragraphs[0].add_run(val)
        if "BUY/LOCK" in row[5]:
            run.bold = True
            set_cell_background(c, "FEF08A") # yellow
        elif "Peak" in row[5]:
            run.bold = True
            set_cell_background(c, "FEE2E2") # red
        elif i % 2 == 1:
            set_cell_background(c, "F8FAFC")

doc.add_paragraph()

# Questions 4 & 5: Corporate Strategic Actions & Effectiveness
doc.add_heading("5. Questions 4 & 5: Corporate Strategic Responses & Effectiveness Measurement", level=1)
doc.add_paragraph(
    "Strategic Interventions:\n"
    "1. Evaluate forward contracting with Farmer Producer Organizations (FPOs) against supplier quotes, delivery terms, and quality requirements.\n"
    "2. Evaluate warehouse buffering against documented insurance, handling, spoilage, and financing costs.\n"
    "3. Review catalog pricing against realized input costs, customer demand, and contractual constraints.\n"
    "4. Cross-Mandi Arbitrage: Direct purchase orders across Karnataka, Maharashtra, and AP mandis based on transmission lag.\n\n"
    "Measuring Effectiveness:\n"
    "• Procurement Cost Variance (PCV): Quantify the INR difference between spot market cost and executed forward contract prices.\n"
    "• Gross Margin Stability: Set a margin-stability threshold using a documented business baseline before a pilot.\n"
    "• Difference-in-Differences (DiD): Compare treated procurement lines against unhedged spot control commodity lines."
)

# Questions 6 & 7: Django & FastAPI Architecture Proposal
doc.add_heading("6. Questions 6 & 7: Django & FastAPI Architecture Proposal", level=1)
doc.add_paragraph(
    "Proposed dual-service architecture (deployment behavior and performance are unvalidated):\n"
    "• Django example: Database models, Django REST Framework viewsets, and admin configuration.\n"
    "• FastAPI example: In-memory model loading, Pydantic validation, and automatic OpenAPI documentation; latency requires deployment benchmarking.\n"
    "• Redis caching and scheduled retraining are design options and are not represented as measured production services."
)

# Question 8: Frontend Integration (Next.js & React Native)
doc.add_heading("7. Question 8: Frontend Integration (Next.js & React Native)", level=1)
doc.add_paragraph(
    "• Next.js 15 Web Dashboard example (NextJs_AgriDashboard.tsx) providing forward curve graphs, "
    "interactive procurement volume sliders (100-5000 quintals) calculating projected cost variance, and corporate action approval buttons.\n"
    "• React Native Mobile App: Field officer procurement tool (ReactNative_AgriApp.tsx) featuring offline caching with AsyncStorage, "
    "pull-to-refresh synchronization, and automated visual BUY DIP alerts when visiting regional mandis."
)

# Question 9: Model Monitoring & Drift Detection
doc.add_heading("8. Question 9: Production Monitoring, Model Drift & Updating", level=1)
doc.add_paragraph(
    "Proposed monitoring protocol (not yet implemented):\n"
    "1. Track data and forecast errors over time; choose drift thresholds based on validation data and business risk.\n"
    "2. Compare candidate models with a time-aware validation plan before deployment.\n"
    "3. Require human review, rollback criteria, and operational monitoring before any canary release or model promotion."
)

# Supporting Artifacts
doc.add_heading("9. Supporting Artifacts", level=1)
doc.add_paragraph(
    "Project artifacts:\n"
    "1. AgriPrice_Forecasting_Way2Agri.ipynb (Full executed Jupyter Notebook)\n"
    "2. Django and FastAPI examples in the project repository (optional supporting material)\n"
    "3. Next.js and React Native frontend examples in the project repository (optional supporting material)\n"
    "4. Technical Word Report (optional supporting material)"
)

doc_path = os.path.join(os.path.dirname(__file__), "..", "docs", "Way2Agribusiness_Assignment_Report.docx")
doc.save(doc_path)
print(f"Executive Word Document Report saved successfully at: {doc_path}")
