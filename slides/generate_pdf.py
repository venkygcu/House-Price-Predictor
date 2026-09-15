"""Create a compact PDF slide deck from the latest model metrics."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent.parent
METRICS = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
OUTPUT = Path(__file__).with_name("california_housing_presentation.pdf")

slides = [
    ("House Price Predictor", ["Linear Regression on the California Housing dataset", "End-to-end machine-learning portfolio project"]),
    ("Problem and data", ["Predict median house value (MedHouseVal)", "20,640 California block groups and 8 numeric features", "Target unit: $100,000s of 1990 USD"]),
    ("Workflow", ["Load and explore data", "Coerce invalid/blank entries; median-impute missing values", "80/20 train-test split with random_state=42", "Train a LinearRegression pipeline"]),
    ("Exploration", ["Summary statistics and missing-value counts", "Feature and target distributions", "Correlation with price and geography", "Diagnostic plots: actual vs predicted and residuals"]),
    ("Held-out evaluation", [f"MAE: {METRICS['mae_100k_usd']} $100k", f"RMSE: {METRICS['rmse_100k_usd']} $100k", f"R²: {METRICS['r2']}", "Metrics calculated on 4,128 untouched test rows"]),
    ("Outputs", ["Saved preprocessing + model pipeline (.joblib)", "Metrics, predictions, EDA tables, and report", "Notebook plus editable Marp slide source"]),
    ("Next steps", ["Compare Ridge/Lasso and tree-based models", "Use cross-validation and current property-level data", "Evaluate bias, drift, and local error before deployment"]),
]

with PdfPages(OUTPUT) as pdf:
    for title, bullets in slides:
        fig = plt.figure(figsize=(13.333, 7.5), facecolor="#f8fafc")
        fig.text(0.08, 0.82, title, fontsize=30, fontweight="bold", color="#0f172a")
        fig.text(0.08, 0.73, "California Housing • Linear Regression", fontsize=13, color="#0f766e")
        fig.text(0.12, 0.58, "\n\n".join(f"• {bullet}" for bullet in bullets), fontsize=19, color="#1e293b", va="top")
        fig.text(0.08, 0.06, "Educational baseline — not for real-estate valuation decisions", fontsize=10, color="#64748b")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

print(f"Created {OUTPUT}")
