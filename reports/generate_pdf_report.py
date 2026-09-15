"""Generate a concise three-page PDF report from the latest project outputs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
OUTPUT = REPORTS / "california_housing_model_report.pdf"
metrics = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))


def add_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.07, 0.92, title, fontsize=23, fontweight="bold", color="#0f172a")
    fig.text(0.07, 0.875, subtitle, fontsize=11, color="#0f766e")


with PdfPages(OUTPUT) as pdf:
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="#f8fafc")
    add_header(fig, "California Housing: Linear Regression", "Short model report")
    fig.text(0.08, 0.76, "Objective", fontsize=16, fontweight="bold", color="#0f172a")
    fig.text(0.08, 0.67, "Predict median owner-occupied house value using eight\nCalifornia block-group features from scikit-learn's dataset.", fontsize=13, color="#334155")
    fig.text(0.08, 0.53, "Method", fontsize=16, fontweight="bold", color="#0f172a")
    fig.text(0.08, 0.35, "• 20,640 rows; 80/20 train-test split (seed 42)\n• Invalid/blank numeric values coerced to missing\n• Median imputation + LinearRegression in one pipeline\n• Evaluation performed on 4,128 held-out rows", fontsize=13, color="#334155", linespacing=1.7)
    fig.text(0.08, 0.16, f"MAE   {metrics['mae_100k_usd']:.4f} $100k     RMSE   {metrics['rmse_100k_usd']:.4f} $100k     R²   {metrics['r2']:.4f}", fontsize=15, fontweight="bold", color="#0f766e")
    fig.text(0.08, 0.07, "Target values are in $100,000s of 1990 USD. This is an educational baseline, not a valuation tool.", fontsize=9, color="#64748b")
    pdf.savefig(fig); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69), facecolor="#f8fafc")
    fig.subplots_adjust(top=.86, hspace=.35)
    add_header(fig, "Exploratory data analysis", "Distributions and feature relationships")
    axes[0].imshow(mpimg.imread(FIGURES / "distributions.png")); axes[0].axis("off")
    axes[1].imshow(mpimg.imread(FIGURES / "target_correlations.png")); axes[1].axis("off")
    fig.text(.08, .05, "Median income and location-related features show useful association with the target; the target's cap makes the high-value range harder to model.", fontsize=10, color="#334155")
    pdf.savefig(fig); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69), facecolor="#f8fafc")
    fig.subplots_adjust(top=.86, hspace=.32)
    add_header(fig, "Evaluation and improvement ideas", "Held-out predictions and residual diagnostics")
    axes[0].imshow(mpimg.imread(FIGURES / "actual_vs_predicted.png")); axes[0].axis("off")
    axes[1].imshow(mpimg.imread(FIGURES / "residuals.png")); axes[1].axis("off")
    fig.text(.08, .05, "Next: compare Ridge/Lasso and tree ensembles with cross-validation; use newer, property-level data and assess local error, bias, and drift.", fontsize=10, color="#334155")
    pdf.savefig(fig); plt.close(fig)

print(f"Created {OUTPUT}")
