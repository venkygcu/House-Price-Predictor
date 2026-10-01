"""Small desktop form for scoring one California Housing input row."""
from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
import json

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL = joblib.load(ROOT / "artifacts" / "california_housing_linear_regression.joblib")
FEATURES = list(MODEL.feature_names_in_)
DEFAULTS = MODEL.named_steps["imputer"].statistics_
METRICS = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))


def show_model_comparison() -> None:
    """Show the held-out Task 2 results and identify the deployed model."""
    window = tk.Toplevel(root)
    window.title("Model comparison")
    window.resizable(False, False)
    frame = ttk.Frame(window, padding=14)
    frame.grid(sticky="nsew")
    ttk.Label(
        frame,
        text=f"Selected model: {METRICS['selected_model']} (lowest test RMSE)",
        font=("Segoe UI", 10, "bold"),
    ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))

    columns = ("model", "mae_100k_usd", "rmse_100k_usd", "r2")
    table = ttk.Treeview(frame, columns=columns, show="headings", height=3)
    headings = {
        "model": ("Model", 160),
        "mae_100k_usd": ("MAE ($100k)", 100),
        "rmse_100k_usd": ("RMSE ($100k)", 110),
        "r2": ("R-squared", 90),
    }
    for column, (title, width) in headings.items():
        table.heading(column, text=title)
        table.column(column, width=width, anchor="center")
    for row in METRICS["model_comparison"]:
        table.insert("", "end", values=(
            row["model"], f"{row['mae_100k_usd']:.4f}",
            f"{row['rmse_100k_usd']:.4f}", f"{row['r2']:.4f}",
        ))
    table.grid(row=1, column=0, columnspan=4)
    ttk.Label(
        frame,
        text="Scores are from the same held-out test split. Lower MAE/RMSE and higher R-squared are better.",
        wraplength=450,
    ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(9, 0))


def predict() -> None:
    try:
        values = {feature: float(entries[feature].get().strip()) for feature in FEATURES}
    except ValueError:
        messagebox.showerror("Invalid input", "Enter a number in every field.")
        return
    predicted_100k = float(MODEL.predict(pd.DataFrame([values]))[0])
    result.set(f"Predicted median value: ${predicted_100k * 100_000:,.0f} (1990 USD)")


def run_console_fallback() -> None:
    """Keep prediction and comparison available when this Python lacks Tk."""
    print("Tk is unavailable; starting the console predictor instead.")
    while True:
        choice = input("Choose: [1] predict, [2] compare models, [q] quit: ").strip().lower()
        if choice == "q":
            return
        if choice == "2":
            print(pd.DataFrame(METRICS["model_comparison"]).to_string(index=False))
            print(f"Selected model: {METRICS['selected_model']} (lowest held-out RMSE)")
            continue
        if choice != "1":
            print("Choose 1, 2, or q.")
            continue

        values = {}
        try:
            for feature, default in zip(FEATURES, DEFAULTS):
                raw = input(f"{feature} [{default:.4f}]: ").strip()
                values[feature] = float(raw) if raw else float(default)
        except ValueError:
            print("Enter a valid number for each feature; try again.")
            continue
        predicted_100k = float(MODEL.predict(pd.DataFrame([values]))[0])
        print(f"Predicted median value: ${predicted_100k * 100_000:,.0f} (1990 USD)")


try:
    root = tk.Tk()
except tk.TclError:
    run_console_fallback()
    raise SystemExit(0)

root.title("California House Price Predictor")
root.resizable(False, False)
frame = ttk.Frame(root, padding=18)
frame.grid()
ttk.Label(frame, text="California House Price Predictor", font=("Segoe UI", 15, "bold")).grid(column=0, row=0, columnspan=2, pady=(0, 6))
ttk.Label(frame, text="Enter the eight California Housing dataset features.").grid(column=0, row=1, columnspan=2, pady=(0, 4))
ttk.Label(frame, text=f"Active model: {METRICS['selected_model']}", foreground="#0f766e").grid(column=0, row=2, columnspan=2, pady=(0, 8))
entries: dict[str, ttk.Entry] = {}
for row, (feature, default) in enumerate(zip(FEATURES, DEFAULTS), start=3):
    ttk.Label(frame, text=feature).grid(column=0, row=row, sticky="w", padx=(0, 12), pady=3)
    entry = ttk.Entry(frame, width=22)
    entry.insert(0, f"{default:.4f}")
    entry.grid(column=1, row=row, pady=3)
    entries[feature] = entry
result = tk.StringVar(value="Enter values and select Predict Price.")
ttk.Button(frame, text="Predict Price", command=predict).grid(column=0, row=11, columnspan=2, pady=(14, 5))
ttk.Button(frame, text="Compare Models", command=show_model_comparison).grid(column=0, row=12, columnspan=2, pady=(0, 8))
ttk.Label(frame, textvariable=result, wraplength=350, font=("Segoe UI", 10, "bold")).grid(column=0, row=13, columnspan=2)
ttk.Label(frame, text="Educational dataset only; not for real property valuations.", foreground="#666666").grid(column=0, row=14, columnspan=2, pady=(12, 0))
root.mainloop()
