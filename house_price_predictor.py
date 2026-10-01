"""Train, evaluate, and report a California Housing linear-regression model.

Run from the project root with: ``python house_price_predictor.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

# File-only plots keep the project runnable in terminals and CI without Tk.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def save_eda(data: pd.DataFrame) -> None:
    """Create compact, reproducible EDA tables and charts."""
    data.describe().T.to_csv(REPORTS / "summary_statistics.csv")
    data.isna().sum().rename("missing_values").to_csv(REPORTS / "missing_values.csv")

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for axis, column in zip(axes.flat, data.columns):
        values = data.loc[:, [column]].iloc[:, 0].dropna().to_numpy()
        axis.hist(values, bins=30, color="#2563eb", edgecolor="white")
        axis.set_title(column)
    axes.flat[-1].set_visible(False)
    fig.suptitle("Feature and target distributions", y=1.01, fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURES / "distributions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    correlations = data.corr(numeric_only=True)["MedHouseVal"].sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    correlations.plot.barh(ax=ax, color="#0f766e")
    ax.set(title="Correlation with median house value", xlabel="Pearson correlation")
    fig.tight_layout()
    fig.savefig(FIGURES / "target_correlations.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.style.use("ggplot")

    features, target = fetch_california_housing(as_frame=True, return_X_y=True)
    X = pd.DataFrame(features).apply(pd.to_numeric, errors="coerce")
    y = pd.Series(pd.to_numeric(target, errors="coerce"), name="MedHouseVal")
    data = pd.concat([X, y], axis=1)
    save_eda(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE),
    }
    fitted_models = {}
    comparison_rows = []
    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", estimator),
        ])
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test)
        fitted_models[name] = pipeline
        comparison_rows.append({
            "model": name,
            "mae_100k_usd": float(mean_absolute_error(y_test, predicted)),
            "rmse_100k_usd": float(mean_squared_error(y_test, predicted) ** 0.5),
            "r2": float(r2_score(y_test, predicted)),
        })

    comparison = pd.DataFrame(comparison_rows).sort_values("rmse_100k_usd").reset_index(drop=True)
    selected_name = str(comparison.loc[0, "model"])
    model = fitted_models[selected_name]
    predictions = pd.Series(model.predict(X_test), index=y_test.index, name="predicted")

    metrics = {
        "mae_100k_usd": round(float(mean_absolute_error(y_test, predictions)), 4),
        "rmse_100k_usd": round(float(mean_squared_error(y_test, predictions) ** 0.5), 4),
        "r2": round(float(r2_score(y_test, predictions)), 4),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": list(X.columns),
        "target_unit": "$100,000 (1990 USD)",
        "random_state": RANDOM_STATE,
        "selected_model": selected_name,
        "model_comparison": comparison.round(4).to_dict(orient="records"),
    }

    joblib.dump(model, ARTIFACTS / "california_housing_linear_regression.joblib")
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    comparison.round(4).to_csv(REPORTS / "model_comparison.csv", index=False)
    results = pd.DataFrame({"actual": y_test, "predicted": predictions})
    results["residual"] = results["actual"] - results["predicted"]
    results.to_csv(REPORTS / "predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, predictions, alpha=0.35, color="#2563eb", edgecolors="none")
    bounds = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
    ax.plot(bounds, bounds, "--", color="#ef4444", label="Perfect prediction")
    ax.set(xlabel="Actual value ($100k)", ylabel="Predicted value ($100k)", title="Actual vs. predicted")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(predictions, results["residual"], alpha=0.35, color="#7c3aed", edgecolors="none")
    ax.axhline(0, color="#ef4444", linestyle="--")
    ax.set(xlabel="Predicted value ($100k)", ylabel="Residual (actual - predicted)", title="Residual plot")
    fig.tight_layout()
    fig.savefig(FIGURES / "residuals.png", dpi=160)
    plt.close(fig)

    if hasattr(model.named_steps["regressor"], "coef_"):
        coefficients = pd.Series(model.named_steps["regressor"].coef_, index=X.columns).sort_values()
        fig, ax = plt.subplots(figsize=(7, 5))
        coefficients.plot.barh(ax=ax, color="#0f766e", title=f"{selected_name} coefficients (scaled features)")
        fig.tight_layout()
        fig.savefig(FIGURES / "coefficients.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    comparison.set_index("model")["rmse_100k_usd"].plot.bar(ax=ax, color="#0f766e")
    ax.set(ylabel="RMSE ($100,000s)", title="Held-out model comparison (lower is better)")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison.png", dpi=160)
    plt.close(fig)

    (REPORTS / "model_report.md").write_text(
        "# California Housing Model Comparison Report\n\n"
        f"- Dataset: {len(data):,} California block groups; 8 numeric input features.\n"
        f"- Split: {len(X_train):,} training / {len(X_test):,} test rows (random_state={RANDOM_STATE}).\n"
        "- Preprocessing: non-numeric/blank values are coerced to missing and median-imputed in the pipeline.\n"
        "- Preprocessing: median imputation and StandardScaler are fitted within each training pipeline.\n"
        "- Models: Linear Regression, Ridge Regression (alpha=1), and Decision Tree (max_depth=5).\n"
        f"- Selected model: {selected_name} (lowest held-out RMSE among candidates).\n"
        f"- MAE: {metrics['mae_100k_usd']} $100k.\n"
        f"- RMSE: {metrics['rmse_100k_usd']} $100k.\n"
        f"- R²: {metrics['r2']}.\n\n"
        "## Held-out model comparison\n\n"
        + comparison.round(4).to_string(index=False)
        + "\n\nThe target represents median value in $100,000s of 1990 USD; this is an educational baseline, not a real-estate valuation tool.\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
