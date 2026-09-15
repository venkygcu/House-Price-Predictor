---
marp: true
theme: default
paginate: true
title: California Housing Price Predictor
---

# House Price Predictor

## Linear Regression on California Housing

**Goal:** demonstrate a reproducible end-to-end machine-learning workflow.

---

# Problem & data

- Predict median house value (`MedHouseVal`)
- California Housing dataset from scikit-learn
- 20,640 California block groups; 8 numeric features
- Target is measured in **$100,000s** (1990 USD)

Inputs include median income, house age, average rooms, population, latitude, and longitude.

---

# ML workflow

```text
Load → explore → preprocess → split → train → evaluate → report
```

- Numeric coercion changes invalid/blank values to missing values
- Median imputation runs inside the scikit-learn pipeline
- 80% training / 20% test split with `random_state=42`
- Baseline estimator: `LinearRegression`

---

# Exploratory data analysis

- Summary statistics and missing-value counts are exported to `reports/`
- Distribution charts show the spread of every feature and the target
- Correlation chart identifies relationships with median price
- Median income and geographic position are especially informative signals
- The target has an upper cap, which limits a linear model at high prices

---

# Evaluation

Run `python house_price_predictor.py` to generate current metrics.

- **MAE:** mean absolute prediction error
- **RMSE:** penalizes large prediction errors more strongly
- **R²:** variation in the test target explained by the model
- Actual-vs-predicted and residual plots diagnose model error

---

# Outputs

- Fitted pipeline: `artifacts/california_housing_linear_regression.joblib`
- EDA tables, metrics, predictions, and report: `reports/`
- Visuals: distribution, correlation, coefficient, actual-vs-predicted, and residual charts
- Notebook: `notebooks/house_price_predictor.ipynb`

---

# Results & next steps

- Linear regression is a clear, interpretable baseline.
- Compare Ridge/Lasso and tree ensembles with cross-validation next.
- Use current, property-level data before any real-world valuation.
- Assess local errors, bias, and data drift before deployment.
