# California House Price Predictor

A reproducible machine-learning project using scikit-learn's California Housing dataset. Task 2 adds feature scaling and a comparison of three regression models.

## Workflow

- Load and inspect the dataset; save summary statistics, missing-value counts, distributions, and target correlations.
- Split rows into 80% training and 20% test data (`random_state=42`).
- Use a pipeline with median imputation and `StandardScaler`; fit preprocessing on training rows only.
- Compare Linear Regression, Ridge Regression (`alpha=1.0`), and Decision Tree (`max_depth=5`) using held-out MAE, RMSE, and R-squared.
- Select the candidate with the lowest held-out RMSE and save its fitted pipeline.
- Generate actual-vs-predicted, residual, model comparison, and EDA charts.

## Run it

```powershell
python -m pip install -r requirements.txt
python house_price_predictor.py
python reports/generate_pdf_report.py
python slides/generate_pdf.py
jupyter notebook notebooks/task2_feature_engineering_model_comparison.ipynb
python predict_ui.py
```

The first run may download the public California Housing dataset through scikit-learn.

## Outputs

- `artifacts/california_housing_linear_regression.joblib` ? fitted preprocessing and selected model pipeline.
- `reports/metrics.json`, `reports/model_comparison.csv`, and `reports/model_report.md` — evaluation results and model selection.
- `reports/predictions.csv` ? held-out actual values, predictions, and residuals.
- `reports/summary_statistics.csv` and `reports/missing_values.csv` ? EDA tables.
- `reports/california_housing_model_report.pdf` ? generated model report.
- `reports/figures/` — EDA, model comparison, and diagnostic charts.
- `notebooks/task2_feature_engineering_model_comparison.ipynb` — guided Task 2 notebook.
- `notebooks/task1_ml_linear_regression.ipynb` ? prior Task 1 submission notebook.
- `slides/presentation.md` and `slides/california_housing_presentation.pdf` ? presentation source and PDF.
- `predict_ui.py` ? desktop form for scoring dataset-format inputs.

## Dataset and responsible use

The target, `MedHouseVal`, is the median owner-occupied value in $100,000s of 1990 USD. Features are aggregated at California block-group level, so this model is an educational baseline, not a real-estate valuation tool.
