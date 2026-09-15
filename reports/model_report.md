# California Housing Linear Regression Report

- Dataset: 20,640 California block groups; 8 numeric input features.
- Split: 16,512 training / 4,128 test rows (random_state=42).
- Preprocessing: non-numeric/blank values are coerced to missing and median-imputed in the pipeline.
- Model: LinearRegression baseline.
- MAE: 0.5332 $100k.
- RMSE: 0.7456 $100k.
- R²: 0.5758.

The target represents median value in $100,000s of 1990 USD; this is an educational baseline, not a real-estate valuation tool.
