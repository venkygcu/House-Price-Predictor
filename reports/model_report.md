# California Housing Model Comparison Report

- Dataset: 20,640 California block groups; 8 numeric input features.
- Split: 16,512 training / 4,128 test rows (random_state=42).
- Preprocessing: non-numeric/blank values are coerced to missing and median-imputed in the pipeline.
- Preprocessing: median imputation and StandardScaler are fitted within each training pipeline.
- Models: Linear Regression, Ridge Regression (alpha=1), and Decision Tree (max_depth=5).
- Selected model: Decision Tree (lowest held-out RMSE among candidates).
- MAE: 0.5223 $100k.
- RMSE: 0.7242 $100k.
- R²: 0.5997.

## Held-out model comparison

            model  mae_100k_usd  rmse_100k_usd     r2
    Decision Tree        0.5223         0.7242 0.5997
 Ridge Regression        0.5332         0.7456 0.5758
Linear Regression        0.5332         0.7456 0.5758

The target represents median value in $100,000s of 1990 USD; this is an educational baseline, not a real-estate valuation tool.
