# Code Running Instructions

## Appliance Energy Forecasting Project

These instructions reproduce the final analysis and generated outputs.

### 1. Requirements

Python 3.12 was used for the final verified run. Install dependencies with:

```bash
pip install -r requirements.txt
```

An internet connection may be required on the first Chronos run because pretrained Amazon Chronos-T5 Small weights may need to be downloaded.

### 2. Run the tests

```bash
python -m pytest
```

Final verification result:

```text
4 passed
```

### 3. Run the complete analysis

```bash
python scripts/run_full_analysis.py
```

The pipeline performs preprocessing, hourly resampling, exploratory analysis, benchmark forecasting, SARIMAX modelling, feature engineering, XGBoost forecasting, Chronos-T5 Small forecasting, evaluation, and output generation.

### 4. Forecast design

```text
Model-development holdout: 336 hours (14 days)
Final forecast horizon:     24 hours
```

The two periods are intentionally distinguished. The 14-day holdout supports longer out-of-sample model comparison, while the final 24-hour window is the common short-horizon forecast required by the assignment.

### 5. Generated outputs

After a successful run, the project generates:

```text
outputs/analysis_summary.json

outputs/figures/forecast_comparison.png
outputs/figures/forecast_final_24h.png
outputs/figures/full_series.png
outputs/figures/hourly_profile.png
outputs/figures/sarimax_residual_acf.png
outputs/figures/sarimax_residual_hist.png
outputs/figures/target_acf.png
outputs/figures/xgboost_conditional_feature_importance.png
outputs/figures/xgboost_operational_feature_importance.png

outputs/forecasts/all_forecasts.csv
outputs/forecasts/final_24h_forecasts.csv
outputs/forecasts/sarimax_confidence_intervals.csv

outputs/metrics/model_comparison.csv
outputs/metrics/model_comparison_24h.csv
outputs/metrics/sarimax_aic_grid.csv
```

### 6. SARIMAX grid search

The required search evaluates:

```text
p = 0..6
d = 0..2
q = 0..6
```

for 147 combinations. The best converged specification in the final analysis is:

```text
SARIMAX(4,1,1)(1,0,1)[24]
AIC = 32541.768613
```

### 7. Final reproducible 24-hour results

| Model | MAE | RMSE | MASE | Bias |
|---|---:|---:|---:|---:|
| XGBoost Conditional | 52.97 | 71.69 | 1.003 | -1.73 |
| XGBoost Operational | 50.18 | 82.87 | 0.950 | -39.56 |
| SARIMAX Conditional | 60.74 | 88.89 | 1.150 | -21.36 |
| Chronos-T5 Small | 62.82 | 103.82 | 1.189 | -61.27 |
| Mean | 74.51 | 107.63 | 1.411 | -42.60 |
| Daily Seasonal Naive | 63.68 | 113.64 | 1.206 | 4.65 |
| Weekly Seasonal Naive | 74.31 | 117.97 | 1.407 | -71.53 |
| Naive | 111.74 | 124.03 | 2.116 | 74.93 |
| Drift | 111.97 | 124.26 | 2.120 | 75.54 |

The Chronos forecast is made reproducible by setting the Python, NumPy and PyTorch random seeds.

### 8. Operational versus conditional forecasts

Operational XGBoost uses information available at forecast origin, including historical appliance demand and calendar/lag/rolling features.

Conditional models may use realised future sensor or weather values from the test period. These should not be interpreted as fully operational forecasts unless those covariates are themselves forecast or otherwise known in advance.

### 9. Troubleshooting

Check syntax:

```bash
python -m py_compile scripts/run_full_analysis.py
```

Run tests:

```bash
python -m pytest
```

A `ConvergenceWarning` from statsmodels may appear during SARIMAX fitting. The final run can still complete successfully; convergence status is recorded during the AIC search.

An unauthenticated Hugging Face Hub warning may also appear when Chronos weights are loaded. It does not indicate a failed forecast.

### 10. Quick start

```bash
pip install -r requirements.txt
python -m pytest
python scripts/run_full_analysis.py
```

Inspect `outputs/` after completion.
