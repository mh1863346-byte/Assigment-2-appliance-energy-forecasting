# Appliance Energy Forecasting

This repository contains the complete reproducible code and outputs for a time-series forecasting study of household appliance energy consumption using the **UCI Appliances Energy Prediction dataset**.

The project compares classical benchmark forecasts, SARIMAX, feature-based XGBoost models, and the **Amazon Chronos-T5 Small** time-series foundation model.

The original 10-minute observations are resampled to hourly values. A **14-day holdout period (336 hours)** is used for model-development evaluation, while the final common forecasting comparison uses a **24-hour horizon**.

## Project objectives

The analysis aims to:

- prepare and explore the appliance-energy dataset;
- investigate seasonality, autocorrelation and stationarity;
- compare mean, naive, daily seasonal naive, weekly seasonal naive and drift forecasts;
- select a SARIMAX model using the required AIC parameter search;
- add sensor, weather, calendar, lag and rolling-window covariates;
- build operational and conditional XGBoost forecasts;
- evaluate Amazon Chronos-T5 Small as a time-series foundation model;
- compare all models using common accuracy metrics;
- distinguish true operational forecasts from conditional forecasts using future covariates;
- recommend a practical smart-home forecasting approach.

## Dataset

The project uses the **Appliances Energy Prediction** dataset from the UCI Machine Learning Repository.

The raw data contain **19,735 observations** sampled every 10 minutes. After hourly resampling, the processed dataset contains **3,290 observations**, covering:

```text
2016-01-11 17:00:00 to 2016-05-27 18:00:00
```

The forecasting target is:

```text
Appliances
```

The dataset also contains indoor temperature and relative-humidity measurements, outdoor weather variables, lighting energy use and timestamp information.

The preprocessing pipeline:

1. parses the timestamp;
2. sorts observations chronologically;
3. converts variables to numeric form;
4. resamples the 10-minute data to hourly means;
5. interpolates missing hourly values where required;
6. checks for remaining missing values;
7. writes the processed hourly dataset.

The final processed data contain **0 missing values**.

## Forecasting design

Two evaluation periods are deliberately distinguished.

### 14-day holdout

```text
336 hours
```

The final 14 days are used as the main model-development holdout for evaluating benchmark, SARIMAX and feature-based models over a longer out-of-sample period.

### Final 24-hour forecast

```text
24 hours
```

The final short-horizon comparison follows the assignment requirement and includes the benchmark models, SARIMAX, XGBoost and Chronos-T5 Small.

The time series is never randomly shuffled because that would allow future information to leak into training.

## Exploratory analysis and stationarity

The analysis includes:

- complete hourly time-series plots;
- average appliance use by hour of day;
- summary statistics;
- ACF analysis;
- stationarity testing with the Augmented Dickey-Fuller test;
- differencing considerations;
- residual diagnostics after SARIMAX fitting.

The final ADF result is:

```text
ADF statistic = -9.016389
p-value       = 5.9349e-15
```

This strongly rejects the unit-root null hypothesis for the analysed training series. Differencing is nevertheless considered during SARIMAX selection because model likelihood and residual dependence also influence the appropriate specification.

## Benchmark models

Five benchmark forecasts are implemented:

- **Mean** — historical training mean.
- **Naive** — most recent observed value.
- **Daily seasonal naive** — same hour from the previous day (`lag = 24`).
- **Weekly seasonal naive** — same hour from the previous week (`lag = 168`).
- **Drift** — extrapolates the average change across the training period.

These models provide transparent reference points against which more complex methods are judged.

## SARIMAX

Daily seasonality is represented using:

```text
seasonal period = 24
```

The assignment-required AIC search evaluates:

```text
p = 0..6
d = 0..2
q = 0..6
```

giving:

```text
7 × 3 × 7 = 147 parameter combinations
```

The final search produced **27 converged models**. The best converged specification was:

```text
SARIMAX(4,1,1)(1,0,1)[24]
AIC = 32541.768613
```

The SARIMAX analysis also generates residual ACF and residual-distribution diagnostics plus forecast confidence intervals.

The conditional SARIMAX model uses:

```text
T_out
RH_out
Windspeed
Visibility
Tdewpoint
```

Because realised future weather values are not necessarily available at forecast origin, this result is explicitly described as a **conditional forecast**.

## Feature engineering

The XGBoost models use engineered predictors including:

### Calendar features

```text
hour
day of week
weekend indicator
hour_sin
hour_cos
dow_sin
dow_cos
```

### Appliance-demand lags

```text
1, 2, 3, 6, 12, 24, 48 and 168 hours
```

### Rolling features

Rolling means and standard deviations are constructed from shifted historical target values using windows including:

```text
3, 6, 12, 24 and 168 hours
```

Shifting before rolling prevents future target leakage.

### Sensor and weather variables

The conditional feature model can additionally use indoor temperature, humidity and outdoor/weather measurements.

## XGBoost models

Two XGBoost variants are evaluated.

### XGBoost operational

Uses calendar information and historical target-derived features that are available at the forecast origin.

### XGBoost conditional

Additionally uses sensor/weather information from the forecast period.

The conditional version can be useful for scenario analysis, but it is not a strict operational forecast unless those future covariates are themselves forecast or otherwise known.

## Chronos foundation model

The foundation-model component uses:

```text
Amazon Chronos-T5 Small
```

Chronos is evaluated over the final **24-hour forecast horizon**.

For reproducibility, the final pipeline fixes the Python, NumPy and PyTorch random seeds. Repeated verified runs produced the same final Chronos metrics.

## Evaluation metrics

All models are compared using:

- **MAE** — mean absolute error;
- **RMSE** — root mean squared error;
- **MASE** — mean absolute scaled error;
- **Bias** — average signed forecast error.

Lower MAE, RMSE and MASE are preferred. Bias values closer to zero indicate less systematic over- or under-prediction.

## Final reproducible 24-hour results

| Model | MAE | RMSE | MASE | Bias |
|---|---:|---:|---:|---:|
| XGBoost Conditional | 52.97 | **71.69** | 1.003 | -1.73 |
| XGBoost Operational | **50.18** | 82.87 | **0.950** | -39.56 |
| SARIMAX Conditional | 60.74 | 88.89 | 1.150 | -21.36 |
| Chronos-T5 Small | 62.82 | 103.82 | 1.189 | -61.27 |
| Mean | 74.51 | 107.63 | 1.411 | -42.60 |
| Daily Seasonal Naive | 63.68 | 113.64 | 1.206 | 4.65 |
| Weekly Seasonal Naive | 74.31 | 117.97 | 1.407 | -71.53 |
| Naive | 111.74 | 124.03 | 2.116 | 74.93 |
| Drift | 111.97 | 124.26 | 2.120 | 75.54 |

The conditional XGBoost model has the lowest final 24-hour RMSE. Operational XGBoost has the lowest MAE and MASE while avoiding dependence on realised future sensor/weather values.

Chronos-T5 Small outperforms the simple benchmark models by RMSE but does not outperform SARIMAX or either XGBoost model.

## Main findings

1. Appliance demand contains exploitable short-lag, daily and weekly temporal structure.
2. Seasonal benchmarks are substantially stronger than naive and drift forecasts.
3. SARIMAX improves on the strongest seasonal benchmark in the final 24-hour evaluation.
4. Engineered lag, rolling and calendar features make XGBoost highly competitive.
5. Conditional XGBoost achieves the lowest final RMSE, but its future covariates must be interpreted carefully.
6. Operational XGBoost provides strong performance using information available at forecast origin.
7. Chronos provides a useful foundation-model comparison but does not outperform the best task-specific models.
8. Additional model complexity does not automatically produce better forecasts.

## Practical recommendation

**Operational XGBoost** is recommended for practical smart-home forecasting.

It offers a strong balance of:

- accuracy;
- deployability;
- nonlinear modelling capability;
- computational cost;
- use of predictors available at forecast time.

Conditional XGBoost has the lowest final RMSE, but its future sensor/weather inputs would require a separate forecasting pipeline.

SARIMAX remains useful where interpretability and explicit confidence intervals are priorities.

## Repository structure

```text
Assigment-2-appliance-energy-forecasting/
├── README.md
├── RUNNING_INSTRUCTIONS.md
├── requirements.txt
├── energydata_complete.csv
├── scripts/
│   ├── run_full_analysis.py
│   ├── run_pipeline.py
│   └── sarimax_grid_search.py
├── tests/
│   └── test_pipeline.py
├── data/
│   └── processed/
└── outputs/
    ├── analysis_summary.json
    ├── figures/
    ├── forecasts/
    └── metrics/
```

## Generated outputs

A successful final run generates:

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

## Installation

Python 3.12 was used for the final verified analysis.

Create a virtual environment if desired:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Chronos may download pretrained weights on its first execution, so internet access may be required.

## Running the tests

Run:

```bash
python -m pytest
```

Final verified result:

```text
4 passed
```

## Running the complete analysis

Run:

```bash
python scripts/run_full_analysis.py
```

The script performs the main workflow and writes the resulting figures, forecasts, metrics and summary information under `outputs/`.

For more detailed execution instructions and troubleshooting, see:

```text
RUNNING_INSTRUCTIONS.md
```

## Reproducibility

The final verified pipeline uses fixed random seeds for Python, NumPy, XGBoost and PyTorch/Chronos where applicable.

The final configuration is:

```text
Holdout period:        336 hours (14 days)
Final forecast horizon: 24 hours
```

The final automated test suite passes and repeated Chronos runs produced identical final metrics.

## Limitations and future work

The dataset covers a single household and only several months, limiting generalisability. Household demand also contains irregular occupant-driven peaks that are difficult to predict.

Further work could:

- use rolling-origin evaluation over multiple 24-hour windows;
- forecast exogenous weather/sensor variables rather than use realised values;
- tune XGBoost using time-series cross-validation;
- investigate probabilistic forecast calibration;
- compare additional or larger foundation models;
- test generalisation across multiple households and seasons.

## References

Candanedo, L. M., Feldheim, V. and Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. *Energy and Buildings*, 140, 81–97.

Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

Ansari, A. F. et al. (2024). Chronos: Learning the Language of Time Series. *arXiv:2403.07815*.

Hyndman, R. J. and Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. 3rd ed. OTexts.

UCI Machine Learning Repository. *Appliances Energy Prediction Dataset*.

## Author

Time-Series Forecasting Assignment  
University of Hertfordshire
