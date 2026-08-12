# scripts/run_demo_pipeline.py

# ============================================================
# Demo forecasting pipeline:
# Appliance Energy Prediction
#
# Models:
#   1. Benchmarks
#   2. SARIMAX
#   3. Feature-based model
#   4. Foundation model placeholder
#
# Output:
#   outputs/forecasts/all_forecasts.csv
#   outputs/metrics/model_comparison.csv
#   outputs/figures/forecast_comparison.png
# ============================================================

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

from statsmodels.tsa.statespace.sarimax import SARIMAX


# ------------------------------------------------------------
# 0. Configuration
# ------------------------------------------------------------

RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

PROCESSED_DIR = DATA_DIR / "processed"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURE_DIR = OUTPUT_DIR / "figures"

for path in [PROCESSED_DIR, FORECAST_DIR, METRICS_DIR, FIGURE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

TARGET = "Appliances"

# Hourly data:
# 24 observations = 1 day
# 168 observations = 1 week
DAILY_PERIOD = 24
WEEKLY_PERIOD = 168

# Use final 14 days as the test set
TEST_STEPS = 14 * 24


# ------------------------------------------------------------
# 1. Load and prepare data
# ------------------------------------------------------------

def load_appliance_data():
    """
    Load Appliances Energy Prediction data from the original UCI CSV.
    This version is more reliable because the timestamp column is preserved.
    """

    print("Downloading data from UCI...")

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"

    df = pd.read_csv(url)

    print("\nColumns in downloaded data:")
    print(df.columns.tolist())

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET])

    print("Original data shape:", df.shape)

    hourly = df.resample("h").mean()
    hourly = hourly.interpolate("time")
    hourly = hourly.dropna()

    print("Hourly data shape:", hourly.shape)

    hourly.to_csv(PROCESSED_DIR / "appliance_hourly.csv")

    return hourly
    
# ------------------------------------------------------------
# 2. Evaluation
# ------------------------------------------------------------

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def mase(y_true, y_pred, y_train, seasonality=24):
    """
    Mean absolute scaled error.

    Uses the in-sample seasonal naive forecast error as scale.
    """

    y_train = pd.Series(y_train).astype(float)

    seasonal_errors = np.abs(
        y_train.iloc[seasonality:].values
        - y_train.iloc[:-seasonality].values
    )

    scale = seasonal_errors.mean()

    if scale == 0:
        return np.nan

    return np.mean(np.abs(y_true - y_pred)) / scale


def evaluate_forecast(name, y_true, y_pred, y_train):
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred, index=y_true.index).astype(float)

    return {
        "model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MASE": mase(y_true, y_pred, y_train, seasonality=DAILY_PERIOD),
        "Bias": np.mean(y_pred - y_true),
    }


# ------------------------------------------------------------
# 3. Benchmark models
# ------------------------------------------------------------

def mean_forecast(y_train, horizon, index):
    return pd.Series(y_train.mean(), index=index, name="mean")


def naive_forecast(y_train, horizon, index):
    return pd.Series(y_train.iloc[-1], index=index, name="naive")


def seasonal_naive_forecast(y_train, horizon, index, seasonality):
    """
    Recursive seasonal naive forecast.

    For hourly data:
        seasonality=24 gives same hour yesterday.
        seasonality=168 gives same hour last week.
    """

    values = []

    history = list(y_train.values)

    for i in range(horizon):
        values.append(history[-seasonality])
        history.append(values[-1])

    return pd.Series(values, index=index)


def drift_forecast(y_train, horizon, index):
    slope = (y_train.iloc[-1] - y_train.iloc[0]) / (len(y_train) - 1)

    values = [
        y_train.iloc[-1] + slope * step
        for step in range(1, horizon + 1)
    ]

    return pd.Series(values, index=index, name="drift")


# ------------------------------------------------------------
# 4. SARIMAX model
# ------------------------------------------------------------

def fit_sarimax(y_train, X_train=None):
    """
    Fit a relatively simple SARIMAX model.

    Since we use hourly data, seasonal period 24 captures daily seasonality.
    Weekly seasonality is handled by benchmarks and feature models in this demo.
    """

    model = SARIMAX(
        y_train,
        exog=X_train,
        order=(1, 0, 1),
        seasonal_order=(1, 1, 1, 24),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fit = model.fit(disp=False)

    return fit


def forecast_sarimax(fit, horizon, index, X_test=None):
    fc = fit.get_forecast(
        steps=horizon,
        exog=X_test,
    )

    mean = fc.predicted_mean
    mean.index = index
    mean.name = "sarimax"

    return mean


# ------------------------------------------------------------
# 5. Feature engineering for machine learning
# ------------------------------------------------------------

def add_time_features(df):
    out = df.copy()

    out["hour"] = out.index.hour
    out["dayofweek"] = out.index.dayofweek
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)

    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)

    return out


def make_ml_table(df, target=TARGET):
    """
    Create a supervised learning table.

    Important:
    All lag and rolling features use only past target values.
    """

    out = add_time_features(df)

    # Lag features for hourly data
    lags = [
        1,      # 1 hour ago
        2,      # 2 hours ago
        3,      # 3 hours ago
        6,      # 6 hours ago
        12,     # 12 hours ago
        24,     # same hour yesterday
        48,     # two days ago
        168,    # same hour last week
    ]

    for lag in lags:
        out[f"lag_{lag}"] = out[target].shift(lag)

    # Rolling features
    windows = [
        3,
        6,
        12,
        24,
        168,
    ]

    for window in windows:
        out[f"roll_mean_{window}"] = (
            out[target]
            .shift(1)
            .rolling(window)
            .mean()
        )

        out[f"roll_std_{window}"] = (
            out[target]
            .shift(1)
            .rolling(window)
            .std()
        )

    return out.dropna()


def fit_feature_model(X_train, y_train):
    """
    Feature-based model.

    This uses sklearn's HistGradientBoostingRegressor so that the demo
    works without requiring xgboost. Students can replace this with XGBoost.
    """

    model = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.03,
        max_leaf_nodes=31,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    return model


def forecast_feature_model(model, X_test, index):
    pred = model.predict(X_test)

    return pd.Series(
        pred,
        index=index,
        name="feature_model",
    )


# ------------------------------------------------------------
# 6. Foundation model placeholder
# ------------------------------------------------------------

def forecast_foundation_model_placeholder(y_train, horizon, index):
    """
    Placeholder for a foundation-model forecast.

    Students should replace this with Chronos, TimesFM, or TimeGPT.

    For the demo pipeline, this returns the daily seasonal naive forecast.
    This allows the whole pipeline to run before students add the actual
    foundation model.

    Example replacements:
        - Chronos-2
        - TimeGPT
        - TimesFM
    """

    print(
        "\nFoundation model placeholder used. "
        "Replace this function with Chronos, TimesFM, or TimeGPT."
    )

    return seasonal_naive_forecast(
        y_train=y_train,
        horizon=horizon,
        index=index,
        seasonality=DAILY_PERIOD,
    ).rename("foundation_placeholder")


# ------------------------------------------------------------
# 7. Plotting
# ------------------------------------------------------------

def plot_forecasts(train, test, forecast_df):
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot final part of training data for context
    train.tail(14 * 24).plot(
        ax=ax,
        label="Training data",
        linewidth=1.5,
    )

    test.plot(
        ax=ax,
        label="Test data",
        linewidth=2.0,
        color="black",
    )

    for col in forecast_df.columns:
        if col != "actual":
            forecast_df[col].plot(
                ax=ax,
                label=col,
                alpha=0.9,
            )

    ax.set_title("Appliance energy forecasting")
    ax.set_ylabel("Appliance energy use")
    ax.set_xlabel("Date")
    ax.legend()

    fig.tight_layout()

    return fig


# ------------------------------------------------------------
# 8. Main pipeline
# ------------------------------------------------------------

def run_pipeline():
    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    data = load_appliance_data()

    y = data[TARGET]

    train = y.iloc[:-TEST_STEPS]
    test = y.iloc[-TEST_STEPS:]

    horizon = len(test)

    print("\nTrain period:")
    print(train.index.min(), "to", train.index.max())

    print("\nTest period:")
    print(test.index.min(), "to", test.index.max())

    forecasts = {}
    results = []

    # --------------------------------------------------------
    # Benchmarks
    # --------------------------------------------------------

    forecasts["mean"] = mean_forecast(
        y_train=train,
        horizon=horizon,
        index=test.index,
    )

    forecasts["naive"] = naive_forecast(
        y_train=train,
        horizon=horizon,
        index=test.index,
    )

    forecasts["seasonal_naive_daily"] = seasonal_naive_forecast(
        y_train=train,
        horizon=horizon,
        index=test.index,
        seasonality=DAILY_PERIOD,
    )

    forecasts["seasonal_naive_weekly"] = seasonal_naive_forecast(
        y_train=train,
        horizon=horizon,
        index=test.index,
        seasonality=WEEKLY_PERIOD,
    )

    forecasts["drift"] = drift_forecast(
        y_train=train,
        horizon=horizon,
        index=test.index,
    )

    # --------------------------------------------------------
    # SARIMAX with selected exogenous variables
    # --------------------------------------------------------

    candidate_exog_cols = [
        "T_out",
        "RH_out",
        "Windspeed",
        "Visibility",
        "Tdewpoint",
    ]

    exog_cols = [
        col for col in candidate_exog_cols
        if col in data.columns
    ]

    if len(exog_cols) > 0:
        X = data[exog_cols]

        X_train = X.iloc[:-TEST_STEPS]
        X_test = X.iloc[-TEST_STEPS:]

        print("\nSARIMAX exogenous columns:")
        print(exog_cols)
    else:
        X_train = None
        X_test = None

        print("\nNo SARIMAX exogenous columns found. Fitting target-only SARIMAX.")

    sarimax_fit = fit_sarimax(
        y_train=train,
        X_train=X_train,
    )

    forecasts["sarimax"] = forecast_sarimax(
        fit=sarimax_fit,
        horizon=horizon,
        index=test.index,
        X_test=X_test,
    )

    # --------------------------------------------------------
    # Feature-based model
    # --------------------------------------------------------

    ml_data = make_ml_table(data, target=TARGET)

    ml_train = ml_data.iloc[:-TEST_STEPS]
    ml_test = ml_data.iloc[-TEST_STEPS:]

    feature_cols = [
        col for col in ml_data.columns
        if col != TARGET
    ]

    X_train_ml = ml_train[feature_cols]
    y_train_ml = ml_train[TARGET]

    X_test_ml = ml_test[feature_cols]
    y_test_ml = ml_test[TARGET]

    feature_model = fit_feature_model(
        X_train=X_train_ml,
        y_train=y_train_ml,
    )

    feature_pred = forecast_feature_model(
        model=feature_model,
        X_test=X_test_ml,
        index=y_test_ml.index,
    )

    # Align to the main test index
    forecasts["feature_model"] = feature_pred.reindex(test.index)

    # --------------------------------------------------------
    # Foundation model placeholder
    # --------------------------------------------------------

    forecasts["foundation_model"] = forecast_foundation_model_placeholder(
        y_train=train,
        horizon=horizon,
        index=test.index,
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    for name, pred in forecasts.items():
        pred = pred.reindex(test.index)

        # Some feature-model forecasts may start slightly later if features
        # require lags. For this demo, drop missing aligned points.
        valid = pred.notna() & test.notna()

        results.append(
            evaluate_forecast(
                name=name,
                y_true=test.loc[valid],
                y_pred=pred.loc[valid],
                y_train=train,
            )
        )

    results_df = (
        pd.DataFrame(results)
        .sort_values("MASE")
        .reset_index(drop=True)
    )

    print("\nModel comparison:")
    print(results_df.round(3))

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    forecast_df = pd.DataFrame({"actual": test})

    for name, pred in forecasts.items():
        forecast_df[name] = pred.reindex(test.index)

    forecast_df.to_csv(FORECAST_DIR / "all_forecasts.csv")
    results_df.to_csv(METRICS_DIR / "model_comparison.csv", index=False)

    fig = plot_forecasts(
        train=train,
        test=test,
        forecast_df=forecast_df,
    )

    fig.savefig(
        FIGURE_DIR / "forecast_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    print("\nSaved outputs:")
    print(FORECAST_DIR / "all_forecasts.csv")
    print(METRICS_DIR / "model_comparison.csv")
    print(FIGURE_DIR / "forecast_comparison.png")

    return results_df, forecast_df


if __name__ == "__main__":
    run_pipeline()