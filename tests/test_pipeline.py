import numpy as np
import pandas as pd

def mase(y_true,y_pred,y_train,m=24):
    scale=np.mean(np.abs(np.asarray(y_train)[m:]-np.asarray(y_train)[:-m]))
    return np.mean(np.abs(np.asarray(y_true)-np.asarray(y_pred)))/scale

def test_mase_perfect_forecast_is_zero():
    train=np.arange(100,dtype=float); y=np.arange(10,dtype=float)
    assert mase(y,y,train,24)==0

def test_lag_uses_past_value():
    s=pd.Series([10,20,30,40])
    lag=s.shift(1)
    assert np.isnan(lag.iloc[0]) and lag.iloc[2]==20

def test_output_forecast_length():
    f=pd.read_csv('outputs/forecasts/all_forecasts.csv')
    assert len(f)==24

def test_no_missing_actuals():
    f=pd.read_csv('outputs/forecasts/all_forecasts.csv')
    assert f['actual'].isna().sum()==0
