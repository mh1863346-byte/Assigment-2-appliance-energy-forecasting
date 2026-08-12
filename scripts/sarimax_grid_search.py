"""AIC grid search required by the brief.
Loops p=0..6, d=0..2, q=0..6. Seasonal order is configurable.
The script checkpoints every fit because the full 147-model search can take hours.
"""
from itertools import product
from pathlib import Path
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
ROOT=Path(__file__).resolve().parents[1]
y=pd.read_csv(ROOT/'data/processed/appliance_hourly.csv',parse_dates=['date'],index_col='date')['Appliances']
# Use training period only; retain final 14 days for test.
y=y.iloc[:-(14*24)]
results=[]; out=ROOT/'outputs/metrics/sarimax_aic_grid.csv'
for p,d,q in product(range(7),range(3),range(7)):
    try:
        fit=SARIMAX(y,order=(p,d,q),seasonal_order=(1,0,1,24),trend='c',enforce_stationarity=False,enforce_invertibility=False).fit(disp=False,maxiter=50)
        results.append({'p':p,'d':d,'q':q,'P':1,'D':0,'Q':1,'s':24,'AIC':fit.aic,'converged':bool(fit.mle_retvals.get('converged',False))})
    except Exception as e:
        results.append({'p':p,'d':d,'q':q,'P':1,'D':0,'Q':1,'s':24,'AIC':None,'converged':False,'error':str(e)[:200]})
    pd.DataFrame(results).to_csv(out,index=False)
print(pd.DataFrame(results).sort_values('AIC').head(10))
