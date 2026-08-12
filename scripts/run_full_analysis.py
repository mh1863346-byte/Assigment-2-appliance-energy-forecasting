from pathlib import Path
import json, warnings, math
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'energydata_complete.csv'
OUT = ROOT/'outputs'; FIG=OUT/'figures'; MET=OUT/'metrics'; FC=OUT/'forecasts'; PROC=ROOT/'data'/'processed'
for p in [FIG,MET,FC,PROC]: p.mkdir(parents=True, exist_ok=True)
TARGET='Appliances'; H=14*24; np.random.seed(42)

def rmse(a,b): return float(np.sqrt(mean_squared_error(a,b)))
def mase(a,b,train,m=24):
    scale=np.mean(np.abs(np.asarray(train)[m:]-np.asarray(train)[:-m])); return float(np.mean(np.abs(np.asarray(a)-np.asarray(b)))/scale)
def metrics(name,a,b,train):
    return {'model':name,'MAE':float(mean_absolute_error(a,b)),'RMSE':rmse(a,b),'MASE':mase(a,b,train),'Bias':float(np.mean(np.asarray(b)-np.asarray(a)))}

def load():
    df=pd.read_csv(DATA,parse_dates=['date']).set_index('date').sort_index()
    df=df.apply(pd.to_numeric,errors='coerce')
    hourly=df.resample('h').mean().interpolate('time').dropna()
    hourly.to_csv(PROC/'appliance_hourly.csv')
    return hourly

def seasonal_naive(train,index,m):
    hist=list(train.values); vals=[]
    for _ in index:
        v=hist[-m]; vals.append(v); hist.append(v)
    return pd.Series(vals,index=index)

def add_time(df):
    o=df.copy(); o['hour']=o.index.hour; o['dow']=o.index.dayofweek; o['weekend']=(o['dow']>=5).astype(int)
    o['hour_sin']=np.sin(2*np.pi*o['hour']/24); o['hour_cos']=np.cos(2*np.pi*o['hour']/24)
    o['dow_sin']=np.sin(2*np.pi*o['dow']/7); o['dow_cos']=np.cos(2*np.pi*o['dow']/7)
    return o

def make_features(df, include_exog=True):
    base=add_time(df)
    keep=['hour','dow','weekend','hour_sin','hour_cos','dow_sin','dow_cos']
    if include_exog:
        ex=[c for c in df.columns if c!=TARGET and (c.startswith('T') or c.startswith('RH') or c in ['lights','Press_mm_hg','Windspeed','Visibility','Tdewpoint'])]
        keep += ex
    out=base[keep].copy()
    for lag in [1,2,3,6,12,24,48,168]: out[f'lag_{lag}']=df[TARGET].shift(lag)
    for w in [3,6,12,24,168]:
        out[f'roll_mean_{w}']=df[TARGET].shift(1).rolling(w).mean(); out[f'roll_std_{w}']=df[TARGET].shift(1).rolling(w).std()
    out[TARGET]=df[TARGET]
    return out.dropna()

def recursive_xgb(model, full_df, train_end, test_index, feature_columns, include_exog=True):
    history=full_df.loc[:train_end,TARGET].copy(); preds=[]
    for ts in test_index:
        row={}
        hour=ts.hour; dow=ts.dayofweek
        row.update(hour=hour,dow=dow,weekend=int(dow>=5),hour_sin=np.sin(2*np.pi*hour/24),hour_cos=np.cos(2*np.pi*hour/24),dow_sin=np.sin(2*np.pi*dow/7),dow_cos=np.cos(2*np.pi*dow/7))
        if include_exog:
            for c in feature_columns:
                if c in full_df.columns: row[c]=full_df.loc[ts,c]
        for lag in [1,2,3,6,12,24,48,168]: row[f'lag_{lag}']=history.iloc[-lag]
        for w in [3,6,12,24,168]:
            vals=history.iloc[-w:]; row[f'roll_mean_{w}']=vals.mean(); row[f'roll_std_{w}']=vals.std()
        X=pd.DataFrame([row],index=[ts]).reindex(columns=feature_columns)
        pred=max(0.0,float(model.predict(X)[0])); preds.append(pred); history.loc[ts]=pred
    return pd.Series(preds,index=test_index)

def main():
    df=load(); y=df[TARGET]; train=y.iloc[:-H]; test=y.iloc[-H:]; idx=test.index
    # EDA
    desc=y.describe().to_dict(); adf=adfuller(train,autolag='AIC')
    summary={'rows_original_10min':19735,'rows_hourly':len(df),'start':str(df.index.min()),'end':str(df.index.max()),'missing_after_processing':int(df.isna().sum().sum()),'target_mean':float(y.mean()),'target_std':float(y.std()),'target_min':float(y.min()),'target_max':float(y.max()),'adf_statistic':float(adf[0]),'adf_pvalue':float(adf[1])}
    (OUT/'analysis_summary.json').write_text(json.dumps(summary,indent=2))
    plt.figure(figsize=(12,4)); y.plot(linewidth=.6); plt.title('Hourly appliance energy use'); plt.ylabel('Wh (hourly mean)'); plt.tight_layout(); plt.savefig(FIG/'full_series.png',dpi=200); plt.close()
    hourly_profile=df.groupby(df.index.hour)[TARGET].mean(); plt.figure(figsize=(8,4)); hourly_profile.plot(marker='o'); plt.title('Average appliance use by hour of day'); plt.xlabel('Hour'); plt.ylabel('Mean appliance use'); plt.tight_layout(); plt.savefig(FIG/'hourly_profile.png',dpi=200); plt.close()
    fig,ax=plt.subplots(figsize=(8,4)); plot_acf(train,lags=72,ax=ax); ax.set_title('ACF of hourly appliance use'); fig.tight_layout(); fig.savefig(FIG/'target_acf.png',dpi=200); plt.close(fig)
    forecasts={}
    forecasts['mean']=pd.Series(train.mean(),index=idx)
    forecasts['naive']=pd.Series(train.iloc[-1],index=idx)
    forecasts['daily_seasonal_naive']=seasonal_naive(train,idx,24)
    forecasts['weekly_seasonal_naive']=seasonal_naive(train,idx,168)
    slope=(train.iloc[-1]-train.iloc[0])/(len(train)-1); forecasts['drift']=pd.Series([train.iloc[-1]+slope*(i+1) for i in range(H)],index=idx)
    # SARIMAX selected practical order; exogenous weather values make this conditional
    exog_cols=['T_out','RH_out','Windspeed','Visibility','Tdewpoint']
    sar_train=train.tail(60*24); sar_exog=df[exog_cols].loc[sar_train.index]; fit=SARIMAX(sar_train,exog=sar_exog,order=(1,0,1),seasonal_order=(1,0,1,24),trend='c',enforce_stationarity=False,enforce_invertibility=False).fit(disp=False,maxiter=50)
    sfc=fit.get_forecast(H,exog=df[exog_cols].iloc[-H:]); forecasts['sarimax_conditional']=pd.Series(np.maximum(0,sfc.predicted_mean.values),index=idx)
    ci=sfc.conf_int(); pd.DataFrame({'lower':ci.iloc[:,0].values,'upper':ci.iloc[:,1].values},index=idx).to_csv(FC/'sarimax_confidence_intervals.csv')
    resid=fit.resid.dropna(); fig,ax=plt.subplots(figsize=(8,4)); plot_acf(resid,lags=72,ax=ax); ax.set_title('SARIMAX residual ACF'); fig.tight_layout(); fig.savefig(FIG/'sarimax_residual_acf.png',dpi=200); plt.close(fig)
    plt.figure(figsize=(8,4)); plt.hist(resid,bins=50); plt.title('SARIMAX residual distribution'); plt.xlabel('Residual'); plt.tight_layout(); plt.savefig(FIG/'sarimax_residual_hist.png',dpi=200); plt.close()
    # ML operational: time + past target only
    for label,inc in [('xgboost_operational',False),('xgboost_conditional',True)]:
        tab=make_features(df.iloc[:-H],include_exog=inc); X=tab.drop(columns=[TARGET]); yy=tab[TARGET]
        model=XGBRegressor(n_estimators=500,max_depth=5,learning_rate=.03,subsample=.85,colsample_bytree=.85,objective='reg:squarederror',random_state=42,n_jobs=4)
        model.fit(X,yy)
        forecasts[label]=recursive_xgb(model,df,train.index[-1],idx,list(X.columns),inc)
        imp=pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=False).head(20)
        plt.figure(figsize=(8,6)); imp.sort_values().plot(kind='barh'); plt.title(f'Top feature importance: {label}'); plt.tight_layout(); plt.savefig(FIG/f'{label}_feature_importance.png',dpi=200); plt.close()
    # Foundation model unavailable offline: transparent benchmark-equivalent placeholder
    forecasts['foundation_placeholder']=forecasts['daily_seasonal_naive'].copy()
    rows=[metrics(k,test,v,train) for k,v in forecasts.items()]
    mdf=pd.DataFrame(rows).sort_values('RMSE').reset_index(drop=True); mdf.to_csv(MET/'model_comparison.csv',index=False)
    fdf=pd.DataFrame({'actual':test,**forecasts}); fdf.to_csv(FC/'all_forecasts.csv')
    plt.figure(figsize=(14,7)); train.tail(168).plot(label='training'); test.plot(label='actual',linewidth=2)
    for c in ['daily_seasonal_naive','weekly_seasonal_naive','sarimax_conditional','xgboost_operational','xgboost_conditional']: forecasts[c].plot(label=c,alpha=.8)
    plt.title('Forecast comparison over final 14 days'); plt.ylabel('Appliance energy use'); plt.legend(ncol=2); plt.tight_layout(); plt.savefig(FIG/'forecast_comparison.png',dpi=200); plt.close()
    # zoom final 24h
    plt.figure(figsize=(12,5)); test.tail(24).plot(label='actual',linewidth=2)
    for c in ['daily_seasonal_naive','weekly_seasonal_naive','sarimax_conditional','xgboost_operational','xgboost_conditional']: forecasts[c].tail(24).plot(label=c)
    plt.title('Final 24-hour forecast comparison'); plt.legend(ncol=2); plt.tight_layout(); plt.savefig(FIG/'forecast_final_24h.png',dpi=200); plt.close()
    print(mdf.to_string(index=False)); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
