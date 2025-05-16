import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

def preprocess_data(df):
    df = df.rename(columns={"date": "ds", "consumption": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    return df

def forecast_prophet(df, periods=30):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']]

def forecast_arima(df, periods=30):
    model = ARIMA(df['y'], order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    future_dates = pd.date_range(start=df['ds'].iloc[-1] + pd.Timedelta(days=1), periods=periods)
    return pd.DataFrame({"ds": future_dates, "yhat": forecast})
