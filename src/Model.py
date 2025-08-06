import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def train_arima_model(series: pd.Series, order=(2,0,2)):
    """
    Fit an ARIMA model to a time series.

    Args:
        series (pd.Series): Time series data.
        order (tuple): ARIMA(p,d,q) order.

    Returns:
        model_fit: Fitted ARIMA model.
    """
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit

def forecast_arima(model_fit, steps=30):
    """
    Forecast future values using a trained ARIMA model.

    Args:
        model_fit: Trained ARIMA model.
        steps (int): Number of steps to forecast.

    Returns:
        pd.Series: Forecasted values.
    """
    forecast = model_fit.forecast(steps=steps)
    return forecast

