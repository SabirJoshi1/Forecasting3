import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

def run_full_forecast_pipeline(df):
    if df.empty or len(df) < 30:
        return None  # or raise an error depending on UX strategy

    # Preprocessing & Aggregation
    df = df.groupby('Date').agg({
        'Sales': 'sum',
        'Holiday': 'max',
        '#Order': 'sum',
        'Discount': 'sum',
        'Store_id': 'nunique'
    }).asfreq('D').ffill()
    df['Date'] = df.index

    # Focus on last 365 days
    latest_year = df[df.index >= (df.index.max() - pd.Timedelta(days=365))].copy()
    latest_year['log_sales'] = np.log1p(latest_year['Sales'])

    log_sales_series = latest_year['log_sales']
    exog = latest_year[['Holiday', '#Order', 'Discount', 'Store_id']]
    n = len(log_sales_series)

    # Data Split
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_y = log_sales_series[:train_end]
    val_y = log_sales_series[train_end:val_end]
    test_y = log_sales_series[val_end:]

    train_exog = exog[:train_end]
    val_exog = exog[train_end:val_end]
    test_exog = exog[val_end:]

    # Model
    model = ARIMA(train_y, order=(2, 1, 2), exog=train_exog)
    model_fit = model.fit()

    # Forecast on validation
    forecast_log = model_fit.forecast(steps=len(val_y), exog=val_exog)
    forecast = np.expm1(forecast_log)
    actual = np.expm1(val_y)
    val_dates = latest_year.index[train_end:val_end]

    # Metrics
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mae = mean_absolute_error(actual, forecast)
    safety_stock = forecast.std() * 1.5
    recommended_stock = forecast + safety_stock

    # Forecast Table
    forecast_df = pd.DataFrame({
        'Date': val_dates,
        'Actual Sales': actual.values,
        'Forecasted Sales': forecast.values
    })

    inventory_df = pd.DataFrame({
        'Date': val_dates,
        'Forecasted Sales': forecast.values,
        'Recommended Stock Level': recommended_stock.values,
        'Safety Stock': safety_stock
    })

    summary_df = pd.DataFrame({
        'Date': val_dates,
        'Forecasted Sales': forecast.values,
        'Actual Sales': actual.values,
        'Error': actual.values - forecast.values,
        'Recommended Stock': recommended_stock.values,
        'Safety Stock': safety_stock
    })

    return {
        'forecast_df': forecast_df,
        'inventory_df': inventory_df,
        'summary_df': summary_df,
        'forecast': forecast,
        'actual': actual,
        'test_dates': val_dates,  # using validation window
        'safety_stock': safety_stock,
        'recommended_stock': recommended_stock,
        'rmse': rmse,
        'mae': mae
    }
