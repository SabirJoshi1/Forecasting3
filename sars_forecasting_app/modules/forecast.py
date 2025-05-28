import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

def run_baseline_naive(series):
    forecast = series.shift(1).dropna()
    actual = series[1:]
    return actual, forecast

def run_baseline_mean(series):
    forecast = pd.Series(series.mean(), index=series.index)
    return series, forecast

def cross_validate_model(series, exog, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_scores = []

    for train_index, test_index in tscv.split(series):
        train_y, test_y = series.iloc[train_index], series.iloc[test_index]
        train_exog, test_exog = exog.iloc[train_index], exog.iloc[test_index]

        try:
            model = ARIMA(train_y, order=(2, 1, 2), exog=train_exog)
            model_fit = model.fit()
            forecast_log = model_fit.forecast(steps=len(test_y), exog=test_exog)
            forecast = np.expm1(forecast_log)
            actual = np.expm1(test_y)
            rmse = np.sqrt(mean_squared_error(actual, forecast))
            rmse_scores.append(rmse)
        except Exception:
            rmse_scores.append(np.nan)
            continue

    return rmse_scores

def run_full_forecast_pipeline(df):
    if df.empty or len(df) < 30:
        return None

    # Preprocessing & Aggregation
    df = df.groupby('Date').agg({
        'Sales': 'sum',
        'Holiday': 'max',
        '#Order': 'sum',
        'Discount': 'sum',
        'Store_id': 'nunique'
    }).asfreq('D').ffill()
    df['Date'] = df.index

    latest_year = df[df.index >= (df.index.max() - pd.Timedelta(days=365))].copy()
    latest_year['log_sales'] = np.log1p(latest_year['Sales'])

    log_sales_series = latest_year['log_sales']
    exog = latest_year[['Holiday', '#Order', 'Discount', 'Store_id']]
    n = len(log_sales_series)

    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_y = log_sales_series[:train_end]
    val_y = log_sales_series[train_end:val_end]
    test_y = log_sales_series[val_end:]

    train_exog = exog[:train_end]
    val_exog = exog[train_end:val_end]
    test_exog = exog[val_end:]

    model = ARIMA(train_y, order=(2, 1, 2), exog=train_exog)
    model_fit = model.fit()

    forecast_log = model_fit.forecast(steps=len(val_y), exog=val_exog)
    forecast = np.expm1(forecast_log)
    actual = np.expm1(val_y)
    val_dates = latest_year.index[train_end:val_end]

    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mae = mean_absolute_error(actual, forecast)
    safety_stock = forecast.std() * 1.5
    recommended_stock = forecast + safety_stock

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

    # Baseline comparisons
    baseline_actual_naive, baseline_forecast_naive = run_baseline_naive(actual)
    baseline_rmse_naive = np.sqrt(mean_squared_error(baseline_actual_naive, baseline_forecast_naive))

    baseline_actual_mean, baseline_forecast_mean = run_baseline_mean(actual)
    baseline_rmse_mean = np.sqrt(mean_squared_error(baseline_actual_mean, baseline_forecast_mean))

    # Cross-validation RMSE
    cv_rmse_scores = cross_validate_model(log_sales_series, exog)

    return {
        'forecast_df': forecast_df,
        'inventory_df': inventory_df,
        'summary_df': summary_df,
        'forecast': forecast,
        'actual': actual,
        'test_dates': val_dates,
        'safety_stock': safety_stock,
        'recommended_stock': recommended_stock,
        'rmse': rmse,
        'mae': mae,
        'baseline_rmse_naive': baseline_rmse_naive,
        'baseline_rmse_mean': baseline_rmse_mean,
        'cv_rmse_scores': cv_rmse_scores
    }
