import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from modules.loader import load_data
from modules.forecast import apply_arimax

st.set_page_config(page_title="SARS Forecasting Platform", layout="wide")

# --- Theme Toggle ---
theme_mode = st.sidebar.radio("🎨 Theme Mode", ["Dark", "Light"], index=0)

if theme_mode == "Dark":
    st.markdown("""
        <style>
            body {
                background-image: url('1B.jpg');
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
                background-position: center;
            }
            html, body, [class*="css"] {
                background-color: rgba(0, 0, 0, 0.85);
                color: #f0f0f0;
                font-family: "Segoe UI", sans-serif;
            }
            .stButton>button {
                background-color: #00c3ff;
                color: white;
                font-weight: bold;
            }
            h1, h2 {
                color: #00c3ff;
                margin-top: 2rem;
            }
            h3, h4 {
                color: #66e0ff;
            }
            .section-text {
                font-size: 16px;
                padding-bottom: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            html, body, [class*="css"] {
                background-color: #ffffff;
                color: #111;
                font-family: "Segoe UI", sans-serif;
            }
            .stButton>button {
                background-color: #0057a8;
                color: white;
                font-weight: bold;
            }
            h1, h2 {
                color: #0057a8;
                margin-top: 2rem;
            }
            h3, h4 {
                color: #228be6;
            }
            .section-text {
                font-size: 16px;
                padding-bottom: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)

# --- Heading ---
st.markdown("<h1 style='text-align: center;'>📊 SARS Forecasting Platform</h1>", unsafe_allow_html=True)

st.markdown("""
<h2>Welcome</h2>
<p class='section-text'>
    Upload last year's sales data to forecast future sales using ARIMAX.
    This tool helps reduce stockouts, optimize inventory, and improve performance.
</p>
""", unsafe_allow_html=True)

# --- File Upload ---
uploaded_file = st.file_uploader("📁 Upload Last Year Sales File (CSV)", type=["csv"])
if not uploaded_file:
    st.warning("⚠️ Please upload a CSV file to continue.")
    st.stop()

raw_df = load_data(uploaded_file)

# --- Filters ---
if 'filters_applied' not in st.session_state:
    st.session_state.filters_applied = False

with st.sidebar:
    st.header("🔎 Filter Parameters")
    selected_region = st.selectbox("🌍 Select Region", sorted(raw_df['Region_Code'].unique()))
    selected_store_type = st.selectbox("🏪 Select Store Type", sorted(raw_df['Store_Type'].unique()))
    selected_location_type = st.selectbox("📍 Select Location Type", sorted(raw_df['Location_Type'].unique()))
    if st.button("✅ Apply Filters"):
        st.session_state.filters_applied = True

if not st.session_state.filters_applied:
    st.stop()

filtered_df = raw_df[(raw_df['Region_Code'] == selected_region) &
                     (raw_df['Store_Type'] == selected_store_type) &
                     (raw_df['Location_Type'] == selected_location_type)]

forecast_df, inventory_df, summary_df, forecast, actual, val_dates, safety_stock, recommended_stock, rmse = apply_arimax(filtered_df)

# --- Dashboard Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📁 Download Reports", "⚖️ Scenario Comparison"])

with tab1:
    st.header("🎯 Forecasting Objective")
    st.markdown("This dashboard uses ARIMAX to improve inventory accuracy and reduce operational risks.")

    st.subheader("📌 Key Performance Indicators")
    kpi_df = pd.DataFrame({
        'KPI': ['Total Forecast Period', 'Avg Forecasted Sales', 'Validation RMSE'],
        'Value': [f"{len(val_dates)} days", f"${forecast.mean():,.0f}", f"${rmse:,.0f}"]
    })
    st.dataframe(kpi_df, use_container_width=True)

    st.subheader("📉 Forecast vs Actual Sales")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=val_dates, y=actual, mode='lines', name='Actual', hovertemplate='Date: %{x}<br>Sales: %{y:.0f}'))
    fig.add_trace(go.Scatter(x=val_dates, y=forecast, mode='lines', name='Forecast', hovertemplate='Date: %{x}<br>Sales: %{y:.0f}'))
    fig.update_layout(title='Forecast vs Actual', xaxis_title='Date', yaxis_title='Sales Volume', hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Forecast Table")
    st.dataframe(forecast_df.style.format({"Actual Sales": "{:.0f}", "Forecasted Sales": "{:.0f}"}), use_container_width=True)

    st.subheader("📦 Inventory Plan")
    st.dataframe(inventory_df.style.format({
        "Forecasted Sales": "{:.0f}",
        "Recommended Stock Level": "{:.0f}",
        "Safety Stock": "{:.0f}"
    }), use_container_width=True)

with tab2:
    st.header("📁 Download Reports")
    st.download_button("Download Forecast Table", forecast_df.to_csv(index=False), "forecast.csv")
    st.download_button("Download Inventory Plan", inventory_df.to_csv(index=False), "inventory.csv")

with tab3:
    st.header("⚖️ Scenario Comparison")
    st.dataframe(summary_df.style.format({
        "Forecasted Sales": "{:.0f}",
        "Actual Sales": "{:.0f}",
        "Error": "{:+.0f}",
        "Recommended Stock": "{:.0f}",
        "Safety Stock": "{:.0f}"
    }), use_container_width=True)

    st.subheader("🔍 Insights")
    total_error = summary_df['Error'].abs().sum()
    avg_stock_buffer = summary_df['Safety Stock'].mean()
    days_understock = (summary_df['Error'] > avg_stock_buffer).sum()
    st.markdown(f"""
    - **Total Absolute Forecast Error:** {total_error:,.0f} units
    - **Average Safety Stock:** {avg_stock_buffer:,.0f} units
    - **Days Understocked:** {days_understock} days
    """)
