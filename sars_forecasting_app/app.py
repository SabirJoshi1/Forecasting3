import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go
from modules.loader import load_data
from modules.forecast import run_full_forecast_pipeline

# ======================
# STREAMLIT CONFIGURATION
# ======================
def setup_page():
    st.set_page_config(page_title="SARS Forecasting Platform", layout="wide")
    
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

# ======================
# DATA LOADING & FILTERING
# ======================
def load_and_filter_data():
    st.markdown("<h1 style='text-align: center;'>📊 SARS Forecasting Platform</h1>", unsafe_allow_html=True)
    st.markdown("""
    <h2>Welcome</h2>
    <p class='section-text'>
        Upload last year's sales data to forecast future sales using ARIMAX.
        This tool helps reduce stockouts, optimize inventory, and improve performance.
    </p>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📁 Upload Last Year Sales File (CSV)", type=["csv"])
    if not uploaded_file:
        st.warning("⚠️ Please upload a CSV file to continue.")
        st.stop()

    raw_df = load_data(uploaded_file)

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

    return raw_df[
        (raw_df['Region_Code'] == selected_region) &
        (raw_df['Store_Type'] == selected_store_type) &
        (raw_df['Location_Type'] == selected_location_type)
    ]

# ======================
# VISUALIZATION COMPONENTS
# ======================
def create_forecast_chart(test_dates, actual, forecast):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_dates, y=actual, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=test_dates, y=forecast, mode='lines', name='Forecast'))
    fig.update_layout(title='Forecast vs Actual', xaxis_title='Date', yaxis_title='Sales Volume', hovermode='x unified')
    return fig

def create_inventory_chart(test_dates, forecast, recommended_stock):
    forecast_trace = go.Scatter(x=test_dates, y=forecast, mode='lines', name='Forecasted Sales', line=dict(color='blue'))
    stock_trace = go.Scatter(x=test_dates, y=recommended_stock, mode='lines', name='Recommended Stock', line=dict(dash='dash', color='orange'))
    upper = go.Scatter(x=test_dates, y=np.maximum(forecast, recommended_stock), mode='lines', line=dict(width=0), showlegend=False)
    lower = go.Scatter(x=test_dates, y=np.minimum(forecast, recommended_stock), mode='lines', fill='tonexty', name='Safety Buffer', fillcolor='rgba(255,165,0,0.3)', line=dict(width=0))
    return go.Figure(data=[forecast_trace, stock_trace, upper, lower], layout=go.Layout(title='Forecast vs Recommended Inventory Level', xaxis_title='Date', yaxis_title='Sales Volume', hovermode='x unified'))

# ======================
# MAIN DASHBOARD TABS
# ======================
def show_kpi_metrics(test_dates, forecast, rmse):
    col1, col2, col3 = st.columns(3)
    col1.metric("🗓️ Forecast Period", f"{len(test_dates)} days")
    col2.metric("📈 Avg Forecasted Sales", f"${forecast.mean():,.0f}")
    col3.metric("📉 Validation RMSE", f"${rmse:,.0f}")

def show_forecast_tab(forecast_df, inventory_df, test_dates, actual, forecast, recommended_stock, rmse):
    st.header("🎯 Forecasting Objective")
    st.markdown("This dashboard uses ARIMAX to improve inventory accuracy and reduce operational risks.")
    st.subheader("📌 Key Performance Indicators")
    show_kpi_metrics(test_dates, forecast, rmse)
    st.subheader("📉 Forecast vs Actual Sales")
    st.plotly_chart(create_forecast_chart(test_dates, actual, forecast), use_container_width=True)
    st.subheader("📋 Forecast Table")
    st.dataframe(forecast_df, use_container_width=True)
    st.subheader("📦 Inventory Plan")
    st.dataframe(inventory_df, use_container_width=True)
    st.subheader("📈 Forecast vs Recommended Inventory Level")
    st.plotly_chart(create_inventory_chart(test_dates, forecast, recommended_stock), use_container_width=True)

def show_reports_tab(forecast_df, inventory_df):
    st.header("📁 Download Reports")
    st.download_button("Download Forecast Table", forecast_df.to_csv(index=False), "forecast.csv")
    st.download_button("Download Inventory Plan", inventory_df.to_csv(index=False), "inventory.csv")

def show_scenario_tab(summary_df, mae):
    st.header("⚖️ Scenario Comparison")
    st.dataframe(summary_df, use_container_width=True)
    st.subheader("🔍 Insights")
    total_error = summary_df['Error'].abs().sum()
    avg_stock_buffer = summary_df['Safety Stock'].mean()
    days_understock = (summary_df['Error'] > avg_stock_buffer).sum()
    st.markdown(f"""
    - **Total Absolute Forecast Error:** {total_error:,.0f} units  
    - **Average Safety Stock:** {avg_stock_buffer:,.0f} units  
    - **Days Understocked:** {days_understock} days  
    - **Validation MAE:** {mae:,.0f} units  
    """)

def show_comparison_tab(rmse, baseline_rmse_naive, baseline_rmse_mean):
    st.header("📏 Model Comparison")
    st.markdown("Compare RMSE values to benchmark ARIMAX against simpler methods:")
    comparison_df = pd.DataFrame({
        'Model': ['ARIMAX', 'Naive', 'Mean'],
        'RMSE': [rmse, baseline_rmse_naive, baseline_rmse_mean]
    })
    st.dataframe(comparison_df)
    fig = go.Figure(go.Bar(x=comparison_df['Model'], y=comparison_df['RMSE'], marker_color=['deepskyblue', 'lightgray', 'gray']))
    fig.update_layout(title="RMSE Comparison", xaxis_title="Model", yaxis_title="RMSE")
    st.plotly_chart(fig, use_container_width=True)

# ======================
# MAIN APPLICATION FLOW
# ======================
def main():
    setup_page()
    filtered_df = load_and_filter_data()
    try:
        results = run_full_forecast_pipeline(filtered_df)
        if results is None:
            return

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📁 Download Reports", "⚖️ Scenario Comparison", "📏 Model Comparison"])

        with tab1:
            show_forecast_tab(results['forecast_df'], results['inventory_df'], results['test_dates'], results['actual'], results['forecast'], results['recommended_stock'], results['rmse'])
        with tab2:
            show_reports_tab(results['forecast_df'], results['inventory_df'])
        with tab3:
            show_scenario_tab(results['summary_df'], results['mae'])
        with tab4:
            show_comparison_tab(results['rmse'], results['baseline_rmse_naive'], results['baseline_rmse_mean'])

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
