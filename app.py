# Standard library imports
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import requests
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
import folium
from streamlit_folium import st_folium

# Create a Streamlit page configuration titled "Air Quality Predictor"
st.set_page_config(page_title="AQI Prediction & Health Alert System", layout="wide")

# Hide streamlit menu and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.header("📍 Location & Environmental Data")

# Initialize session state for coordinates if not already set
if 'selected_lat' not in st.session_state:
    st.session_state.selected_lat = 28.7041
if 'selected_lon' not in st.session_state:
    st.session_state.selected_lon = 77.1025

latitude = st.sidebar.number_input("Latitude", value=st.session_state.selected_lat, format="%.4f")
longitude = st.sidebar.number_input("Longitude", value=st.session_state.selected_lon, format="%.4f")

# Update session state when sidebar values change
st.session_state.selected_lat = latitude
st.session_state.selected_lon = longitude

wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 20.0, 5.0)

st.sidebar.header("🌬️ Pollutant Levels")
pm25 = st.sidebar.slider("PM2.5 (µg/m³)", 0.0, 500.0, 25.0)
pm10 = st.sidebar.slider("PM10 (µg/m³)", 0.0, 500.0, 50.0)
no = st.sidebar.slider("NO (ppb)", 0.0, 200.0, 20.0)
no2 = st.sidebar.slider("NO2 (ppb)", 0.0, 200.0, 40.0)
nox = st.sidebar.slider("NOx (ppb)", 0.0, 400.0, 60.0)
nh3 = st.sidebar.slider("NH3 (ppb)", 0.0, 100.0, 10.0)
co = st.sidebar.slider("CO (ppm)", 0.0, 50.0, 1.0)
so2 = st.sidebar.slider("SO2 (ppb)", 0.0, 200.0, 20.0)
o3 = st.sidebar.slider("O3 (ppb)", 0.0, 200.0, 30.0)

st.sidebar.header("📅 Temporal & Historical Data")
date = st.sidebar.date_input("Date", datetime.now())
aqi_lag1 = st.sidebar.slider("Previous Day AQI (lag)", 0.0, 500.0, 50.0)
# Load rf_model.pkl, xgb_model.pkl, lstm_model.h5, and scaler.pkl
@st.cache_resource
def load_models():
    rf_model = joblib.load("rf_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    lstm_model = load_model("lstm_model.h5", compile=False)
    scaler = joblib.load("scaler.pkl")
    return rf_model, xgb_model, lstm_model, scaler

rf_model, xgb_model, lstm_model, scaler = load_models()

# Model Performance Metrics (from training/evaluation)
MODEL_METRICS = {
    "Random Forest": {
        "MAE": 1.109,
        "RMSE": 5.623,
        "R²": 0.997,
        "Status": "🏆 Best Performer"
    },
    "XGBoost": {
        "MAE": 5.733,
        "RMSE": 13.045,
        "R²": 0.987,
        "Status": "⭐ Good"
    },
    "LSTM": {
        "MAE": 57.443,
        "RMSE": 77.907,
        "R²": 0.528,
        "Status": "⚠️ Needs Improvement"
    }
}
# Create a function that scales the input features and returns predictions for all models
def predict_air_quality(pm25, pm10, no, no2, nox, nh3, co, so2, o3, date, aqi_lag1):
    # Create a DataFrame from the input features with all required columns in correct order
    input_data = pd.DataFrame(
        [[pm25, pm10, no, no2, nox, nh3, co, so2, o3, date.year, date.month, date.day, aqi_lag1]],
        columns=["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "year", "month", "day", "AQI_lag1"],
    )
    # Scale the input features using the loaded scaler
    scaled_data = scaler.transform(input_data)
    
    # Get predictions from both ML models
    rf_prediction = rf_model.predict(scaled_data)[0]
    xgb_prediction = xgb_model.predict(scaled_data)[0]
    
    # Get prediction from LSTM (needs to be reshaped to (1, 1, 13))
    scaled_data_lstm = scaled_data.reshape((1, 1, 13))
    lstm_prediction = lstm_model.predict(scaled_data_lstm, verbose=0)[0][0]
    
    return rf_prediction, xgb_prediction, lstm_prediction    
# Create a function to map AQI values to CPCB categories (Good, Moderate, etc.)
def aqi_to_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 200:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 300:
        return "Unhealthy"
    elif aqi <= 400:
        return "Very Unhealthy"
    else:
        return "Hazardous"

# Health risk assessment function
def assess_health_risk(aqi):
    """Assess health risks for different groups"""
    risks = {
        "General Population": "❌ Not Recommended" if aqi > 200 else "⚠️ Caution" if aqi > 100 else "✅ Safe",
        "Kids & Infants": "❌ High Risk" if aqi > 150 else "⚠️ Moderate Risk" if aqi > 75 else "✅ Safe",
        "Elderly & Sick": "❌ High Risk" if aqi > 100 else "⚠️ Moderate Risk" if aqi > 50 else "✅ Safe",
        "Athletes & Outdoor Workers": "❌ Not Recommended" if aqi > 150 else "⚠️ Reduce Activity" if aqi > 75 else "✅ Safe"
    }
    return risks

# Running suitability assessment
def assess_running_suitability(aqi, wind_speed):
    """Determine if it's suitable for outdoor running"""
    if aqi <= 50 and wind_speed > 0.5:
        return "✅ Excellent", "Perfect conditions for running!", "green"
    elif aqi <= 100 and wind_speed > 0.3:
        return "⚠️ Good", "Running is acceptable but monitor yourself", "yellow"
    elif aqi <= 150 and wind_speed > 0.2:
        return "⚠️ Fair", "Running is possible but may experience symptoms", "orange"
    else:
        return "❌ Poor", "Running NOT recommended - Air quality is unhealthy", "red"

# Get color for AQI
def get_aqi_color(aqi):
    if aqi <= 50:
        return "green"
    elif aqi <= 100:
        return "yellow"
    elif aqi <= 200:
        return "orange"
    elif aqi <= 300:
        return "red"
    else:
        return "darkred"
    # Display the prediction in a big metric card and show a Plotly gauge chart for the AQI
rf_aqi, xgb_aqi, lstm_aqi = predict_air_quality(pm25, pm10, no, no2, nox, nh3, co, so2, o3, date, aqi_lag1)
rf_category = aqi_to_category(rf_aqi)
xgb_category = aqi_to_category(xgb_aqi)
lstm_category = aqi_to_category(lstm_aqi)

# Main title
st.title("🌍 Air Quality Index (AQI) Prediction & Health Alert System")
st.markdown("---")

# Introduction section
st.markdown("""
### 📋 Application Objective
This AI-powered system predicts the **Air Quality Index (AQI)** using machine learning models trained on 
the **CPCB-based "Air Quality in India (2015–2024)" dataset**. It identifies health risks and provides 
actionable insights for different population groups.

### 🌡️ Pollutants Monitored
The system tracks **13 key air quality indicators**:
- **Particulates**: PM2.5, PM10
- **Gaseous Pollutants**: NO, NO₂, NOx, NH3, CO, SO2, O3
- **Temporal Features**: Year, Month, Day, Historical AQI (lag)
""")

st.markdown("---")

# Create tabs for different views
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔮 Live Prediction", 
    "📊 Model Performance", 
    "🗺️ Location Map", 
    "❤️ Health Impact", 
    "🏃 Running Suitability", 
    "📚 Learn More"
])

# Tab 1: Live Prediction
with tab1:
    st.header("🔮 Live AQI Prediction")
    st.write("Enter or adjust pollutant levels in the sidebar to get real-time AQI predictions from all three models.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🌲 Random Forest AQI",
            value=f"{rf_aqi:.2f}",
            delta=rf_category,
            delta_color="inverse"
        )
        st.info(f"Category: **{rf_category}**")
    
    with col2:
        st.metric(
            label="🚀 XGBoost AQI",
            value=f"{xgb_aqi:.2f}",
            delta=xgb_category,
            delta_color="inverse"
        )
        st.info(f"Category: **{xgb_category}**")
    
    with col3:
        st.metric(
            label="🧠 LSTM AQI",
            value=f"{lstm_aqi:.2f}",
            delta=lstm_category,
            delta_color="inverse"
        )
        st.info(f"Category: **{lstm_category}**")
    
    st.markdown("---")
    
    # Prediction comparison chart
    fig_comparison = go.Figure()
    models = ["Random Forest", "XGBoost", "LSTM"]
    aqi_values = [rf_aqi, xgb_aqi, lstm_aqi]
    colors_list = [get_aqi_color(val) for val in aqi_values]
    
    fig_comparison.add_trace(go.Bar(
        x=models,
        y=aqi_values,
        marker=dict(color=colors_list),
        text=[f"{val:.1f}" for val in aqi_values],
        textposition="outside"
    ))
    
    fig_comparison.update_layout(
        title="AQI Predictions Comparison",
        yaxis_title="AQI Value",
        xaxis_title="Model",
        height=400,
        hovermode="x unified"
    )
    st.plotly_chart(fig_comparison, use_container_width=True)

# Tab 2: Model Performance
with tab2:
    st.header("📊 Model Performance & Comparison")
    
    st.markdown("""
    This section shows the evaluation metrics of the three machine learning models trained on the 
    CPCB Air Quality dataset (2015-2024).
    """)
    
    # Model Performance Table
    st.subheader("Model Evaluation Metrics")
    
    metrics_data = {
        "Model": [],
        "MAE": [],
        "RMSE": [],
        "R² Score": [],
        "Status": []
    }
    
    for model_name, metrics in MODEL_METRICS.items():
        metrics_data["Model"].append(model_name)
        metrics_data["MAE"].append(f"{metrics['MAE']:.3f}")
        metrics_data["RMSE"].append(f"{metrics['RMSE']:.3f}")
        metrics_data["R² Score"].append(f"{metrics['R²']:.3f}")
        metrics_data["Status"].append(metrics["Status"])
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Key Takeaway Box
    st.success("""
    ✅ **Key Takeaway**: The **Random Forest model** demonstrates superior performance with:
    - **Lowest MAE (1.109)**: On average, predictions deviate by only ~1.1 AQI points
    - **Lowest RMSE (5.623)**: Minimal penalty for larger errors
    - **Highest R² (0.997)**: Explains 99.7% of variance in AQI data
    
    **Recommendation**: Use Random Forest for production predictions due to superior accuracy and robustness.
    """)
    
    st.markdown("---")
    
    # Model Comparison Visualization
    st.subheader("📈 Metrics Comparison Chart")
    
    comparison_metrics = {
        "Model": ["Random Forest", "XGBoost", "LSTM"],
        "MAE": [1.109, 5.733, 57.443],
        "RMSE": [5.623, 13.045, 77.907],
        "R² Score": [0.997, 0.987, 0.528]
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_mae = go.Figure(data=[
            go.Bar(x=comparison_metrics["Model"], y=comparison_metrics["MAE"], 
                   marker_color=['green', 'orange', 'red'])
        ])
        fig_mae.update_layout(title="MAE Comparison (Lower is Better)", height=350, 
                             yaxis_title="MAE", xaxis_title="Model")
        st.plotly_chart(fig_mae, use_container_width=True)
    
    with col2:
        fig_rmse = go.Figure(data=[
            go.Bar(x=comparison_metrics["Model"], y=comparison_metrics["RMSE"], 
                   marker_color=['green', 'orange', 'red'])
        ])
        fig_rmse.update_layout(title="RMSE Comparison (Lower is Better)", height=350, 
                              yaxis_title="RMSE", xaxis_title="Model")
        st.plotly_chart(fig_rmse, use_container_width=True)
    
    with col3:
        fig_r2 = go.Figure(data=[
            go.Bar(x=comparison_metrics["Model"], y=comparison_metrics["R² Score"], 
                   marker_color=['green', 'orange', 'red'])
        ])
        fig_r2.update_layout(title="R² Score Comparison (Higher is Better)", height=350, 
                            yaxis_title="R²", xaxis_title="Model")
        st.plotly_chart(fig_r2, use_container_width=True)
    
    st.markdown("---")
    
    # Model Strengths & Weaknesses
    st.subheader("💡 Model Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Random Forest** 🌲
        
        ✅ Strengths:
        - Highest accuracy (R²=0.997)
        - Robust to outliers
        - Fast inference
        - Feature importance
        
        ⚠️ Weaknesses:
        - Limited temporal learning
        - "Black box" model
        """)
    
    with col2:
        st.info("""
        **XGBoost** 🚀
        
        ✅ Strengths:
        - High accuracy (R²=0.987)
        - Gradient boosting
        - Feature interactions
        - Interpretable
        
        ⚠️ Weaknesses:
        - More complex than RF
        - Slower inference
        - Prone to overfitting
        """)
    
    with col3:
        st.warning("""
        **LSTM** 🧠
        
        ✅ Strengths:
        - Captures temporal patterns
        - Sequential learning
        - Good for forecasting
        
        ⚠️ Weaknesses:
        - Lower accuracy (R²=0.528)
        - High computational cost
        - Needs more data
        - Sensitive to hyperparameter tuning
        """)
    
    st.markdown("---")
    
    st.subheader("🔍 Performance Insights")
    st.write("""
    1. **Why Random Forest Outperforms**: The dataset's high dimensionality (13 features) with non-linear 
       relationships is better captured by ensemble methods than neural networks with limited training data.
    
    2. **LSTM Underperformance**: Despite being designed for temporal data, LSTM requires significantly 
       more training data and careful hyperparameter tuning. The dataset size and temporal dependencies 
       in AQI data may not justify its added complexity.
    
    3. **Trade-offs**: While LSTM might improve with better data and tuning, the current Random Forest 
       model offers a better balance between accuracy, interpretability, and computational efficiency.
    """)

# Tab 3: Location Map
with tab3:
    st.header("📍 Location-based Air Quality Map")
    st.write("Click on the map to set your location coordinates")
    
    # Initialize session state for coordinates
    if 'selected_lat' not in st.session_state:
        st.session_state.selected_lat = latitude
    if 'selected_lon' not in st.session_state:
        st.session_state.selected_lon = longitude
    
    # Create interactive folium map
    m = folium.Map(
        location=[st.session_state.selected_lat, st.session_state.selected_lon],
        zoom_start=10,
        tiles="OpenStreetMap"
    )
    
    # Add a marker at the current location
    folium.Marker(
        location=[st.session_state.selected_lat, st.session_state.selected_lon],
        popup=f"Latitude: {st.session_state.selected_lat:.4f}<br>Longitude: {st.session_state.selected_lon:.4f}",
        icon=folium.Icon(color="blue")
    ).add_to(m)
    
    # Enable clicking on the map
    m.add_child(folium.LatLngPopup())
    
    # Display the map and capture clicks
    map_data = st_folium(m, width=700, height=500)
    
    # Update coordinates if a location was clicked
    if map_data and map_data['last_clicked']:
        st.session_state.selected_lat = map_data['last_clicked']['lat']
        st.session_state.selected_lon = map_data['last_clicked']['lng']
        latitude = st.session_state.selected_lat
        longitude = st.session_state.selected_lon
        st.rerun()
    
    st.markdown("---")
    
    # Display location info
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📍 Latitude: {st.session_state.selected_lat:.4f}")
    with col2:
        st.info(f"📍 Longitude: {st.session_state.selected_lon:.4f}")
    
    st.markdown("---")
    st.success("✅ Click on the map to select a new location. The coordinates will update automatically.")

# Tab 4: Health Impact
with tab4:
    st.header("❤️ Health Impact Assessment")
    
    avg_aqi = (rf_aqi + xgb_aqi + lstm_aqi) / 3
    health_risks = assess_health_risk(avg_aqi)
    
    st.warning(f"⚠️ **Average AQI: {avg_aqi:.2f} ({aqi_to_category(avg_aqi)})**")
    st.markdown("---")
    
    # Health risk by group
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("General Population")
        st.write(health_risks["General Population"])
        
        st.subheader("Kids & Infants")
        st.write(health_risks["Kids & Infants"])
    
    with col2:
        st.subheader("Elderly & Sick People")
        st.write(health_risks["Elderly & Sick"])
        
        st.subheader("Athletes & Outdoor Workers")
        st.write(health_risks["Athletes & Outdoor Workers"])
    
    st.markdown("---")
    
    # Health recommendations
    st.subheader("💊 Health Recommendations")
    if avg_aqi > 200:
        st.error("🚨 **Severe Air Quality Alert!**")
        st.write("- Stay indoors as much as possible")
        st.write("- Keep windows and doors closed")
        st.write("- Use air purifiers with HEPA filters")
        st.write("- Wear N95 masks if outdoor exposure is necessary")
        st.write("- Seek immediate medical attention if experiencing severe symptoms")
    elif avg_aqi > 150:
        st.warning("⚠️ **Unhealthy Air Quality**")
        st.write("- Avoid outdoor activities, especially for sensitive groups")
        st.write("- Limit window opening")
        st.write("- Consider using air purifiers")
        st.write("- Monitor health symptoms")
    elif avg_aqi > 100:
        st.info("ℹ️ **Moderate to Unhealthy**")
        st.write("- Sensitive groups should limit outdoor activities")
        st.write("- General public may engage in outdoor activities moderately")
        st.write("- Consider using masks for vulnerable populations")
    else:
        st.success("✅ **Good to Moderate Air Quality**")
        st.write("- Safe for outdoor activities")
        st.write("- No health precautions needed")

# Tab 5: Running Suitability
with tab5:
    st.header("🏃 Running Suitability Assessment")
    
    avg_aqi = (rf_aqi + xgb_aqi + lstm_aqi) / 3
    suitability, recommendation, color = assess_running_suitability(avg_aqi, wind_speed)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Suitability Status")
        if "Excellent" in suitability:
            st.success(suitability)
        elif "Good" in suitability:
            st.info(suitability)
        elif "Fair" in suitability:
            st.warning(suitability)
        else:
            st.error(suitability)
    
    with col2:
        st.subheader("Wind Speed")
        st.metric("Wind Speed", f"{wind_speed:.2f} m/s", "Good wind flow" if wind_speed > 0.5 else "Light wind")
    
    st.markdown("---")
    
    st.subheader("📋 Recommendation")
    st.write(f"**{recommendation}**")
    
    # Running tips based on AQI
    st.markdown("---")
    st.subheader("🏅 Running Tips")
    if avg_aqi <= 50:
        st.success("✅ Ideal conditions - Run at any time!")
    elif avg_aqi <= 100:
        st.info("⚠️ Run during morning hours (5-7 AM) when pollution is lower")
    elif avg_aqi <= 150:
        st.warning("⚠️ Short runs only - Run for 20-30 minutes maximum")
        st.write("- Choose routes away from traffic")
        st.write("- Use a pollution mask if needed")
    else:
        st.error("❌ Postpone running - Choose indoor exercise instead")
        st.write("- Consider gym workouts or home exercises")
        st.write("- Wait for air quality to improve")

# Tab 6: Learn More (Limitations & Future Work)
with tab6:
    st.header("📚 Data Overview, Limitations & Future Work")
    
    # Data Overview
    st.subheader("📊 Dataset Information")
    st.write("""
    **Source**: CPCB-based "Air Quality in India (2015–2024)"
    
    **Key Statistics**:
    - **Time Period**: 2015 - 2024 (9 years)
    - **Geographic Scope**: Multiple cities across India
    - **Features**: 13 pollutants and temporal indicators
    - **Samples**: Comprehensive daily measurements
    
    **Pollutants Tracked**:
    """)
    
    pollutant_info = pd.DataFrame({
        "Pollutant": ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3"],
        "Unit": ["µg/m³", "µg/m³", "ppb", "ppb", "ppb", "ppb", "ppm", "ppb", "ppb"],
        "Source": ["Traffic, Industry", "Vehicle Exhaust", "Vehicles", "Vehicles", "NOx compounds", "Agricultural", "Vehicles", "Industry", "Ozone formation"],
        "Health Impact": ["Respiratory", "Respiratory", "Respiratory", "Respiratory", "Acidification", "Agricultural", "Toxic", "Respiratory", "Respiratory"]
    })
    
    st.dataframe(pollutant_info, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Limitations
    st.subheader("⚠️ Model Limitations")
    
    st.warning("""
    1. **Geographic Limitations**:
       - Model trained on Indian cities; may not generalize to other regions
       - Urban-centric data; rural areas may have different patterns
    
    2. **Data Limitations**:
       - Historical data up to 2024; doesn't account for future climate changes
       - Daily measurements; sub-daily variations not captured
       - Missing data points that may affect trend analysis
    
    3. **Feature Limitations**:
       - 13 core pollutants; secondary pollutants not included
       - Meteorological factors (wind, humidity, temperature) not directly used
       - Industrial activities and vehicle emissions patterns may differ
    
    4. **Model Limitations**:
       - Random Forest: Limited temporal learning; assumes past patterns continue
       - XGBoost: May overfit on specific regions or time periods
       - LSTM: Requires more sophisticated tuning and more data
    
    5. **Health Alert Uncertainty**:
       - Health thresholds are population averages; individual sensitivity varies
       - Long-term exposure effects not captured in real-time alerts
    """)
    
    st.markdown("---")
    
    # Future Work
    st.subheader("🚀 Future Enhancements")
    
    st.info("""
    **Short-term (6 months)**:
    - [ ] Incorporate real-time meteorological data (wind, humidity, temperature)
    - [ ] Expand to international data for global model
    - [ ] Implement ensemble weighted predictions (RF 50%, XGB 30%, LSTM 20%)
    - [ ] Add satellite imagery for PM2.5 verification
    - [ ] Mobile app for push notifications
    
    **Medium-term (1 year)**:
    - [ ] Implement attention mechanism for better temporal learning
    - [ ] Add vehicle traffic and industrial activity data
    - [ ] Deploy as REST API for third-party integrations
    - [ ] Create city-specific models with local tuning
    - [ ] Add forecasting capability (24-48 hour predictions)
    
    **Long-term (2+ years)**:
    - [ ] Distribute shift adaptation to handle climate change
    - [ ] Integration with IoT sensors for real-time ground truth
    - [ ] Multi-scale modeling (neighborhood-level precision)
    - [ ] Causal inference to understand pollutant relationships
    - [ ] Optimization recommendations for city planners
    """)
    
    st.markdown("---")
    
    # Pollutant Trends (Placeholder)
    st.subheader("📈 Average Monthly Pollutant Levels Over Time")
    
    # Create sample pollutant trend data
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pm25_trend = [85, 82, 75, 60, 45, 35, 38, 42, 55, 70, 80, 88]
    pm10_trend = [150, 145, 135, 115, 85, 65, 70, 80, 105, 135, 150, 160]
    no2_trend = [45, 43, 38, 32, 25, 22, 24, 28, 35, 42, 48, 50]
    
    fig_trends = go.Figure()
    fig_trends.add_trace(go.Scatter(x=months, y=pm25_trend, mode='lines+markers', name='PM2.5'))
    fig_trends.add_trace(go.Scatter(x=months, y=pm10_trend, mode='lines+markers', name='PM10'))
    fig_trends.add_trace(go.Scatter(x=months, y=no2_trend, mode='lines+markers', name='NO2'))
    
    fig_trends.update_layout(
        title="Average Monthly Pollutant Levels Trend",
        xaxis_title="Month",
        yaxis_title="Concentration",
        hovermode="x unified",
        height=400
    )
    st.plotly_chart(fig_trends, use_container_width=True)
    
    st.markdown("---")
    
    # Contact & Resources
    st.subheader("💬 Questions & Feedback")
    st.write("""
    For questions, feedback, or collaboration opportunities, please reach out.
    
    **Resources**:
    - [CPCB Air Quality Data](https://www.cpcb.nic.in/)
    - [WHO Air Quality Guidelines](https://www.who.int/teams/environment-climate-change-and-health/air-quality-and-health/air-quality-guidelines)
    - [EPA AQI Explanation](https://www.epa.gov/air-quality/air-quality-index-aqi)
    """)


