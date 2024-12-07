# Import required prerequisites and libraries
import pandas as pd
import warnings
import requests
import streamlit as st
import os
from matplotlib import pyplot as plt
from datetime import datetime
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Custom Integrate CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("EveryAir/style.css")

st.set_page_config(
    page_title="everyAir",
    page_icon="everyAir.png"
)

# Load Dataset (Historical)
file_path = "EveryAir/Asia_Dataset.csv"
df = pd.read_csv(file_path)

# User's Input & Selection
st.sidebar.image("EveryAir/Location1.svg", use_container_width=True)
city_coordinates = {
    'Tokyo': (35.6895, 139.6917),
    'Delhi': (28.6139, 77.2090),
    'Beijing': (39.9042, 116.4074),
    'Bangkok': (13.7563, 100.5018),
    'Lahore': (31.5204, 74.3587),
    'New Delhi': (28.6139, 77.2090),
    'Mumbai': (19.0760, 72.8777),
    'Kolkata': (22.5726, 88.3639),
}
city = st.sidebar.selectbox("Select City", list(city_coordinates.keys()), index=0)
st.sidebar.write(f"City Selected: {city}")
latitude, longitude = city_coordinates[city]
st.sidebar.code(f"\tLatitude: {latitude}")
st.sidebar.code(f"\tLongitude: {longitude}")

# Convert month string to numerical val
month_map = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

# Data filter for selected city
data = df[df['City'] == city]
data = data.melt(id_vars=['Rank', 'City', 'Country', '2023'], 
                 var_name='Month', value_name='PM2.5')
data['PM2.5'] = pd.to_numeric(data['PM2.5'], errors='coerce')
data['Month'] = data['Month'].map(month_map)
data = data.dropna(subset=['PM2.5'])
print(data.isnull().sum())

# Features and target
X = data[['Month', '2023']]
y = data['PM2.5']

# Splitting | Train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Current Date
current_month = datetime.now().month
current_year = datetime.now().year

# Training Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error for {city}: {mae}")

# API Key & Default Vals
API_KEY = os.getenv('OPENWEATHERMAP_API_KEY', '1608a88c9b9447cdb307c577157dcac5')
lat_default, lon_default = 35.6895, 139.6917  # Default: Tokyo
city_default = "Tokyo"  # Default City

# Input to float
try:
    latitude = float(latitude)
    longitude = float(longitude)
except ValueError:
    st.error("Latitude and Longitude must be numeric.")
    latitude = lat_default
    longitude = lon_default

# Real-Time Data
def fetch_real_time_pm2_5(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            data = response.json()
            pm2_5 = data['list'][0]['components']['pm2_5']
            return pm2_5
        except (KeyError, IndexError):
            st.error("Parsing Error")
            return None

real_time_pm2_5 = fetch_real_time_pm2_5(latitude, longitude)

# Streamlit Web Deployment
# Logo
logo = "EveryAir/everyAirFinal.svg"
col1, col2, col3 = st.columns([1, 1.7, 1])
with col2:
    st.image(logo, use_container_width=400)

if st.button("Click to start the app"): 
    st.session_state.show_content = True
    # Gauge meter for PM2.5
if st.session_state.show_content:
    def create_gauge_chart(pm2_5_value, prediction_value, city):
        fig = go.Figure()

        # Gauge meter according to Real-Time Data
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=pm2_5_value,
            title={"text": f"Real-Time PM2.5 Level for {city}"},
            gauge={
                "axis": {"range": [0, 500]},
                "steps": [
                    {"range": [0, 50], "color": "green"},    
                    {"range": [50, 100], "color": "yellow"},
                    {"range": [100, 150], "color": "orange"},
                    {"range": [150, 200], "color": "red"},  
                    {"range": [200, 500], "color": "darkred"}
                ],
                "bar": {"color": "black"}
            }
        ))
        fig.update_layout(
            height=400,
            width=670,
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=False)


    def show_city_on_map(city, latitude, longitude, prediction_value=None, real_time_pm2_5=None):
    # Map with LAT & LON
        st.code(f"Real-time PM2.5 value: {real_time_pm2_5}")
        m = folium.Map(
            location=[latitude, longitude],
            zoom_start=12,
        )
    
        folium.Marker(
            location=[latitude, longitude],
            popup=f"PM2.5 Level: {real_time_pm2_5:.2f} µg/m³" if real_time_pm2_5 is not None else "PM2.5 Data not available",
            icon=folium.Icon(color='navy', icon='info-sign')
        ).add_to(m)
    
        if real_time_pm2_5 is not None:
            if real_time_pm2_5 < 50:
                color = 'green'
            elif real_time_pm2_5 < 100:
                color = 'yellow'
            elif real_time_pm2_5 < 150:
                color = 'orange'
            elif real_time_pm2_5 < 200:
                color = 'red'
            else:
                color = 'darkred'
    
            folium.CircleMarker(
                location=[latitude, longitude],
                radius=100,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                popup=f"Real-Time PM2.5: {real_time_pm2_5:.2f} µg/m³"
            ).add_to(m)
    
        # Prediction indicator
        if prediction_value is not None:
            folium.Circle(
                location=[latitude, longitude],
                radius=2000,
                color='red',
                weight=2,
                fill=True,
                fill_color='red',
                fill_opacity=0.3,
                popup=f"Predicted PM2.5: {prediction_value:.2f} µg/m³"
            ).add_to(m)
    
        # Streamlit Map Render
        folium_static(m)

    # Predict PM2.5
    if real_time_pm2_5 is not None:

        input_features = [[current_month, current_year]]
        predicted_pm2_5 = model.predict(input_features)[0]

        # Display gauge
        create_gauge_chart(real_time_pm2_5, predicted_pm2_5, city)

        # Yearly avg prediction input
        st.write("### Predict PM2.5 for a specific month")
        month = st.selectbox("Select Month:", list(month_map.keys()))
        yearly_avg = st.number_input("Enter yearly average (2023) pollution level:", value=100.0)

        # Prediction
        if st.button("Predict"):
            month_numeric = month_map[month]
            input_features = [[month_numeric, yearly_avg]]
            prediction = model.predict([[month_numeric, yearly_avg]])

        st.write(f"⚠️\tPredicted PM2.5 level for {month}: **{prediction[0]:.2f}**\t⚠️")
        st.header("Forecast Results")
        st.code(f"📡 Real-Time PM2.5 Data for {city}: {real_time_pm2_5:.2f} µg/m³")
        st.code(f"🔮 Predicted PM2.5 Level for {city}: {predicted_pm2_5:.2f} µg/m³")

        # Visualisation
        fig = go.Figure()
        # Plot historical data
        fig.add_trace(go.Scatter(x=data['Month'], y=data['PM2.5'], mode='lines', name='Historical PM2.5'))

        # Plot only when real-time PM2.5 is available
        if real_time_pm2_5 is not None:
            fig.add_trace(go.Scatter(
                x=[current_month], y=[predicted_pm2_5], mode='markers+text', name='Prediction',
                marker=dict(color='red', size=10),
                text=[f"{predicted_pm2_5:.2f} µg/m³"], textposition="top center"
            ))


        fig.update_layout(
            title=f"PM2.5 Forecast vs Historical Data for {city}",
            xaxis_title="Month",
            yaxis_title="PM2.5 (µg/m³)",
            template="plotly_dark"
        )

        st.plotly_chart(fig)

        # If default: January's prediction
        if real_time_pm2_5 is None:
            fig.add_trace(go.Scatter(
                x=[1], y=[model.predict([[1, 100.0]])[0]], mode='markers+text', name='January Prediction',
                marker=dict(color='red', size=10),
                text=[f"{model.predict([[1, 100.0]])[0]:.2f} µg/m³"], textposition="top center"
            ))
            st.plotly_chart(fig)

        show_city_on_map(city, latitude, longitude, prediction_value=predicted_pm2_5, real_time_pm2_5=real_time_pm2_5)
