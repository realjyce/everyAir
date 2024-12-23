# Import required prerequisites and libraries
import pandas as pd
import warnings
import requests
import streamlit as st
import os
import time
from matplotlib import pyplot as plt
from datetime import datetime
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Import Other ML Libraries
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

#Page Title and Favicon
st.set_page_config(
    page_title="everyAir – Your everyday air",
    page_icon="⛅",
)

# Custom Integrate CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("EveryAir/style.css")

# Load Dataset (Historical)
file_path = "EveryAir/Asia_Dataset.csv"
df = pd.read_csv(file_path)

# User's Input & Selection
st.sidebar.image("EveryAir/Location1.svg", width=283, use_container_width=False)

# API Call for city coordinates | OpenWeatherMap & GeoNames (for cities)
API_KEY = os.getenv('OPENWEATHERMAP_API_KEY', '1608a88c9b9447cdb307c577157dcac5') #API Key for OpenWeatherAPI

# GeoNames API
GEO_NAMES_API = "jyce" # Username for API Key
@st.cache_data
def get_cities(GEO_NAMES_API, max_cities=10):
    url = f"http://api.geonames.org/searchJSON?formatted=true&continentCode=AS&maxRows=100&lang=en&username={GEO_NAMES_API}"
    response = requests.get(url);

    if response.status_code == 200:
        data = response.json()
        cities = [city['name'] for city in data['geonames'][:max_cities]]
        return cities
    else:
        return []
cities = get_cities(GEO_NAMES_API, max_cities=10)
# GeoCoding API | OpenWeatherMap
@st.cache_data
def get_coords(city_name, API_KEY, state_code="", country_code="", limit=1):

    url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name},{state_code},{country_code}&limit={limit}&appid={API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data:
            return data[0]['lat'], data[0]['lon']
        else:
            return None
    else:
        return None
    

city_coordinates = {}

# City Coords & Listing
for city in cities:
    coords = get_coords(city, API_KEY)
    if coords:
        city_coordinates[city] = coords

cities = [city for city in cities if city not in ["Asia", "Southern Asia"]]

if cities:
    city = st.sidebar.selectbox("Select City", cities)
    if city in city_coordinates:
        latitude, longitude = city_coordinates[city]
        st.sidebar.write(f"🌍 City Selected: **{city}**")
        st.sidebar.code(f"\tLatitude: {latitude:.4f}°")
        st.sidebar.code(f"\tLongitude: {longitude:.4f}°")
    else:
        st.sidebar.error("Not Found")


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

# Fetch Additional Weather Data | Temperature & Humidity
@st.cache_data
def fetch_additional(lat, lon): 
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            data = response.json()
            temperature = data.get('main', {}).get('temp', None) - 273.15
            humidity = data.get('main', {}).get('humidity', None)
            return temperature, humidity
        except (KeyError, IndexError) as e:
            st.error("Parsing Error: {e}")
            return None, None
    else:
        st.error("Error: {response.status_code}")
        return None

# Input to float
try:
    latitude = float(latitude)
    longitude = float(longitude)
except ValueError:
    st.error("Latitude and Longitude must be numeric.")

# Features and target
X = data[['Month', '2023']].copy()
# New Humidity and temperature feature
if 'Temperature' in data.columns and 'Humidity' in data.columns:
    X['Temperature'] = data['Temperature']
    X['Humidity'] = data['Humidity']

y = data['PM2.5']

# Splitting | Train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Current Date
current_month = datetime.now().month
current_year = datetime.now().year

temperature, humidity = fetch_additional(latitude, longitude)
if temperature is not None:
    st.sidebar.write("🌦️ Live Weather Information:")
    st.sidebar.code(f"🌡️ Temperature: {temperature:.0f}°C")
    st.sidebar.code(f"💧 Humidity: {humidity} %")

    X['Temperature'] = temperature
    X['Humidity'] = humidity
else:
    st.sidebar.write("Data Fetch Failed:\n");
    st.sidebar.write("Please try again later (￣﹏￣；)");

# Extended Models
models = {
    "Random Forest": RandomForestRegressor(n_estimators= 100, random_state=42),
    "Linear Regression": LinearRegression(),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
}

model_scores = {}
for model_type, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    model_scores[model_type] = mae

model_scores = {}
for model_type, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    model_scores[model_type] = mae

best_model = min(model_scores, key=model_scores.get)
best_model_instance = models[best_model]

# Real-Time Data

@st.cache_data
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

# Loading Animation
globe = ["🌍", "🌎", "🌏"]
dots = [".", "..", "..."]
placeholder = st.empty()

if 'show_content' not in st.session_state:
    st.session_state.show_content = False
if st.button("Click to start the app"): 
    st.session_state.show_content = True
    for _ in range(10):
      for emoji in globe:
          placeholder.markdown(f"<h3 style='text-align:center; color: #004a0d; font-family: monospace, sans-serif; margin-top: 0.9em; margin-bottom: -15em; transition: opacity 350ms ease-in-out, transform 350ms ease-in-out; transform: translateY(0.2em);'>rotating planet... {emoji} </h3>", unsafe_allow_html=True)
          time.sleep(0.5)
    # Gauge meter for PM2.5
if st.session_state.show_content:
    def create_gauge_chart(pm2_5_value, prediction_value, city):
        fig = go.Figure()
        placeholder.empty()
        # Gauge meter acc   ording to Real-Time Data
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
        predicted_pm2_5 = best_model_instance.predict(input_features)[0]

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
            prediction = best_model_instance.predict([[month_numeric, yearly_avg]])

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
                x=[1], y=[best_model_instance.predict([[1, 100.0]])[0]], mode='markers+text', name='January Prediction',
                marker=dict(color='red', size=10),
                text=[f"{best_model_instance.predict([[1, 100.0]])[0]:.2f} µg/m³"], textposition="top center"
            ))
            st.plotly_chart(fig)

        show_city_on_map(city, latitude, longitude, prediction_value=predicted_pm2_5, real_time_pm2_5=real_time_pm2_5)
