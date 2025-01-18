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
from sklearn.preprocessing import StandardScaler
from folium.plugins import MarkerCluster
from folium.plugins import HeatMap
import geopandas as gpd
from geopy.distance import geodesic

# Import Other ML Libraries
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Page Title and Favicon
st.set_page_config(
    page_title="everyAir – Your everyday air",
    page_icon="⛅",
)

# Custom Integrate CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("EveryAir/style.css")

# Load Datasets
file_path = "EveryAir/Asia_Dataset.csv"
df = pd.read_csv(file_path)
street_df = pd.read_csv("EveryAir/Global_Street_Density.csv")
street_df = street_df[street_df['Regions'].str.contains('Asia', case=False, na=False)]
street_df = street_df[street_df['Area_of_Interest'].str.contains('Total', case=False, na=False)]

pop_df = pd.read_csv("EveryAir/pop_density.csv")
ndvi_df = pd.read_csv("EveryAir/ndvi.CSV")

ndvi_df.replace(99999.0, np.nan, inplace=True)

def round_to_nearest_0_10(value):
    return round(value * 10) / 10
def round_to_nearest_0_5(value):
    return round(value)

df_merged = pd.merge(df, street_df, on='City', how='outer')
df_merged = pd.merge(df_merged, pop_df, on='City', how='inner')
df_merged_filtered = df_merged.groupby('Country_x').head(10)

@st.cache_data(ttl=3600)
def get_cities(df, max_cities=150):
    # Extracting city names from the 'City' column of the DataFrame
    cities =  df['City'].dropna().unique().tolist()
    cities = sorted(cities)
    return cities[:max_cities]

cities = get_cities(street_df, max_cities=150)

# User's Input & Selection
st.sidebar.image("EveryAir/Location1.svg", width=283, use_container_width=False)

# API Call for city coordinates | OpenWeatherMap
API_KEY = '1608a88c9b9447cdb307c577157dcac5' #API Key for OpenWeatherAPI

# GeoCoding API | OpenWeatherMap
@st.cache_data(ttl=3600)
def get_coords(city_name, API_KEY, state_code="", country_code="", limit=100):
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
for city in cities:
    coords = get_coords(city, API_KEY)
    if coords:
        city_coordinates[city] = coords

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

# Fetch Additional Weather Data | Meteorological Data
@st.cache_data(ttl=3600)
def fetch_additional(lat, lon): 
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            data = response.json()
            temperature = data.get('main', {}).get('temp', None) - 273.15
            humidity = data.get('main', {}).get('humidity', None)
            wind_speed = data.get('wind', {}).get('speed', None)
            min_temp = data.get('main', {}).get('temp_min', None) - 273.15
            max_temp = data.get('main', {}).get('temp_max', None) - 273.15
            rainfall = data.get('rain', {}).get('1h', 0)

            return temperature, humidity, wind_speed, min_temp, max_temp, rainfall
        except (KeyError, IndexError) as e:
            st.error(f"Parsing Failed: {e}")
            return None, None, None, None, None, None
    else:
        st.error(f"Error: {response.status_code}")
        return None

def fetch_urban(df_merged, city):
    urban = df_merged[df_merged['City'] == city].iloc[0]
    
    # Get values for the selected city
    pop_density = urban['pop_density']
    street_density = urban['Street_density_Km_per_Km2']
    
    return pop_density, street_density

def fetch_ndvi(lat_query, lon_query):
    longitude_row = list(map(float, ndvi_df.columns[1:])) # Longitude
    latitude_col = list(map(float, ndvi_df.iloc[1:, 0]))  # Latitude
    ndvi_values = ndvi_df.iloc[1:, 1:].values
    ndvi_ext = pd.DataFrame(ndvi_values, columns=longitude_row, index=latitude_col)

    lat_rounded = round(lat_query * 10) / 10
    lon_rounded = round(lon_query)
    lon_rounded = min(longitude_row, key=lambda x: abs(x - lon_rounded))
    lat_rounded = min(latitude_col, key=lambda x: abs(x - lat_rounded))
    ndvi = ndvi_ext.loc[lat_rounded, lon_rounded];
    
    return ndvi

def fetch_industrial_sites_near_city(lat, lon, radius_km=50):
    url = "http://overpass-api.de/api/interpreter"
    
    # Roughly convert radius in km to degrees (1 degree ≈ 111 km)
    delta = radius_km / 111.0
    
    # Bounding box
    bbox = f"{lat - delta},{lon - delta},{lat + delta},{lon + delta}"
    
    query = f"""
    [out:json][timeout:1800];
    (
      node["landuse"="industrial"]({bbox});
      way["landuse"="industrial"]({bbox});
      relation["landuse"="industrial"]({bbox});
    );
    out body;
    >;
    out skel qt;
    """
    
    # Send query to Overpass API
    response = requests.get(url, params={'data': query})
    
    # Check the response status
    if response.status_code == 200:
        data = response.json()
        if 'elements' in data:
            industrial_sites = []
            for element in data['elements']:
                if 'lat' in element and 'lon' in element:
                    industrial_sites.append((element['lat'], element['lon']))
            return industrial_sites
        else:
            print(f"No 'elements' found in response: {data}")
            return []
    else:
        print(f"Error Fetching Data: {response.status_code} - {response.text}")
        return []

industrial_sites = fetch_industrial_sites_near_city(latitude, longitude)

# Input to float
try:
    latitude = float(latitude)
    longitude = float(longitude)
except ValueError:
    st.error("Latitude and Longitude are NaN.")

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

temperature, humidity, wind_speed, min_temp, max_temp, rainfall = fetch_additional(latitude, longitude) # fetch weather
ndvi = fetch_ndvi(latitude, longitude) # fetch NDVI based on lat/lon
dist = fetch_industrial_sites_near_city(latitude, longitude) # fetch nearest industrial sites
if temperature is not None:
    st.sidebar.write("🌦️ Live Weather Information:")
    st.sidebar.code(f"🌡️ Temperature: {temperature:.0f}°C")
    st.sidebar.code(f"💧 Humidity: {humidity} %")
    st.sidebar.code(f"☔ Rainfall: {rainfall} mm")
else:
    st.sidebar.write("Data Fetch Failed:\n")
    st.sidebar.write("Please try again later (￣﹏￣；)")

pop_density, street_density = fetch_urban(df_merged, city)
if pop_density is not None:
    X['pop_density'] = pop_density
    X['street_density'] = street_density
    X['NDVI'] = ndvi
    X['dist'] = np.min(dist)

# Extended Models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
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

best_model = min(model_scores, key=model_scores.get)
best_model_instance = models[best_model]

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
            st.error("Parsing Failed")
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
umbrella = ["🌂", "☂️", "☔"]
weathers = ["🌤️", "⛅", "🌦️", "🌧️", "⛈️"]
placeholder = st.empty()

if 'show_content' not in st.session_state:
    st.session_state.show_content = False
if st.button("Click to start the app"): 
    st.session_state.show_content = True
    for _ in range(2):
        for emoji in globe:
            placeholder.markdown(f"<h3 style='text-align:center; color: #004a0d; font-family: monospace, sans-serif; margin-top: 0.9em; margin-bottom: -15em; transition: opacity 350ms ease-in-out, transform 350ms ease-in-out; transform: translateY(0.2em);'>rotating planet... {emoji} </h3>", unsafe_allow_html=True)
            time.sleep(0.3)
    placeholder.empty()
    for _ in range(2):
        for emoji in umbrella:
            placeholder.markdown(f"<h3 style='text-align:center; color: #004a0d; font-family: monospace, sans-serif; margin-top: 0.9em; margin-bottom: -15em; transition: opacity 350ms ease-in-out, transform 350ms ease-in-out; transform: translateY(0.2em);'>collecting rain... {emoji} </h3>", unsafe_allow_html=True)
            time.sleep(0.5)
    for _ in range(2):
        for emoji in weathers:
            placeholder.markdown(f"<h3 style='text-align:center; color: #004a0d; font-family: monospace, sans-serif; margin-top: 0.9em; margin-bottom: -15em; transition: opacity 350ms ease-in-out, transform 350ms ease-in-out; transform: translateY(0.2em);'>checking weather... {emoji} </h3>", unsafe_allow_html=True)
            time.sleep(0.4)
    placeholder.empty()
    for _ in range(2):
        placeholder.markdown(f"<h3 style='text-align:center; color: #004a0d; font-family: monospace, sans-serif; margin-top: 0.9em; margin-bottom: -15em; transition: opacity 350ms ease-in-out, transform 350ms ease-in-out; transform: translateY(0.2em);'>enjoy today's air!</h3>", unsafe_allow_html=True)
        time.sleep(0.4)
    

# Gauge meter for PM2.5
if st.session_state.show_content:
    st.title("Weather & Urban")
    input_features = [[current_month, current_year]]
    def create_gauge_chart(pm2_5_value, prediction_value, city):
        fig = go.Figure()
        placeholder.empty()
        # Gauge meter according to Real-Time Data
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=pm2_5_value,
            title={"text": f"Real-Time PM2.5 Level for {city}"},
            gauge={
                "axis": {"range": [0, 500]},
                "steps": [
                    {"range": [0, 50], "color": "#5bfc6b"},    
                    {"range": [50, 100], "color": "#dcfc5b"},
                    {"range": [100, 150], "color": "#fc965b"},
                    {"range": [150, 200], "color": "#ff4b33"},  
                    {"range": [200, 500], "color": "#9e1010"}
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

    def show_city_on_map(city, latitude, longitude, input_features, best_model_instance, real_time_pm2_5=None):
        st.write(f"### Heatmap for {city}")

        m = folium.Map(location=[latitude, longitude], zoom_start=9)

        prediction_value = best_model_instance.predict(input_features)[0]
        folium.Marker(
            location=[latitude, longitude],
            popup=f"PM2.5 Level: {real_time_pm2_5:.2f} µg/m³" if real_time_pm2_5 is not None else "PM2.5 Data not available",
            icon=folium.Icon(color='navy', icon='info-sign')
        ).add_to(m)

        lat_grid = np.linspace(latitude - 0.1, latitude + 0.1, 21)
        lon_grid = np.linspace(longitude - 0.1, longitude + 0.1, 21)
        lat_grid, lon_grid = np.meshgrid(lat_grid, lon_grid)
        grid_points = np.vstack([lat_grid.ravel(), lon_grid.ravel()]).T

        predictions = best_model_instance.predict(grid_points)

        heatmap_data = [
            [lat_grid.ravel()[i], lon_grid.ravel()[i], predictions[i]] 
            for i in range(len(grid_points))
        ]

        HeatMap(heatmap_data, radius=15, blur=25, max_val=200).add_to(m)

        folium_static(m)


    def heatmap_show(pm2_5_grid, lat_grid, lon_grid):
        plt.figure(figsize=(8, 6))
        plt.imshow(pm2_5_grid, extent=[lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()],
                   origin='lower', cmap='YlOrRd', interpolation='nearest', vmin=0, vmax=200)

        plt.colorbar(label='PM2.5 Level')

        plt.title('Predicted PM2.5 Heatmap')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        st.pyplot(plt)

    # Predict PM2.5
    if real_time_pm2_5 is not None:
        input_features = [[current_month, current_year]]
        predicted_pm2_5 = best_model_instance.predict(input_features)[0]

        lat_grid = np.linspace(latitude, latitude + 0.3, 5)
        lon_grid = np.linspace(longitude, longitude + 0.3, 5)
        
        lat_grid, lon_grid = np.meshgrid(lat_grid, lon_grid)
        grid_points = np.vstack([lat_grid.ravel(), lon_grid.ravel()]).T
        heatmap_pm2_5 = best_model_instance.predict(grid_points).reshape(lat_grid.shape)
        print(heatmap_pm2_5)


        # The Extra Weather Details
        if temperature is not None and pop_density is not None: 
            col1,col2,col3,col4 = st.columns(4)
            with col1:
                st.metric("👤 Population Density", round(pop_density), "people/km2")
            with col2:
                st.metric("🛣️ Street Density", street_density, "km/km2")
            with col3:
                st.metric("🍃 NDVI", ndvi, "%Index")
            with col4:
                st.metric("🏭 Nearest Indust.", round(np.min(dist)), "km")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🌡️ Temperature", round(temperature), "°C")
            with col2:
                st.metric("❄️ Min Temp", round(min_temp), "°C")
            with col3:
                st.metric("🔥 Max Temp", round(max_temp), "°C")
            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("💨 Wind Speed", wind_speed, "m/s")
            with col5:
                st.metric("💧 Humidity", humidity, "%")
            with col6:
                st.metric("☔ Rainfall", rainfall, "mm")
        

        # Display gauge
        create_gauge_chart(real_time_pm2_5, predicted_pm2_5, city)

        # Yearly avg prediction input
        st.write("### Predict PM2.5 for a specific month")
        month = st.selectbox("Select Month:", list(month_map.keys()))
        yearly_avg = st.number_input("Enter yearly average pollution level:", value=100.0)
        # Prediction
        if st.button("Predict"):
            with st.spinner(text="Predicting..."):
                time.sleep(2)
            month_numeric = month_map[month]
            input_features = [[month_numeric, yearly_avg]]
            prediction = best_model_instance.predict([[month_numeric, yearly_avg]])
            with st.status("Predicting Data..."):
                st.write("Fetching trained data...")
                time.sleep(2)
                st.write("Making prediction...")
                time.sleep(1)
                st.write("Success!")
                time.sleep(1)
        st.success(f"🔎\tPredicted PM2.5 level for {month}: **{prediction[0]:.2f}**\t")
        st.title("Forecast Results")

        # Visualisation
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=data['Month'], y=data['PM2.5'], mode='lines', name='Historical PM2.5'))

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
        st.code(f"📡 Real-Time PM2.5 Data for {city}: {real_time_pm2_5:.2f} µg/m³")
        st.code(f"🔮 Predicted PM2.5 Level for {city}: {predicted_pm2_5:.2f} µg/m³")
        show_city_on_map(city, latitude, longitude, input_features, best_model_instance, real_time_pm2_5=real_time_pm2_5)   
