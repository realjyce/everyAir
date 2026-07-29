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
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from folium.plugins import MarkerCluster
from folium.plugins import HeatMap
import geopandas as gpd

# Import Other ML Libraries
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
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

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_HEADERS = {"User-Agent": "everyAir/1.0 (https://github.com/realjyce/everyAir)"}


@st.cache_data(ttl=3600)
def fetch_industrial_sites_near_city(lat, lon, radius_km=50):
    delta = radius_km / 111.0
    bbox = f"{lat - delta},{lon - delta},{lat + delta},{lon + delta}"
    query = f"""
    [out:json][timeout:60];
    (
      way["landuse"="industrial"]({bbox});
      relation["landuse"="industrial"]({bbox});
      node["man_made"="works"]({bbox});
      way["man_made"="works"]({bbox});
    );
    out center;
    """
    try:
        response = requests.get(OVERPASS_URL, params={"data": query},
                                headers=OVERPASS_HEADERS, timeout=60)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return None

    sites = []
    for element in payload.get("elements", []):
        if "lat" in element and "lon" in element:
            sites.append((element["lat"], element["lon"]))
        elif "center" in element:
            sites.append((element["center"]["lat"], element["center"]["lon"]))
    return sites


def km_to_nearest_site(lats, lons, sites):
    site = np.radians(np.asarray(sites, dtype=float))
    la = np.radians(np.asarray(lats, dtype=float))[:, None]
    lo = np.radians(np.asarray(lons, dtype=float))[:, None]
    dlat = site[:, 0][None, :] - la
    dlon = site[:, 1][None, :] - lo
    a = np.sin(dlat / 2) ** 2 + np.cos(la) * np.cos(site[:, 0][None, :]) * np.sin(dlon / 2) ** 2
    return (6371.0 * 2 * np.arcsin(np.sqrt(a))).min(axis=1)


# Input to float
try:
    latitude = float(latitude)
    longitude = float(longitude)
except ValueError:
    st.error("Latitude and Longitude are NaN.")

MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
SEASON_FEATURES = ['month_sin', 'month_cos', 'log_anchor', 'country_clim', 'global_clim']


def _seasonal_panel():
    pooled = df.melt(id_vars=['Rank', 'City', 'Country', '2023'], value_vars=MONTH_NAMES,
                     var_name='MonthName', value_name='PM2.5')
    pooled['Month'] = pooled['MonthName'].map({m: i + 1 for i, m in enumerate(MONTH_NAMES)})
    pooled['PM2.5'] = pd.to_numeric(pooled['PM2.5'], errors='coerce')
    pooled = pooled.dropna(subset=['PM2.5'])

    total = pooled.groupby('City')['PM2.5'].transform('sum')
    count = pooled.groupby('City')['PM2.5'].transform('count')
    pooled['anchor'] = (total - pooled['PM2.5']) / (count - 1)
    pooled = pooled[(pooled['anchor'] > 0) & (count > 6)].reset_index(drop=True)

    pooled['ratio'] = pooled['PM2.5'] / pooled['anchor']
    pooled['month_sin'] = np.sin(2 * np.pi * pooled['Month'] / 12)
    pooled['month_cos'] = np.cos(2 * np.pi * pooled['Month'] / 12)
    pooled['log_anchor'] = np.log1p(pooled['anchor'])
    return pooled


def _attach_climatology(frame, country_table, global_table):
    frame = frame.copy()
    frame['country_clim'] = frame.set_index(['Country', 'Month']).index.map(country_table)
    frame['global_clim'] = frame['Month'].map(global_table)
    frame['country_clim'] = frame['country_clim'].fillna(frame['global_clim'])
    return frame


def _fit_quantiles(features, log_ratio):
    centre = HistGradientBoostingRegressor(
        random_state=42, max_iter=600, learning_rate=0.04, max_leaf_nodes=63)
    lower = HistGradientBoostingRegressor(
        random_state=42, loss='quantile', quantile=0.1, max_iter=400, learning_rate=0.05)
    upper = HistGradientBoostingRegressor(
        random_state=42, loss='quantile', quantile=0.9, max_iter=400, learning_rate=0.05)
    for model in (centre, lower, upper):
        model.fit(features, log_ratio)
    return centre, lower, upper


@st.cache_resource(show_spinner=False)
def train_seasonal_model():
    panel = _seasonal_panel()

    fold_mae, flat_mae, covered = [], [], []
    for train_idx, test_idx in GroupKFold(n_splits=5).split(panel, panel['ratio'], panel['City']):
        train, test = panel.iloc[train_idx], panel.iloc[test_idx]
        country_table = train.groupby(['Country', 'Month'])['ratio'].mean()
        global_table = train.groupby('Month')['ratio'].mean()
        train_f = _attach_climatology(train, country_table, global_table)
        test_f = _attach_climatology(test, country_table, global_table)

        centre, lower, upper = _fit_quantiles(
            train_f[SEASON_FEATURES], np.log(train_f['ratio'].clip(lower=1e-3)))
        anchors = test_f['anchor'].values
        predicted = np.exp(centre.predict(test_f[SEASON_FEATURES])) * anchors
        low = np.exp(lower.predict(test_f[SEASON_FEATURES])) * anchors
        high = np.exp(upper.predict(test_f[SEASON_FEATURES])) * anchors
        truth = test_f['PM2.5'].values

        fold_mae.append(mean_absolute_error(truth, predicted))
        flat_mae.append(mean_absolute_error(truth, anchors))
        covered.append(float(np.mean((truth >= low) & (truth <= high))))

    country_table = panel.groupby(['Country', 'Month'])['ratio'].mean()
    global_table = panel.groupby('Month')['ratio'].mean()
    full = _attach_climatology(panel, country_table, global_table)
    models = _fit_quantiles(full[SEASON_FEATURES], np.log(full['ratio'].clip(lower=1e-3)))

    return models, country_table, global_table, {
        'model_mae': float(np.mean(fold_mae)),
        'flat_mae': float(np.mean(flat_mae)),
        'coverage': float(np.mean(covered)) * 100,
        'rows': int(len(panel)),
        'cities': int(panel['City'].nunique()),
    }


season_models, country_clim, global_clim, model_report = train_seasonal_model()

city_months = data.sort_values('Month')
city_anchor = float(city_months['PM2.5'].mean())
city_country = str(df.loc[df['City'] == city, 'Country'].iloc[0]) if (df['City'] == city).any() else ''


def predict_month(month_value, anchor=None):
    anchor = city_anchor if anchor is None else anchor
    shape = global_clim.get(month_value, 1.0)
    row = pd.DataFrame([{
        'month_sin': np.sin(2 * np.pi * month_value / 12),
        'month_cos': np.cos(2 * np.pi * month_value / 12),
        'log_anchor': np.log1p(anchor),
        'country_clim': country_clim.get((city_country, month_value), shape),
        'global_clim': shape,
    }])[SEASON_FEATURES]
    centre, lower, upper = season_models
    return (float(np.exp(centre.predict(row)[0]) * anchor),
            float(np.exp(lower.predict(row)[0]) * anchor),
            float(np.exp(upper.predict(row)[0]) * anchor))


# Current Date
current_month = datetime.now().month
current_year = datetime.now().year

temperature, humidity, wind_speed, min_temp, max_temp, rainfall = fetch_additional(latitude, longitude)
ndvi = fetch_ndvi(latitude, longitude)
SEARCH_RADIUS_KM = 50
industrial_sites = fetch_industrial_sites_near_city(latitude, longitude)
industry_lookup_failed = industrial_sites is None
industrial_sites = industrial_sites or []
dist = (km_to_nearest_site([latitude], [longitude], industrial_sites).tolist()
        if industrial_sites else [])
nearest_industrial = min(dist) if dist else float(SEARCH_RADIUS_KM)
if temperature is not None:
    st.sidebar.write("Live weather")
    st.sidebar.code(f"Temperature: {temperature:.0f}°C")
    st.sidebar.code(f"Humidity: {humidity} %")
    st.sidebar.code(f"Rainfall: {rainfall} mm")
else:
    st.sidebar.write("Weather fetch failed, try again later.")

pop_density, street_density = fetch_urban(df_merged, city)

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
# The gate only exists before the app has started. Leaving the button on the
# page afterwards is what made it look like nothing had happened.
_gate = st.empty()
_start = False
if not st.session_state.show_content:
    with _gate.container():
        _l, _mid, _r = st.columns([1, 1.7, 1])
        with _mid:
            _start = st.button("Show me the air", use_container_width=True, type="primary")

if _start:
    st.session_state.show_content = True
    _gate.empty()
    for label, frames, pause in (("rotating planet", globe, 0.22),
                                 ("collecting rain", umbrella, 0.22),
                                 ("checking weather", weathers, 0.16)):
        for emoji in frames:
            placeholder.markdown(
                f"<h3 class='ea-loading'>{label}... {emoji}</h3>", unsafe_allow_html=True)
            time.sleep(pause)
    placeholder.empty()
    st.rerun()
    

# Gauge meter for PM2.5
if st.session_state.show_content:
    input_features = [[current_month, current_year]]
    def create_gauge_chart(pm2_5_value, prediction_value, city):
        fig = go.Figure()
        placeholder.empty()
        # Gauge meter according to Real-Time Data
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=pm2_5_value,
            title={"text": f"PM2.5 right now in {city} (µg/m³)"},
            gauge={
                "axis": {"range": [0, 250], "tickvals": [0, 9, 35.4, 55.4, 125.4, 250]},
                "steps": [
                    {"range": [0, 9], "color": "#5bfc6b"},
                    {"range": [9, 35.4], "color": "#dcfc5b"},
                    {"range": [35.4, 55.4], "color": "#fc965b"},
                    {"range": [55.4, 125.4], "color": "#ff4b33"},
                    {"range": [125.4, 250], "color": "#9e1010"}
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 3},
                    "thickness": 0.85,
                    "value": 15,
                },
                "bar": {"color": "#111111"}
            }
        ))
        fig.update_layout(
            height=340,
            template="plotly_dark",
            margin=dict(l=0, r=0, t=64, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Bands are US EPA PM2.5 breakpoints in µg/m³. The white line marks the "
            "WHO 24-hour guideline of 15 µg/m³. Live value is CAMS model output from "
            "OpenWeather, not a ground station."
        )

    def show_city_on_map(city, latitude, longitude, real_time_pm2_5=None, predicted_pm2_5=None):
        st.subheader(f"Where it settles in {city}")

        m = folium.Map(location=[latitude, longitude], zoom_start=9)

        folium.Marker(
            location=[latitude, longitude],
            popup=f"PM2.5 Level: {real_time_pm2_5:.2f} µg/m³" if real_time_pm2_5 is not None else "PM2.5 Data not available",
            icon=folium.Icon(color='navy', icon='info-sign')
        ).add_to(m)

        def draw_single_marker(note):
            folium.CircleMarker(
                location=[latitude, longitude],
                radius=30, color="#d7191c", weight=1,
                fill=True, fill_color="#d7191c", fill_opacity=0.28,
            ).add_to(m)
            folium_static(m, width=1120, height=520)
            st.caption(note)

        draw_single_marker(
            f"{predicted_pm2_5:.1f} µg/m³ predicted for {city} this month. The model "
            f"works at city level, so there is one value for the whole area rather than "
            f"a surface."
            if predicted_pm2_5 is not None else
            "No prediction available for this city."
        )


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
        predicted_pm2_5, predicted_low, predicted_high = predict_month(current_month)

        # Two rows of four rather than 4/3/3, which never lined up. The unit
        # belongs in the label: st.metric's third argument is a delta, so
        # passing "people/km2" there rendered a green up-arrow next to every
        # number as though everything had just improved.
        ICONS = {
            "population": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
            "streets": '<path d="M4 19V5"/><path d="M20 19V5"/><path d="M12 5v3"/><path d="M12 11v3"/><path d="M12 17v2"/>',
            "green": '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6"/>',
            "industry": '<path d="M2 20h20"/><path d="M4 20V9l5 3V9l5 3V9l5 3v8"/><path d="M9 20v-4h3v4"/>',
            "temp": '<path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0Z"/>',
            "wind": '<path d="M12.8 19.6A2 2 0 1 0 14 16H2"/><path d="M17.5 8a2.5 2.5 0 1 1 2 4H2"/><path d="M9.8 4.4A2 2 0 1 1 11 8H2"/>',
            "humidity": '<path d="M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5s-3.5-4-4-6.5c-.5 2.5-2 4.9-4 6.5S5 13 5 15a7 7 0 0 0 7 7Z"/>',
            "rain": '<path d="M4 14.9A7 7 0 1 1 15.7 8h1.8a4.5 4.5 0 0 1 0 9H7"/><path d="M8 19v2"/><path d="M12 19v3"/><path d="M16 19v2"/>',
        }

        def tile(icon, label, value, note=""):
            return (
                f'<div class="ea-tile">'
                f'<svg class="ea-tile__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">{ICONS[icon]}</svg>'
                f'<div class="ea-tile__body">'
                f'<span class="ea-tile__label">{label}</span>'
                f'<span class="ea-tile__value">{value}</span>'
                f'<span class="ea-tile__note">{note}</span>'
                f'</div></div>'
            )

        if temperature is not None and pop_density is not None:
            industry_value = f"{round(nearest_industrial)} km" if dist else f"{SEARCH_RADIUS_KM}+ km"
            industry_note = "to nearest site" if dist else "none mapped nearby"

            st.markdown("<p class='ea-group'>Urban</p>", unsafe_allow_html=True)
            st.markdown(
                "<div class='ea-grid'>"
                + tile("population", "Population", f"{round(pop_density):,}".replace(",", " "), "people per km²")
                + tile("streets", "Street density", f"{street_density:g}", "road km per km²")
                + tile("green", "Greenness", f"{ndvi:.2f}", "NDVI, 0 to 1")
                + tile("industry", "Industry", industry_value, industry_note)
                + "</div>",
                unsafe_allow_html=True,
            )

            _temp_note = (f"{round(min_temp)}° to {round(max_temp)}° today"
                          if round(min_temp) != round(max_temp) else "feels steady today")

            st.markdown("<p class='ea-group'>Weather</p>", unsafe_allow_html=True)
            st.markdown(
                "<div class='ea-grid'>"
                + tile("temp", "Temperature", f"{round(temperature)}°C", _temp_note)
                + tile("wind", "Wind", f"{wind_speed:g}", "metres per second")
                + tile("humidity", "Humidity", f"{humidity}%", "relative")
                + tile("rain", "Rainfall", f"{rainfall:g}", "mm in the last hour")
                + "</div>",
                unsafe_allow_html=True,
            )
        

        # Display gauge
        create_gauge_chart(real_time_pm2_5, predicted_pm2_5, city)

        # Yearly avg prediction input
        st.subheader("Predict a month")
        _city_avg = city_anchor
        pc1, pc2 = st.columns(2)
        with pc1:
            month = st.selectbox("Month", list(month_map.keys()))
        with pc2:
            yearly_avg = st.number_input(
                "Typical yearly level (µg/m³)", value=round(_city_avg, 1), step=1.0,
                help="The model predicts the seasonal shape and scales it by this level.")
        # Prediction
        if st.button("Predict"):
            with st.spinner(text="Predicting..."):
                time.sleep(2)
            month_numeric = month_map[month]
            prediction, prediction_low, prediction_high = predict_month(
                month_numeric, anchor=yearly_avg)
            with st.status("Predicting Data..."):
                st.write("Fetching trained data...")
                time.sleep(2)
                st.write("Making prediction...")
                time.sleep(1)
                st.write("Success!")
                time.sleep(1)
            # This sat one level out, so it read `prediction` on the very first
            # render, before the button had ever been pressed and the name
            # existed. The NameError took every section below it down with it.
            st.success(f"🔎\tPredicted PM2.5 level for {month}: **{prediction[0]:.2f}**\t")

        st.subheader("Forecast")
        st.caption(
            f"Seasonal shape learned from {model_report['rows']:,} city-months across "
            f"{model_report['cities']:,} cities. Tested on cities held out of training: "
            f"{model_report['model_mae']:.1f} µg/m³ mean absolute error, against "
            f"{model_report['flat_mae']:.1f} for assuming every month equals the city's "
            f"annual average. The 80% band covers {model_report['coverage']:.0f}% of held-out "
            f"months. This is climatology, not a weather forecast: it says what a month "
            f"typically looks like, not what next week will bring."
        )

        # Visualisation
        fig = go.Figure()

        # 2px line, round joins: thin marks, per the chart spec.
        fig.add_trace(go.Scatter(
            x=data['Month'], y=data['PM2.5'], mode='lines', name='Historical PM2.5',
            line=dict(width=2, shape='spline', smoothing=0.4),
        ))

        if real_time_pm2_5 is not None:
            fig.add_trace(go.Scatter(
                x=[current_month, current_month], y=[predicted_low, predicted_high],
                mode='lines', name='80% interval', showlegend=True,
                line=dict(color='rgba(255,75,51,0.55)', width=6),
                hovertemplate='%{y:.1f} µg/m³<extra>80% interval</extra>',
            ))
            fig.add_trace(go.Scatter(
                x=[current_month], y=[predicted_pm2_5], mode='markers+text', name='Prediction',
                marker=dict(color='red', size=10),
                text=[f"{predicted_pm2_5:.2f} µg/m³"], textposition="top center"
            ))

        # Recessive grid, no vertical rules, legend along the top so the plot
        # keeps the full width instead of losing a column to a legend box.
        fig.update_layout(
            title=dict(text=f"PM2.5 through the year in {city}", x=0, xanchor='left', y=0.96),
            xaxis_title="Month",
            yaxis_title="PM2.5 (µg/m³)",
            template="plotly_dark",
            height=420,
            margin=dict(l=0, r=12, t=76, b=44),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
            hovermode='x unified',
        )
        fig.update_xaxes(automargin=True, showgrid=False, zeroline=False,
                         tickmode='array', tickvals=list(range(1, 13)),
                         ticktext=list(month_map.keys()))
        fig.update_yaxes(automargin=True, gridcolor='rgba(255,255,255,0.08)', zeroline=False)

        st.plotly_chart(fig, use_container_width=True)
        show_city_on_map(city, latitude, longitude,
                         real_time_pm2_5=real_time_pm2_5, predicted_pm2_5=predicted_pm2_5)   
