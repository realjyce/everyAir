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

# Features and target
X = data[['Month', '2023']].copy()
# New Humidity and temperature feature
if 'Temperature' in data.columns and 'Humidity' in data.columns:
    X['Temperature'] = data['Temperature']
    X['Humidity'] = data['Humidity']

y = data['PM2.5']

# Splitting | Train-test

# Current Date
current_month = datetime.now().month
current_year = datetime.now().year

temperature, humidity, wind_speed, min_temp, max_temp, rainfall = fetch_additional(latitude, longitude) # fetch weather
ndvi = fetch_ndvi(latitude, longitude) # fetch NDVI based on lat/lon
SEARCH_RADIUS_KM = 50
industrial_sites = fetch_industrial_sites_near_city(latitude, longitude)
industry_lookup_failed = industrial_sites is None
industrial_sites = industrial_sites or []
dist = (km_to_nearest_site([latitude], [longitude], industrial_sites).tolist()
        if industrial_sites else [])
nearest_industrial = min(dist) if dist else float(SEARCH_RADIUS_KM)
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
    X['dist'] = nearest_industrial

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def feature_row(month_value, yearly_value=None):
    row = {c: X[c].iloc[0] for c in X.columns}
    row["Month"] = month_value
    if yearly_value is not None:
        row["2023"] = yearly_value
    return pd.DataFrame([row])[X.columns]


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
            height=340,
            template="plotly_dark",
            margin=dict(l=0, r=0, t=64, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig, use_container_width=True)

    def show_city_on_map(city, latitude, longitude, input_features, best_model_instance, real_time_pm2_5=None, predicted_pm2_5=None):
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

        if "dist" not in X.columns or not industrial_sites:
            draw_single_marker(
                "The industrial site lookup failed, so the map shows the single city "
                "prediction rather than a surface."
                if industry_lookup_failed else
                "No industrial sites mapped nearby, so nothing varies the prediction "
                "across the map. One value for the whole city."
            )
            return

        span, steps = 0.4, 28
        mesh_lat, mesh_lon = np.meshgrid(
            np.linspace(latitude - span, latitude + span, steps),
            np.linspace(longitude - span, longitude + span, steps),
        )
        flat_lat, flat_lon = mesh_lat.ravel(), mesh_lon.ravel()

        frame = pd.DataFrame([{c: X[c].iloc[0] for c in X.columns} for _ in range(len(flat_lat))])
        frame["Month"] = current_month
        frame["dist"] = km_to_nearest_site(flat_lat, flat_lon, industrial_sites)
        values = best_model_instance.predict(frame[X.columns])
        spread = float(values.max()) - float(values.min())

        if spread < 0.5:
            draw_single_marker(
                f"Distance to industry ranges {frame['dist'].min():.0f} to "
                f"{frame['dist'].max():.0f} km across this area, but the model returns the "
                f"same value throughout, so there is no surface to draw. Training uses one "
                f"row per month for a single city, which leaves distance constant and gives "
                f"the model nothing to learn from."
            )
            return

        HeatMap(
            [[float(flat_lat[i]), float(flat_lon[i]), float(values[i])] for i in range(len(values))],
            radius=30, blur=24, min_opacity=0.2,
            max_val=float(values.max()),
            gradient={0.0: "#2c7bb6", 0.4: "#abd9e9", 0.6: "#ffffbf",
                      0.8: "#fdae61", 1.0: "#d7191c"},
        ).add_to(m)
        folium_static(m, width=1120, height=520)
        st.caption(
            f"Predicted {values.min():.1f} to {values.max():.1f} µg/m³ across the area, "
            f"varying with distance to industry."
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
        input_features = feature_row(current_month)
        predicted_pm2_5 = best_model_instance.predict(input_features)[0]

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
        _city_avg = float(data['2023'].iloc[0]) if len(data) else 100.0
        pc1, pc2 = st.columns(2)
        with pc1:
            month = st.selectbox("Month", list(month_map.keys()))
        with pc2:
            yearly_avg = st.number_input(
                "Yearly average (µg/m³)", value=round(_city_avg, 1), step=1.0)
        # Prediction
        if st.button("Predict"):
            with st.spinner(text="Predicting..."):
                time.sleep(2)
            month_numeric = month_map[month]
            input_features = feature_row(month_numeric, yearly_avg)
            prediction = best_model_instance.predict(input_features)
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

        # Visualisation
        fig = go.Figure()

        # 2px line, round joins: thin marks, per the chart spec.
        fig.add_trace(go.Scatter(
            x=data['Month'], y=data['PM2.5'], mode='lines', name='Historical PM2.5',
            line=dict(width=2, shape='spline', smoothing=0.4),
        ))

        if real_time_pm2_5 is not None:
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
        show_city_on_map(city, latitude, longitude, input_features, best_model_instance,
                         real_time_pm2_5=real_time_pm2_5, predicted_pm2_5=predicted_pm2_5)   
