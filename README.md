# everyAir
###### everyAir makes your every day air!
EveryAir” on Smart Air-Quality Prediction System: Forecasting Pollution Level. In this Python based assignment libraries like panda, matplotlib, seaborn, statsmodels, numpy, folium are used to analyse provided data in Asia_Dataset.csv, as well as forecasting future PM2.5 Air Pollution data to help with air pollution prediction in numbers of Asian countries(2000+ in dataset).

Data Preprocessing: Load data, handle missing values and insert current date. Extract viable data from the data.
Import Coordinates: Insert LATs and LONs of the specified cities, and predict the PM2.5 air pollution level accordingly.
Numerical Analysis: Month mapping, converting month strings into numerical values. Analyze PM2.5 trends and build an air pollution prediction model using Random Forest Regressor on Historical data and Real-time data.
Data Visualisation: Plotting air pollution trends and display real-time gauge meter for PM2.5 data for the relevant cities. Geospatial mapping visualisation for predicted and real-time PM2.5 level in the city.

## Screenshots

Shots of the thing actually running, because a folder of `.py` files proves nothing.

### Real-time gauge

![Real-time PM2.5 gauge for Bangkok](docs/screenshots/gauge.png)

Current reading in µg/m³, banded on US EPA PM2.5 breakpoints, with the WHO
24-hour guideline of 15 µg/m³ marked as a white line.

The bands used to be 0-50 green, 50-100 yellow and so on. Those are AQI index
numbers, and the gauge was showing µg/m³. Two different scales on one dial: it
coloured 40 µg/m³ green when that is already unhealthy for anyone with asthma.

The live value comes from OpenWeather, which serves CAMS model output rather
than a ground station. Worth knowing before comparing it to a local monitor.

### Seasonal model

![PM2.5 through the year](docs/screenshots/forecast.png)

Blue line is the twelve month history for the selected city. Red dot is the
model's estimate for the current month, and the prediction panel gives an 80%
interval taken from the spread across the forest's trees.

The model predicts the *shape* of a city's year, not its level. It learns a
monthly multiplier from 25,510 city-months across 2,106 cities, then scales that
by the city's own typical level. Tested on whole cities held out of training it
lands at 4.7 µg/m³ mean absolute error, against 9.5 for assuming every month
equals the annual average. Half the error, on cities it has never seen.

The interval is real rather than a proxy: separate gradient boosters fit the
10th and 90th percentiles, and across held-out months that band contains the
true value 78% of the time against a nominal 80%.

Things that were tried and did not survive measurement. Population density made
it very slightly worse, so it is not a feature. A plain Random Forest on the raw
ratio scored 5.1. What helped was per-country monthly climatology as a feature,
a log target so multiplicative errors are symmetric, and boosting instead of
bagging.

Three things it does not do, said plainly.

It is climatology, not weather. It answers "what does March usually look like
here", not "what will next Tuesday bring". There is no lead time and no initial
condition anywhere in it.

It used to train on twelve rows: one city, one row per month, then a random 20%
split, which trains on December to predict June. The annual mean was also an
input feature, and that is the mean of the twelve values it was predicting.

Static city features turned out to be weak. Pooling cities and predicting level
directly from population density scored 21.9 MAE, worse than plain monthly
climatology at 17.6. Population describes emission potential, not what the air is
doing. That result is why the model predicts shape and takes level as given.

Missing, in rough order of how much they would help: mixing height, which is the
strongest single control on surface concentration, precipitation for wet
deposition, wind vector for ventilation and transport, and upwind active fire
counts. In Southeast Asia the February to April peak is largely agricultural
burning, much of it across a border.

### Weather and urban features

Eight tiles, each with an icon, a label, the number, and what the number is in.
Population density, street density, NDVI greenness and distance to the nearest
industrial site, then temperature, wind, humidity and rainfall.

These are context, not model inputs. The seasonal model runs on month and city
level alone, because pooling tested worse than climatology when it tried to
predict level from these. NDVI is also on a 1 degree grid, roughly 111 km, so it
describes a region rather than a city.

`50+ km` on the industry tile is a real answer rather than a failure. Overpass
gets asked for industrial landuse inside a 50 km box and sometimes there is
genuinely nothing in it, so the tile says so instead of the page falling over.

These are hand rolled rather than `st.metric`. Streamlit's own metric truncates
long labels, and its third argument is a delta, so passing `"people/km2"` there
puts a green up arrow beside the number as though population density had just
improved.

### The map

![Predicted PM2.5 marker over Bangkok](docs/screenshots/map.png)

One marker on the city, sized and coloured by the predicted value, because that
is all the model can honestly support.

It started as a solid square. The grid was feeding (lat, lon) pairs into a
model trained on [month, yearly average], so the values coming back were not
predictions of anything, and since latitude and longitude barely move across a
0.2 degree box they were all near identical.

The proper fix was to feed the model something that genuinely varies per point.
Distance to the nearest industrial site does, so the map now builds a real
grid, computes that distance for every cell, and asks the model. Then it checks
whether the answers actually differ. They do not: training is one row per month
for a single city, which leaves distance constant across all twelve rows, so
the model never learns anything from it and returns the same number everywhere.

So the app draws the marker and says why, rather than shading a flat field that
implies detail it does not have. If the surface ever does vary, the heat layer
appears on its own.