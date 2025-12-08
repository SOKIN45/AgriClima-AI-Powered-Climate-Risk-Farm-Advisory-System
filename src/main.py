import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# ---------------------------
# App config
# ---------------------------
st.set_page_config(
    page_title="AgriClima – Lumbia & El Salvador",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 AgriClima – Lumbia & El Salvador")
st.caption("Live weather • Real PAGASA climate • Hybrid ML predictions • Farmer advice")
st.write("---")

# ---------------------------
# Constants & file paths
# ---------------------------
DATA_PATHS = [
    "data/pagasa_monthly.csv",
    "/mnt/data/pagasa_synthetic_monthly.csv"
]

MODEL_PATH = "models/rainfall_model.joblib"

# Ensure folders exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Stations & coords (approx)
LOCATIONS = {
    "Lumbia": (8.4140, 124.6110),
    "El Salvador": (8.5678, 124.5750),
    "Laguindingan": (8.5350, 124.3860),
    "Cagayan de Oro City": (8.4542, 124.6319)
}

# ---------------------------
# Load dataset
# ---------------------------
@st.cache_data
def load_historical_dataset():
    """
    Try to load a real dataset from data/pagasa_monthly.csv.
    If not found, try fallback synthetic at /mnt/data/pagasa_synthetic_monthly.csv.
    Expected minimal columns: Year, Month, Rainfall_mm, Station
    """
    for p in DATA_PATHS:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                cols = [c.lower() for c in df.columns]

                # Validate required columns
                if (
                    'year' in cols and
                    ('month' in cols or 'month_num' in cols) and
                    ('rain' in "".join(cols) or 'rainfall' in "".join(cols))
                ):
                    # Standardize names
                    rename_map = {}
                    for c in df.columns:
                        if c.lower() == 'month_num':
                            rename_map[c] = 'Month'
                        if c.lower() == 'month':
                            rename_map[c] = 'Month'
                        if c.lower() == 'year':
                            rename_map[c] = 'Year'
                        if 'rain' in c.lower() and 'rainfall' not in df.columns:
                            rename_map[c] = 'Rainfall_mm'
                        if c.lower() == 'rainfall_mm':
                            rename_map[c] = 'Rainfall_mm'
                        if c.lower() == 'station':
                            rename_map[c] = 'Station'

                    df = df.rename(columns=rename_map)

                    # Convert month name → number
                    if df['Month'].dtype == object:
                        def month_to_num(x):
                            try:
                                return int(x)
                            except:
                                try:
                                    return datetime.strptime(x[:3], "%b").month
                                except:
                                    return np.nan

                        df['Month_Num'] = df['Month'].apply(month_to_num)
                    else:
                        df['Month_Num'] = df['Month'].astype(int)

                    # Ensure rainfall column exists
                    if 'Rainfall_mm' not in df.columns:
                        for c in df.columns:
                            if 'rain' in c.lower():
                                df['Rainfall_mm'] = df[c]
                                break

                    # Ensure station column exists
                    if 'Station' not in df.columns:
                        df['Station'] = os.path.splitext(os.path.basename(p))[0]

                    df_standard = df[['Year', 'Month_Num', 'Rainfall_mm', 'Station']].copy()
                    df_standard = df_standard.dropna(subset=['Year', 'Month_Num', 'Rainfall_mm'])

                    df_standard['Month_Num'] = df_standard['Month_Num'].astype(int)
                    df_standard['Year'] = df_standard['Year'].astype(int)

                    return df_standard

            except Exception as e:
                st.warning(f"Could not parse dataset {p}: {e}")
                continue

    base_by_station = {
        "Lumbia": [98.9, 68.0, 49.8, 52.6, 125.0, 212.7, 245.6, 195.8, 219.7, 185.9, 136.0, 113.2],
        "El Salvador": [98.9, 68.0, 49.8, 52.6, 125.0, 212.7, 245.6, 195.8, 219.7, 185.9, 136.0, 113.2],
        "Laguindingan": [97.6, 85.3, 57.6, 62.1, 128.9, 220.1, 247.3, 197.4, 220.8, 191.6, 127.1, 137.5],
        "Cagayan de Oro City": [97.6, 85.3, 57.6, 62.1, 128.9, 220.1, 247.3, 197.4, 220.8, 191.6, 127.1, 137.5]
    }

    stations = list(base_by_station.keys())
    rows = []

    for s in stations:
        for y in range(1991, 2021):
            for m_idx, base in enumerate(base_by_station[s], start=1):
                noise = np.random.normal(0, base * 0.25)
                val = max(0.0, base + noise)
                rows.append({
                    "Year": y,
                    "Month_Num": m_idx,
                    "Rainfall_mm": round(val, 1),
                    "Station": s
                })

    return pd.DataFrame(rows)


hist_df = load_historical_dataset()

# ---------------------------
# Monthly averages per station
# ---------------------------
@st.cache_data
def compute_monthly_averages(df):
    df = df.copy()
    df['Month'] = df['Month_Num']
    monthly = (
        df.groupby(['Station', 'Month'])
        .agg({'Rainfall_mm': 'mean'})
        .reset_index()
        .rename(columns={'Rainfall_mm': 'Hist_Mean_Rain_mm'})
    )
    return monthly


monthly_avg = compute_monthly_averages(hist_df)

# ---------------------------
# Real-time weather fetch
# ---------------------------
@st.cache_data(ttl=300)
def fetch_open_meteo(lat, lon, past_days=7, forecast_days=7):
    try:
        base = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "timezone": "Asia/Manila",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "forecast_days": forecast_days,
            "past_days": past_days,
            "hourly": "precipitation",
        }

        r = requests.get(base, params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    except Exception:
        return None


def summarize_precipitation(open_meteo_json):
    recent_total = None
    forecast_total = None
    current_precip = 0.0

    try:
        if open_meteo_json is None:
            return None, None, None

        daily = open_meteo_json.get("daily", {})
        if 'precipitation_sum' in daily:
            vals = daily['precipitation_sum']
            try:
                forecast_total = sum(vals[:7])
            except:
                forecast_total = sum(vals)

        hourly = open_meteo_json.get("hourly", {})
        if 'precipitation' in hourly:
            hrs = hourly['precipitation']
            if len(hrs) >= 24 * 7:
                recent_total = sum(hrs[-24 * 7:])
            else:
                recent_total = sum(hrs)

        if 'hourly' in open_meteo_json:
            if 'precipitation' in open_meteo_json['hourly']:
                if len(open_meteo_json['hourly']['precipitation']) > 0:
                    current_precip = open_meteo_json['hourly']['precipitation'][-1]

    except Exception:
        pass

    return recent_total, forecast_total, current_precip

# ---------------------------
# ML Model Handling
# ---------------------------
def train_and_save_model(df, save_path=MODEL_PATH):

    X = df[['Year', 'Month_Num', 'Station']].copy()
    X['month_sin'] = np.sin(2 * np.pi * X['Month_Num'] / 12)
    X['month_cos'] = np.cos(2 * np.pi * X['Month_Num'] / 12)

    y = df['Rainfall_mm']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cat_cols = ['Station']
    num_cols = ['Year', 'Month_Num', 'month_sin', 'month_cos']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('num', StandardScaler(), num_cols)
        ]
    )

    pipeline = Pipeline([
        ('pre', preprocessor),
        ('rf', RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, save_path)

    train_r2 = pipeline.score(X_train, y_train)
    test_r2 = pipeline.score(X_test, y_test)

    return pipeline, train_r2, test_r2


@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        try:
            mdl = joblib.load(MODEL_PATH)
            return mdl, None, None
        except Exception as e:
            st.warning(f"Could not load model at {MODEL_PATH}: {e}")

    mdl, tr, te = train_and_save_model(hist_df, save_path=MODEL_PATH)
    return mdl, tr, te

# ---------------------------
# Top Navigation
# ---------------------------
selected_location = st.selectbox(
    "📍 Select Location",
    list(LOCATIONS.keys()),
    key="location_select"
)

tabs = st.tabs(["🌍 Live Weather", "📊 Climate Dashboard", "🤖 ML Prediction", "ℹ️ About"])

# ---------------------------
# TAB 0: Live Weather
# ---------------------------
with tabs[0]:
    st.header("🌍 Live Weather")

    lat, lon = LOCATIONS[selected_location]
    st.write(f"Selected location: **{selected_location}** — Lat: {lat}, Lon: {lon}")

    weather_json = fetch_open_meteo(lat, lon, past_days=7, forecast_days=7)

    if weather_json is None:
        st.error("Unable to fetch real-time weather.")
    else:
        current = weather_json.get("current_weather", {})
        current_temp = current.get('temperature')
        current_windspeed = current.get('windspeed')

        st.subheader("Current")
        c1, c2, c3 = st.columns(3)
        c1.metric("🌡 Temperature (°C)", current_temp)
        c2.metric("💨 Wind speed (m/s)", current_windspeed)

        recent_total, forecast_total, current_precip = summarize_precipitation(weather_json)
        c3.metric("🌧 Recent 7-day rain (mm)", f"{recent_total:.1f}" if recent_total else "N/A")

        st.subheader("7-day Forecast")
        daily = weather_json.get("daily", {})
        if daily and "time" in daily:
            df_fc = pd.DataFrame({
                "Date": pd.to_datetime(daily['time']),
                "Max Temp": daily['temperature_2m_max'],
                "Min Temp": daily['temperature_2m_min'],
                "Rain (mm)": daily['precipitation_sum']
            })
            st.dataframe(df_fc, use_container_width=True)
        else:
            st.info("No daily forecast available.")

# ---------------------------
# TAB 1: Climate Dashboard
# ---------------------------
with tabs[1]:
    st.header("📊 Climate Dashboard (PAGASA Historical Monthly Averages)")

    station = st.selectbox("Select Station", sorted(hist_df["Station"].unique()))

    station_monthly = monthly_avg[monthly_avg['Station'] == station].copy()
    station_monthly = station_monthly.sort_values("Month")

    month_names = {i: datetime(2000, i, 1).strftime("%b").upper() for i in range(1, 13)}
    sel_month = st.selectbox("Select Month", list(range(1, 13)),
                             format_func=lambda x: month_names[x])

    hist_row = station_monthly[station_monthly['Month'] == sel_month]

    if hist_row.empty:
        st.warning("No historical data for this month.")
        hist_mean = None
    else:
        hist_mean = float(hist_row['Hist_Mean_Rain_mm'].values[0])

    col1, col2, col3 = st.columns(3)
    col1.metric("📍 Station", station)
    col2.metric("📅 Month", month_names[sel_month])
    col3.metric("📈 Mean Rainfall (mm)", f"{hist_mean:.1f}" if hist_mean else "N/A")

    st.markdown("### 🌦 Monthly Historical Rainfall (Farmer-Friendly View)")

    df_display = station_monthly.copy()
    df_display['Month'] = df_display['Month'].map(month_names)

    def classify(mm):
        if mm > 200:
            return "⚠ Heavy rain – avoid lowland planting"
        elif mm > 80:
            return "🌱 Moderate – good for planting"
        else:
            return "☀ Low rain – irrigation needed"

    df_display["Meaning"] = df_display["Hist_Mean_Rain_mm"].apply(classify)

    st.table(df_display[['Month', 'Hist_Mean_Rain_mm', 'Meaning']])

    st.subheader("Farming advice")

    if hist_mean is None:
        st.info("No advice available.")
    else:
        if hist_mean > 200:
            st.warning("⚠ Heavy rainfall — ensure drainage, delay planting low areas.")
        elif hist_mean < 80:
            st.info("💧 Dry month — irrigation recommended.")
        else:
            st.success("🌱 Good month for planting.")

# ---------------------------
# TAB 2: ML Prediction
# ---------------------------
with tabs[2]:
    st.header("🤖 ML Prediction — Hybrid Real-Time + PAGASA")

    col_loc, col_year, col_month = st.columns([2, 1, 2])
    with col_loc:
        location = st.selectbox("📍 Location", list(LOCATIONS.keys()))
    with col_year:
        year = st.number_input("Year", min_value=2020, max_value=2035,
                               value=datetime.now().year, step=1)
    with col_month:
        month_num = st.selectbox("Month", list(range(1, 13)),
                                 index=datetime.now().month - 1,
                                 format_func=lambda x: datetime(2000, x, 1).strftime("%B"))

    st.write(f"Predicting **{location} — {datetime(year, month_num, 1).strftime('%B %Y')}**")
    st.divider()

    model, tr, te = load_or_train_model()

    if tr is not None:
        st.info(f"Trained new model: Train R² = {tr:.3f}, Test R² = {te:.3f}")
    else:
        st.success("Loaded trained model.")

    lat, lon = LOCATIONS[location]
    realtime_json = fetch_open_meteo(lat, lon, past_days=7, forecast_days=7)
    recent_total, forecast_total, current_precip = summarize_precipitation(realtime_json)

    hist_val_row = monthly_avg[
        (monthly_avg['Station'] == location) &
        (monthly_avg['Month'] == month_num)
    ]

    hist_mean = float(hist_val_row['Hist_Mean_Rain_mm'].values[0]) if not hist_val_row.empty else None

    if st.button("🔮 Predict (Hybrid)"):
        if model is None:
            st.error("Model not available.")
        else:
            X_pred = pd.DataFrame([{
                'Year': int(year),
                'Month_Num': int(month_num),
                'Station': location,
                'month_sin': np.sin(2 * np.pi * month_num / 12),
                'month_cos': np.cos(2 * np.pi * month_num / 12)
            }])

            try:
                base_pred = model.predict(X_pred)[0]
            except:
                base_pred = float(model.predict(X_pred.values.reshape(1, -1))[0])

            realtime_component = None
            if recent_total is not None:
                realtime_component = recent_total * (30 / 7)
            elif forecast_total is not None:
                realtime_component = forecast_total * (30 / 7)
            elif current_precip is not None:
                realtime_component = current_precip * 30

            if realtime_component is not None:
                w_realtime = 0.35
                w_model = 0.65
                final_pred = w_model * base_pred + w_realtime * realtime_component
            else:
                final_pred = base_pred

            final_pred = max(0.0, float(final_pred))

            st.metric("🌧 Predicted Monthly Rainfall", f"{final_pred:.1f} mm")
            st.write(f"ML model baseline: **{base_pred:.1f} mm**")
            if realtime_component:
                st.write(f"Realtime estimate: **{realtime_component:.1f} mm**")

            if final_pred > 250:
                st.error("⚠ Very heavy rainfall — flood risk.")
            elif final_pred > 150:
                st.warning("⚠ Heavy rainfall — prepare drainage.")
            elif final_pred > 80:
                st.info("🌱 Moderate rainfall — good for crops.")
            else:
                st.success("☀ Low rainfall — irrigation needed.")

# ---------------------------
# TAB 3: About
# ---------------------------
with tabs[3]:
    st.header("ℹ️ About This Project")
    st.markdown("""
### 🌾 AgriClima - AI-Powered Climate Risk Farm Advisory System

This tool helps farmers in **Lumbia–El Salvador** by providing:

- 📊 Monthly climate data (PAGASA normals)
- 🤖 ML rainfall predictions
- 🌍 Real-time weather (Open-Meteo)
- 👨‍🌾 Planting advice
- 📈 Climate analytics

### 📊 Data Source
- PAGASA Climatological Normals (1991–2020)
- Open-Meteo (real-time data)

### ⚙ Technology
- Streamlit
- Python
- Scikit-learn
- Pandas / NumPy

### 🎓 Project
- PIT Machine Learning Project
- Version 1.0.0
    """)

    st.caption("🌍 AgriClima v1.0 | Data Source: PAGASA + Open-Meteo")
