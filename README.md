# 🏔️ GLOF Early Warning System

> AI-powered Glacial Lake Outburst Flood risk prediction for the Himalayas

## 🚀 Quick Start

```bash
# Option 1: One-click setup (Windows)
run.bat

# Option 2: Manual
pip install -r requirements.txt
python -c "from modules.risk_model import train; train()"
streamlit run app.py
```

## 📁 Project Structure

```
glof-predictor/
├── app.py                    ← Streamlit dashboard (main entry)
├── modules/
│   ├── lake_detection.py     ← NDWI lake detection via Google Earth Engine
│   ├── timeseries.py         ← Growth analysis + future prediction
│   ├── terrain.py            ← DEM slope/elevation analysis (richdem)
│   └── risk_model.py         ← Random Forest GLOF risk classifier
├── data/
│   ├── lakes.csv             ← Lake dataset (lat/lon + features)
│   └── lake_timeseries.csv   ← Year-wise lake area data
├── models/
│   └── rf_glof.pkl           ← Trained model (auto-generated)
└── requirements.txt
```

## 🔥 Features

| Feature | Description |
|---|---|
| 🛰️ NDWI Lake Detection | Sentinel-2 via Google Earth Engine |
| 📈 Time-Series Analysis | Historical growth + future prediction |
| 🤖 Risk Prediction | Random Forest → LOW / MEDIUM / HIGH |
| 🧠 Explainable AI | Shows WHY model predicted high risk |
| 🗺️ Interactive Map | Folium map with color-coded risk markers |
| ⏱️ Timeline Prediction | "Lake reaches critical size in X months" |
| 🏆 Lake Ranking | Top 10 most dangerous lakes |

## 🛰️ Using Real GEE Data

```python
# Authenticate once
import ee
ee.Authenticate()

# Then run lake detection
from modules.lake_detection import initialize_gee, build_timeseries
initialize_gee()
df = build_timeseries(2018, 2024)
df.to_csv("data/lake_timeseries.csv", index=False)
```

## 🎯 Pitch Line

> "Unlike traditional systems, our solution not only detects glacial lakes but also
> predicts future GLOF risk using AI and time-series analysis, enabling proactive
> disaster prevention."
