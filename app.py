"""
GLOF Early Warning System — Streamlit Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import joblib, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from modules.timeseries  import predict_future, compute_growth_rate, months_to_critical
from modules.risk_model  import train, predict, MODEL_PATH
from modules.weather     import get_weather, get_weather_display
from modules.alerts      import send_sms_alert, send_bulk_alerts
from modules.simulation  import run_simulation

# ── Bootstrap model ──────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    train()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="GLOF Early Warning System", layout="wide", page_icon="🏔️")
st.title("🏔️ GLOF Early Warning System")
st.caption("Glacial Lake Outburst Flood Risk Prediction — Himalayas")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_lakes():
    return pd.read_csv("data/lakes.csv")

@st.cache_data
def load_ts():
    return pd.read_csv("data/lake_timeseries.csv")

lakes_df = load_lakes()
ts_df    = load_ts()

# ── Score all lakes ───────────────────────────────────────────────────────────
@st.cache_data
def score_all_lakes(df):
    results = []
    for _, row in df.iterrows():
        features = {k: row[k] for k in ['lake_area_km2','growth_rate','mean_slope_deg',
                                         'mean_elevation_m','temp_trend','rainfall_mm',
                                         'ice_melt_rate','moraine_stability',
                                         'seismic_activity','glacier_retreat_rate']}
        result = predict(features)
        results.append({**row.to_dict(), **result})
    return pd.DataFrame(results)

scored = score_all_lakes(lakes_df)
RISK_COLOR = {'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'green'}

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Settings
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("🌡️ Weather API")
    weather_api_key = st.text_input("OpenWeatherMap API Key", type="password",
                                     placeholder="Paste your free API key")
    if weather_api_key:
        import modules.weather as wmod
        wmod.API_KEY = weather_api_key

    st.divider()

    st.subheader("📱 SMS Alerts (Twilio)")
    twilio_sid   = st.text_input("Twilio Account SID",  type="password")
    twilio_token = st.text_input("Twilio Auth Token",   type="password")
    twilio_from  = st.text_input("Twilio Phone Number", placeholder="+15551234567")
    alert_to     = st.text_input("Alert Recipient Number", placeholder="+919876543210")

    if st.button("🚨 Send Alerts for HIGH Risk Lakes"):
        if twilio_sid and twilio_token and twilio_from and alert_to:
            import modules.alerts as amod
            amod.ACCOUNT_SID = twilio_sid
            amod.AUTH_TOKEN  = twilio_token
            amod.FROM_NUMBER = twilio_from
            results = send_bulk_alerts(scored, alert_to)
            for r in results:
                if r['status'] == 'sent':
                    st.success(f"✅ Alert sent for {r['lake']}")
                else:
                    st.error(f"❌ Failed for {r['lake']}: {r.get('error','')}")
        else:
            st.warning("Fill in all Twilio fields first.")

# ════════════════════════════════════════════════════════════════════════════
# ROW 1 — KPI Cards
# ════════════════════════════════════════════════════════════════════════════
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Lakes Monitored", len(scored))
c2.metric("🔴 HIGH Risk",   int((scored['risk'] == 'HIGH').sum()))
c3.metric("🟠 MEDIUM Risk", int((scored['risk'] == 'MEDIUM').sum()))
c4.metric("🟢 LOW Risk",    int((scored['risk'] == 'LOW').sum()))

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# ROW 2 — Map + Risk Table
# ════════════════════════════════════════════════════════════════════════════
col_map, col_table = st.columns([3, 2])

with col_map:
    st.subheader("🗺️ Lake Risk Map")
    m = folium.Map(location=[27.9, 87.5], zoom_start=6, tiles="CartoDB positron")

    for _, row in scored.iterrows():
        color = RISK_COLOR[row['risk']]
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=8 + row['lake_area_km2'] * 3,
            color=color, fill=True, fill_opacity=0.8,
            tooltip=(f"<b>{row['name']}</b><br>Risk: {row['risk']}<br>"
                     f"Area: {row['lake_area_km2']} km²<br>{row['explanation']}")
        ).add_to(m)

    st_folium(m, width=700, height=420)

with col_table:
    st.subheader("📊 Lake Risk Rankings")
    display = scored[['name','lake_area_km2','growth_rate','risk']].copy()
    display = display.sort_values('risk', key=lambda x: x.map({'HIGH':0,'MEDIUM':1,'LOW':2}))
    display.columns = ['Lake','Area (km²)','Growth (km²/yr)','Risk']

    def color_risk(val):
        return {'HIGH':'background-color:#ff4b4b;color:white',
                'MEDIUM':'background-color:#ffa500;color:white',
                'LOW':'background-color:#21c354;color:white'}.get(val,'')

    st.dataframe(display.style.map(color_risk, subset=['Risk']),
                 use_container_width=True, height=380)

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# ROW 3 — Time-Series + Custom Predictor
# ════════════════════════════════════════════════════════════════════════════
col_ts, col_pred = st.columns(2)

with col_ts:
    st.subheader("📈 Lake Area Growth (Time-Series)")
    full_ts = predict_future(ts_df.copy(), years_ahead=4)
    fig = go.Figure()
    hist = full_ts[~full_ts['predicted']]
    pred = full_ts[full_ts['predicted']]
    fig.add_trace(go.Scatter(x=hist['year'], y=hist['lake_area_km2'],
                             mode='lines+markers', name='Historical',
                             line=dict(color='royalblue', width=2)))
    fig.add_trace(go.Scatter(x=pred['year'], y=pred['lake_area_km2'],
                             mode='lines+markers', name='Predicted',
                             line=dict(color='red', dash='dash', width=2)))
    fig.update_layout(xaxis_title="Year", yaxis_title="Area (km²)",
                      height=320, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    growth = compute_growth_rate(ts_df)
    eta    = months_to_critical(ts_df, critical_area_km2=3.0)
    st.info(f"📐 Growth rate: **{growth:.3f} km²/year** | ⏱️ Time to critical: **{eta}**")

with col_pred:
    st.subheader("🤖 Predict Risk for a Custom Lake")
    with st.form("predict_form"):
        f1, f2 = st.columns(2)
        area    = f1.number_input("Lake Area (km²)",           0.1, 1000.0, 1.5)
        gr      = f2.number_input("Growth Rate (km²/yr)",      0.0, 50.0,   0.1)
        slope   = f1.number_input("Mean Slope (°)",            0.0, 90.0,  30.0)
        elev    = f2.number_input("Elevation (m)",             1000, 8000, 4500)
        temp    = f1.number_input("Temp Trend (°C/dec)",       0.0, 5.0,   1.5)
        rain    = f2.number_input("Rainfall (mm/yr)",          100, 3000,  1000)
        melt    = f1.number_input("Ice Melt Rate (m/yr)",      0.0, 30.0,  5.0)
        moraine = f2.number_input("Moraine Stability (0-1)",   0.05, 1.0,  0.7)
        seismic = f1.number_input("Seismic Activity (0-10)",   0.0, 10.0,  2.0)
        retreat = f2.number_input("Glacier Retreat (m/yr)",    0.0, 150.0, 20.0)
        submit  = st.form_submit_button("🔍 Predict Risk")

    if submit:
        result = predict({
            'lake_area_km2': area, 'growth_rate': gr, 'mean_slope_deg': slope,
            'mean_elevation_m': elev, 'temp_trend': temp, 'rainfall_mm': rain,
            'ice_melt_rate': melt, 'moraine_stability': moraine,
            'seismic_activity': seismic, 'glacier_retreat_rate': retreat
        })
        risk = result['risk']
        icon = {'HIGH':'🔴','MEDIUM':'🟠','LOW':'🟢'}[risk]
        st.markdown(f"### {icon} Risk Level: **{risk}**")
        if risk == 'HIGH':
            st.error("⚠️ WARNING: High GLOF risk! Immediate monitoring required.")
        elif risk == 'MEDIUM':
            st.warning("⚠️ CAUTION: Moderate risk. Increased monitoring recommended.")
        else:
            st.success("✅ Low risk. Continue routine monitoring.")
        st.caption(f"🧠 {result['explanation']}")

        proba_df = pd.DataFrame(result['probabilities'].items(), columns=['Risk','Probability'])
        fig2 = px.bar(proba_df, x='Risk', y='Probability', color='Risk',
                      color_discrete_map={'HIGH':'red','MEDIUM':'orange','LOW':'green'}, height=200)
        fig2.update_layout(showlegend=False, margin=dict(t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# ROW 4 — Live Weather + Feature Importance
# ════════════════════════════════════════════════════════════════════════════
col_weather, col_fi = st.columns(2)

with col_weather:
    st.subheader("🌡️ Live Weather at Lake Location")
    selected_lake = st.selectbox("Select a lake", scored['name'].tolist())
    lake_row = scored[scored['name'] == selected_lake].iloc[0]

    if st.button("🔄 Fetch Live Weather"):
        weather = get_weather(lake_row['lat'], lake_row['lon'])
        if weather['temperature_c'] is not None:
            w1, w2 = st.columns(2)
            w1.metric("🌡️ Temperature", f"{weather['temperature_c']}°C")
            w2.metric("💧 Humidity",    f"{weather['humidity_pct']}%")
            w3, w4 = st.columns(2)
            w3.metric("🌧️ Rainfall",   f"{weather['rainfall_mm']:.1f} mm/mo")
            w4.metric("💨 Wind Speed",  f"{weather['wind_speed_ms']} m/s")
            st.caption(f"Conditions: {weather['description']}")
        else:
            st.warning("⚠️ Add your OpenWeatherMap API key in the sidebar to fetch live weather.")

with col_fi:
    st.subheader("🧠 Feature Importance (Explainable AI)")
    model = joblib.load(MODEL_PATH)
    feat_names = ['Lake Area','Growth Rate','Slope','Elevation','Temp Trend','Rainfall','Ice Melt','Moraine Stability','Seismic Activity','Glacier Retreat']
    clf = model.named_steps['clf'] if hasattr(model, 'named_steps') else model
    fi_df = pd.DataFrame({'Feature':feat_names, 'Importance':clf.feature_importances_})
    fi_df = fi_df.sort_values('Importance', ascending=True)
    fig3 = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                  color='Importance', color_continuous_scale='Reds', height=280)
    fig3.update_layout(margin=dict(t=10), coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# ROW 5 — Flood Path Simulation
# ════════════════════════════════════════════════════════════════════════════
st.subheader("🌊 GLOF Flood Path Simulation")
st.caption("Simulates the downstream flood path and identifies affected villages")

sim_col1, sim_col2 = st.columns([3, 2])

with sim_col1:
    sim_lake = st.selectbox("Select lake to simulate", scored['name'].tolist(), key="sim_lake")
    sim_row  = scored[scored['name'] == sim_lake].iloc[0]

    if st.button("▶️ Run Flood Simulation"):
        with st.spinner("Simulating flood path..."):
            sim_result = run_simulation(sim_row['lat'], sim_row['lon'])
            flood_path = sim_result['flood_path']
            villages   = sim_result['affected_villages']

        # Build simulation map
        sim_map = folium.Map(location=[sim_row['lat'], sim_row['lon']],
                             zoom_start=7, tiles="CartoDB positron")

        # Lake marker
        folium.Marker(
            location=[sim_row['lat'], sim_row['lon']],
            tooltip=f"🏔️ {sim_lake} (Source)",
            icon=folium.Icon(color='blue', icon='tint', prefix='fa')
        ).add_to(sim_map)

        # Flood path line
        folium.PolyLine(
            locations=flood_path, color='blue',
            weight=4, opacity=0.7, tooltip="Flood Path"
        ).add_to(sim_map)

        # Flood path points
        for i, point in enumerate(flood_path[1:], 1):
            folium.CircleMarker(
                location=point, radius=4,
                color='cyan', fill=True, fill_opacity=0.6,
                tooltip=f"Flow point {i}"
            ).add_to(sim_map)

        # Affected villages
        for _, v in villages.iterrows():
            folium.Marker(
                location=[v['lat'], v['lon']],
                tooltip=f"⚠️ {v['village']} ({v['distance_km']} km from lake)",
                icon=folium.Icon(color='red', icon='home', prefix='fa')
            ).add_to(sim_map)

        st_folium(sim_map, width=700, height=420)

with sim_col2:
    st.markdown("#### 📋 Simulation Results")
    if st.button("▶️ Run Simulation (Results)", key="sim_results"):
        sim_result = run_simulation(sim_row['lat'], sim_row['lon'])
        villages   = sim_result['affected_villages']

        risk_level = scored[scored['name'] == sim_lake]['risk'].values[0]
        icon = {'HIGH':'🔴','MEDIUM':'🟠','LOW':'🟢'}[risk_level]
        st.markdown(f"**Lake Risk:** {icon} {risk_level}")
        st.markdown(f"**Flood Path Length:** ~{len(sim_result['flood_path'])} waypoints")

        if not villages.empty:
            st.markdown(f"**⚠️ {len(villages)} villages at risk:**")
            st.dataframe(villages[['village','distance_km']].rename(
                columns={'village':'Village','distance_km':'Distance (km)'}),
                use_container_width=True, height=300)
        else:
            st.success("✅ No villages detected in flood path.")
    else:
        st.info("Click 'Run Flood Simulation' on the left to see results here.")

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# ROW 6 — Top 5 Dangerous Lakes
# ════════════════════════════════════════════════════════════════════════════
st.subheader("🏆 Top 5 Most Dangerous Lakes")
top5 = scored.sort_values('risk', key=lambda x: x.map({'HIGH':0,'MEDIUM':1,'LOW':2})).head(5)
for i, (_, row) in enumerate(top5.iterrows(), 1):
    c = RISK_COLOR[row['risk']]
    st.markdown(
        f"**{i}. {row['name']}** — "
        f":{c}[{row['risk']}] | "
        f"Area: {row['lake_area_km2']} km² | "
        f"Growth: {row['growth_rate']} km²/yr | "
        f"_{row['explanation']}_"
    )

st.caption("🛰️ Data: Sentinel-2 (GEE) | 🤖 Model: Random Forest | Built for Hackathon 2025")
