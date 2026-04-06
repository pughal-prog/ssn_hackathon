"""
Module 5: Live Weather Data via OpenWeatherMap API
"""
import requests

API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"  # Free at openweathermap.org
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

def get_weather(lat: float, lon: float) -> dict:
    """Fetch live temperature and rainfall for a lake location."""
    try:
        params = {"lat": lat, "lon": lon, "appid": API_KEY, "units": "metric"}
        res = requests.get(BASE_URL, params=params, timeout=5)
        data = res.json()

        return {
            "temperature_c":  data["main"]["temp"],
            "humidity_pct":   data["main"]["humidity"],
            "rainfall_mm":    data.get("rain", {}).get("1h", 0.0) * 24 * 30,  # estimate monthly
            "description":    data["weather"][0]["description"].title(),
            "wind_speed_ms":  data["wind"]["speed"]
        }
    except Exception as e:
        # Return fallback values if API fails
        return {
            "temperature_c": None,
            "humidity_pct":  None,
            "rainfall_mm":   None,
            "description":   "Unavailable",
            "wind_speed_ms": None
        }

def get_weather_display(lat: float, lon: float) -> str:
    """Return a formatted weather string for display."""
    w = get_weather(lat, lon)
    if w["temperature_c"] is None:
        return "⚠️ Weather data unavailable (check API key)"
    return (
        f"🌡️ {w['temperature_c']}°C | "
        f"💧 Humidity: {w['humidity_pct']}% | "
        f"🌧️ Rain: {w['rainfall_mm']:.1f} mm/mo | "
        f"💨 Wind: {w['wind_speed_ms']} m/s | "
        f"{w['description']}"
    )
