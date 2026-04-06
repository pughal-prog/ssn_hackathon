"""
Module 2: Time-Series Analysis + Future Lake Size Prediction
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def load_timeseries(path="data/lake_timeseries.csv") -> pd.DataFrame:
    return pd.read_csv(path)

def compute_growth_rate(df: pd.DataFrame) -> float:
    """Annual average growth rate in km²/year."""
    return df['lake_area_km2'].diff().mean()

def predict_future(df: pd.DataFrame, years_ahead=3) -> pd.DataFrame:
    """
    Predict future lake area using linear regression.
    Returns a DataFrame with historical + predicted rows.
    """
    X = df['year'].values.reshape(-1, 1)
    y = df['lake_area_km2'].values

    model = LinearRegression().fit(X, y)

    last_year = int(df['year'].max())
    future_years = np.arange(last_year + 1, last_year + years_ahead + 1).reshape(-1, 1)
    future_areas = model.predict(future_years)

    future_df = pd.DataFrame({
        'year': future_years.flatten(),
        'lake_area_km2': future_areas,
        'predicted': True
    })
    df['predicted'] = False
    return pd.concat([df, future_df], ignore_index=True)

def months_to_critical(df: pd.DataFrame, critical_area_km2: float) -> str:
    """
    Estimate how many months until lake reaches critical size.
    Returns human-readable string.
    """
    growth_rate = compute_growth_rate(df)  # km²/year
    current_area = df['lake_area_km2'].iloc[-1]

    if growth_rate <= 0 or current_area >= critical_area_km2:
        return "Already critical or not growing"

    years_needed = (critical_area_km2 - current_area) / growth_rate
    months = int(years_needed * 12)
    return f"~{months} months ({years_needed:.1f} years)"

if __name__ == "__main__":
    df = load_timeseries()
    print("Growth rate:", compute_growth_rate(df), "km²/year")
    full = predict_future(df)
    print(full)
    print("Time to critical (500 km²):", months_to_critical(df, 500))
