"""
Module 7: GLOF Flood Path Simulation
Simulates downstream flood path from a glacial lake using
steepest-descent approximation on a grid.
"""
import numpy as np
import pandas as pd

# Approximate downstream villages near Himalayan lakes (lat, lon, name)
VILLAGES = [
    (27.70, 86.70, "Namche Bazaar"),
    (27.60, 86.80, "Lukla"),
    (27.50, 86.90, "Phaplu"),
    (27.40, 87.00, "Salleri"),
    (27.30, 87.10, "Okhaldhunga"),
    (27.20, 87.20, "Diktel"),
    (27.80, 88.00, "Gangtok"),
    (27.60, 88.10, "Rangpo"),
    (27.50, 88.20, "Siliguri"),
    (27.40, 90.50, "Punakha"),
    (27.30, 90.40, "Wangdue"),
    (27.20, 90.30, "Tsirang"),
]

def simulate_flood_path(lake_lat: float, lake_lon: float,
                        steps: int = 12, step_size: float = 0.08) -> list:
    """
    Simulate flood path as a series of coordinates flowing downstream
    (south, following gravity approximation).
    Returns list of [lat, lon] waypoints.
    """
    path = [[lake_lat, lake_lon]]
    lat, lon = lake_lat, lake_lon

    for i in range(steps):
        # Flood flows south (decreasing lat) with slight meandering
        lat -= step_size + np.random.uniform(-0.01, 0.01)
        lon += np.random.uniform(-0.03, 0.03)
        path.append([round(lat, 4), round(lon, 4)])

    return path

def get_affected_villages(lake_lat: float, lake_lon: float,
                          flood_path: list, radius_deg: float = 0.3) -> pd.DataFrame:
    """
    Return villages within radius of the flood path.
    """
    affected = []
    for v_lat, v_lon, v_name in VILLAGES:
        for point in flood_path:
            dist = ((point[0] - v_lat)**2 + (point[1] - v_lon)**2) ** 0.5
            if dist <= radius_deg:
                # Estimate distance from lake in km (rough)
                lake_dist_km = (((v_lat - lake_lat)**2 + (v_lon - lake_lon)**2) ** 0.5) * 111
                affected.append({
                    "village":      v_name,
                    "lat":          v_lat,
                    "lon":          v_lon,
                    "distance_km":  round(lake_dist_km, 1)
                })
                break  # avoid duplicates

    df = pd.DataFrame(affected)
    if not df.empty:
        df = df.sort_values("distance_km").reset_index(drop=True)
    return df

def run_simulation(lake_lat: float, lake_lon: float) -> dict:
    """Full simulation — returns path + affected villages."""
    path     = simulate_flood_path(lake_lat, lake_lon)
    villages = get_affected_villages(lake_lat, lake_lon, path)
    return {"flood_path": path, "affected_villages": villages}
