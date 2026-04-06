"""
Module 3: Terrain Analysis (Slope, Elevation, Flow Direction)
Uses richdem for DEM processing.
"""
import numpy as np
import richdem as rd

def load_dem(path: str) -> rd.rdarray:
    return rd.LoadGDAL(path)

def get_slope(dem: rd.rdarray) -> np.ndarray:
    slope = rd.TerrainAttribute(dem, attrib='slope_degrees')
    return np.array(slope)

def get_flow_accumulation(dem: rd.rdarray) -> np.ndarray:
    filled = rd.FillDepressions(dem)
    accum  = rd.FlowAccumulation(filled, method='D8')
    return np.array(accum)

def summarize_terrain(dem_path: str) -> dict:
    """Return mean slope and elevation for a DEM file."""
    dem   = load_dem(dem_path)
    slope = get_slope(dem)
    elev  = np.array(dem)
    elev_masked = elev[elev > -9999]  # remove nodata

    return {
        "mean_elevation_m": float(np.mean(elev_masked)),
        "mean_slope_deg":   float(np.mean(slope[slope > 0])),
        "max_slope_deg":    float(np.max(slope))
    }

if __name__ == "__main__":
    # Example usage — replace with actual DEM path
    stats = summarize_terrain("data/himalayas_dem.tif")
    print(stats)
