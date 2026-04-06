"""
Module 1: Lake Detection using NDWI on Google Earth Engine
"""
import ee
import geemap
import pandas as pd

# Himalayas bounding box
HIMALAYAS = ee.Geometry.Rectangle([73.0, 26.0, 97.0, 37.0])

def initialize_gee():
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

def get_ndwi_mask(image):
    """Compute NDWI and return water binary mask."""
    ndwi = image.normalizedDifference(['B3', 'B8'])  # Green - NIR
    return ndwi.gt(0.2).rename('water')

def get_lake_area_km2(year: int) -> float:
    """Return total glacial lake area (km²) for a given year."""
    start = f"{year}-05-01"
    end   = f"{year}-09-30"

    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(HIMALAYAS)
        .filterDate(start, end)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .median()
    )

    water_mask = get_ndwi_mask(collection)

    # Pixel area in km²
    area = water_mask.multiply(ee.Image.pixelArea()).divide(1e6)
    total = area.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=HIMALAYAS,
        scale=100,
        maxPixels=1e10
    )
    return total.getInfo()['water']

def build_timeseries(start_year=2018, end_year=2024) -> pd.DataFrame:
    """Build year-wise lake area time series."""
    records = []
    for yr in range(start_year, end_year + 1):
        area = get_lake_area_km2(yr)
        records.append({'year': yr, 'lake_area_km2': area})
        print(f"{yr}: {area:.2f} km²")
    return pd.DataFrame(records)

def get_water_mask_image(year: int) -> ee.Image:
    """Return water mask image for a given year (for map display)."""
    start = f"{year}-05-01"
    end   = f"{year}-09-30"
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(HIMALAYAS)
        .filterDate(start, end)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .median()
    )
    return get_ndwi_mask(collection)

if __name__ == "__main__":
    initialize_gee()
    df = build_timeseries()
    df.to_csv("data/lake_timeseries.csv", index=False)
    print(df)
