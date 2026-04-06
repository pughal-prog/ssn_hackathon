"""
Module 4: GLOF Risk Prediction Model (Random Forest)
Input features: lake_area, growth_rate, slope, elevation, temp_trend, rainfall
Output: LOW / MEDIUM / HIGH risk
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

FEATURE_COLS = ['lake_area_km2', 'growth_rate', 'mean_slope_deg',
                'mean_elevation_m', 'temp_trend', 'rainfall_mm']
LABEL_COL    = 'risk_label'
MODEL_PATH   = 'models/rf_glof.pkl'

def generate_synthetic_data(n=500) -> pd.DataFrame:
    """
    Generate synthetic training data for demo/hackathon.
    Replace with real labeled data when available.
    """
    np.random.seed(42)
    df = pd.DataFrame({
        'lake_area_km2':   np.random.uniform(10, 600, n),
        'growth_rate':     np.random.uniform(0, 30, n),
        'mean_slope_deg':  np.random.uniform(5, 60, n),
        'mean_elevation_m':np.random.uniform(3000, 6000, n),
        'temp_trend':      np.random.uniform(0, 3, n),
        'rainfall_mm':     np.random.uniform(200, 2000, n),
    })

    # Rule-based labels (proxy for real labels)
    def label(row):
        score = 0
        if row['lake_area_km2'] > 300:   score += 2
        if row['growth_rate'] > 15:       score += 2
        if row['mean_slope_deg'] > 35:    score += 1
        if row['temp_trend'] > 1.5:       score += 1
        if row['rainfall_mm'] > 1200:     score += 1
        if score >= 4: return 'HIGH'
        if score >= 2: return 'MEDIUM'
        return 'LOW'

    df[LABEL_COL] = df.apply(label, axis=1)
    return df

def train(df: pd.DataFrame = None):
    if df is None:
        df = generate_synthetic_data()

    X = df[FEATURE_COLS]
    y = df[LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print(classification_report(y_test, model.predict(X_test)))
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")
    return model

def predict(features: dict) -> dict:
    """
    Predict risk for a single lake.
    features: dict with keys matching FEATURE_COLS
    Returns: {'risk': 'HIGH'|'MEDIUM'|'LOW', 'probabilities': {...}, 'explanation': str}
    """
    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([features])[FEATURE_COLS]

    risk  = model.predict(X)[0]
    proba = dict(zip(model.classes_, model.predict_proba(X)[0]))

    # Explainable AI: top contributing features
    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    top_factors = sorted(importances, key=importances.get, reverse=True)[:3]
    explanation = f"Risk driven by: {', '.join(top_factors)}"

    return {'risk': risk, 'probabilities': proba, 'explanation': explanation}

if __name__ == "__main__":
    train()
    sample = {
        'lake_area_km2': 450, 'growth_rate': 20, 'mean_slope_deg': 40,
        'mean_elevation_m': 4500, 'temp_trend': 2.0, 'rainfall_mm': 1500
    }
    print(predict(sample))
