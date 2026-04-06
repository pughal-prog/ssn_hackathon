"""
Module 4: GLOF Risk Prediction Model
- 5000 data points
- 10 features
- Complex non-linear patterns
- Gradient Boosting Classifier
Output: LOW / MEDIUM / HIGH risk
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble         import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import classification_report
from sklearn.pipeline         import Pipeline

FEATURE_COLS = [
    'lake_area_km2',        # Lake surface area
    'growth_rate',          # Annual area growth
    'mean_slope_deg',       # Terrain slope
    'mean_elevation_m',     # Altitude
    'temp_trend',           # Temperature rise per decade
    'rainfall_mm',          # Annual rainfall
    'ice_melt_rate',        # Glacier ice melt rate (m/year)
    'moraine_stability',    # Dam moraine stability index (0-1, lower = unstable)
    'seismic_activity',     # Earthquake frequency index (0-10)
    'glacier_retreat_rate', # Glacier retreat speed (m/year)
]
LABEL_COL  = 'risk_label'
MODEL_PATH = 'models/rf_glof.pkl'

def generate_synthetic_data(n=5000) -> pd.DataFrame:
    """
    Generate complex synthetic training data with
    non-linear interactions between features.
    """
    np.random.seed(42)

    df = pd.DataFrame({
        'lake_area_km2':        np.random.exponential(scale=150, size=n).clip(10, 800),
        'growth_rate':          np.random.gamma(shape=2, scale=8, size=n).clip(0, 50),
        'mean_slope_deg':       np.random.normal(loc=30, scale=12, size=n).clip(5, 75),
        'mean_elevation_m':     np.random.normal(loc=4500, scale=800, size=n).clip(2000, 7000),
        'temp_trend':           np.random.gamma(shape=1.5, scale=1, size=n).clip(0, 5),
        'rainfall_mm':          np.random.normal(loc=1000, scale=400, size=n).clip(100, 3000),
        'ice_melt_rate':        np.random.gamma(shape=2, scale=3, size=n).clip(0, 30),
        'moraine_stability':    np.random.beta(a=3, b=2, size=n).clip(0.05, 1.0),
        'seismic_activity':     np.random.exponential(scale=2, size=n).clip(0, 10),
        'glacier_retreat_rate': np.random.gamma(shape=2, scale=15, size=n).clip(0, 150),
    })

    def score_row(row):
        score = 0.0

        # Large + fast growing lake = exponential risk
        score += (row['lake_area_km2'] / 100) ** 1.5 * 0.4

        # Growth rate amplified by ice melt
        score += row['growth_rate'] * row['ice_melt_rate'] * 0.02

        # Unstable moraine is critical (inverse relationship)
        score += (1 - row['moraine_stability']) * 4.0

        # Steep slope + high seismic = dangerous combo
        score += (row['mean_slope_deg'] / 30) * (row['seismic_activity'] / 5) * 1.5

        # Temperature drives glacier retreat
        score += row['temp_trend'] * row['glacier_retreat_rate'] * 0.01

        # High rainfall on steep terrain
        score += (row['rainfall_mm'] / 500) * (row['mean_slope_deg'] / 30) * 0.5

        # High elevation = more glacial mass
        score += (row['mean_elevation_m'] / 5000) * 0.8

        # Add noise for realism
        score += np.random.normal(0, 0.3)

        return score

    df['_score'] = df.apply(score_row, axis=1)

    # Use percentile-based thresholds for balanced classes (~33% each)
    high_thresh   = df['_score'].quantile(0.60)
    low_thresh    = df['_score'].quantile(0.30)

    def label(score):
        if score >= high_thresh: return 'HIGH'
        if score >= low_thresh:  return 'MEDIUM'
        return 'LOW'

    df[LABEL_COL] = df['_score'].apply(label)
    df.drop(columns=['_score'], inplace=True)
    return df

def train(df: pd.DataFrame = None):
    if df is None:
        df = generate_synthetic_data(5000)

    print(f"Dataset size: {len(df)}")
    print(f"Class distribution:\n{df[LABEL_COL].value_counts()}\n")

    X = df[FEATURE_COLS]
    y = df[LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline: scaler + Gradient Boosting with class balance
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=10,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Cross-validation score
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved -> {MODEL_PATH}")
    return model

def predict(features: dict) -> dict:
    """
    Predict risk for a single lake.
    Missing new features default to safe values.
    """
    model = joblib.load(MODEL_PATH)

    # Default values for optional new features
    defaults = {
        'ice_melt_rate':        5.0,
        'moraine_stability':    0.7,
        'seismic_activity':     2.0,
        'glacier_retreat_rate': 20.0,
    }
    for k, v in defaults.items():
        features.setdefault(k, v)

    X = pd.DataFrame([features])[FEATURE_COLS]

    risk  = model.predict(X)[0]
    proba = dict(zip(model.classes_, model.predict_proba(X)[0]))

    # Feature importance from the GradientBoosting step
    clf          = model.named_steps['clf']
    importances  = dict(zip(FEATURE_COLS, clf.feature_importances_))
    top_factors  = sorted(importances, key=importances.get, reverse=True)[:3]
    explanation  = f"Risk driven by: {', '.join(top_factors)}"

    return {'risk': risk, 'probabilities': proba, 'explanation': explanation}

if __name__ == "__main__":
    train()
    sample = {
        'lake_area_km2': 450, 'growth_rate': 20, 'mean_slope_deg': 40,
        'mean_elevation_m': 4500, 'temp_trend': 2.0, 'rainfall_mm': 1500,
        'ice_melt_rate': 12.0, 'moraine_stability': 0.3,
        'seismic_activity': 7.0, 'glacier_retreat_rate': 80.0
    }
    print(predict(sample))
