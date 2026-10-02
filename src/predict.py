"""
Load the trained model and score equipment.

Run:  python -m src.predict   (prints the latest risk for every unit)
"""
from pathlib import Path

import joblib
import pandas as pd

from src.features import build_features
from src.generate_data import DATA_DIR, load_raw
from src.train import MODEL_DIR


def load_model(model_dir=MODEL_DIR):
    """Returns {"model": RandomForestClassifier, "features": [column names]}."""
    return joblib.load(Path(model_dir) / "model.joblib")


def predict_proba(bundle, X):
    """Failure probability (0..1) for each row of a feature DataFrame."""
    return bundle["model"].predict_proba(X[bundle["features"]])[:, 1]


def risk_band(p):
    if p > 0.7:
        return "High"
    if p >= 0.3:
        return "Medium"
    return "Low"


def recommend_action(p):
    if p > 0.7:
        return "Schedule inspection within 48 hours and prepare spare parts."
    if p >= 0.3:
        return "Increase monitoring frequency and plan an inspection this week."
    return "Normal operation - continue routine preventive maintenance."


def score_all(bundle, feats):
    """Add failure_probability and risk_band to every daily row."""
    out = feats.copy()
    out["failure_probability"] = predict_proba(bundle, out)
    out["risk_band"] = out["failure_probability"].apply(risk_band)
    return out


def latest_risk(scored):
    """Most recent snapshot per unit, highest risk first."""
    latest = scored.sort_values("date").groupby("equipment_id").tail(1)
    return latest.sort_values("failure_probability", ascending=False).reset_index(drop=True)


def main(data_dir=DATA_DIR, model_dir=MODEL_DIR):
    bundle = load_model(model_dir)
    scored = score_all(bundle, build_features(*load_raw(data_dir)))
    latest = latest_risk(scored)
    latest[["equipment_id", "equipment_type", "date", "failure_probability", "risk_band"]] \
        .to_csv(Path(data_dir) / "latest_predictions.csv", index=False)
    print(latest[["equipment_id", "equipment_type", "failure_probability", "risk_band"]]
          .to_string(index=False))
    return latest


if __name__ == "__main__":
    main()
