"""
Train, evaluate and save the failure-risk model.

Why recall matters more than accuracy here
------------------------------------------
Only a small share of days are "a failure is coming". A model that ALWAYS says
"no failure" would score high accuracy and still be useless. In maintenance:
  * A missed failure (false negative) = unplanned downtime, safety risk, costly repairs.
  * A false alarm (false positive)    = an extra inspection, which is cheap.
So we care most about RECALL: "of all real upcoming failures, how many did we catch?"
That is also why we use class_weight="balanced" - it makes the rare failure class
count as much as the common healthy class during training.

Run:  python -m src.train
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from src.features import FEATURE_COLUMNS, LABEL, build_features
from src.generate_data import DATA_DIR, generate, load_raw

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
THRESHOLD = 0.5


def time_split(df, train_frac=0.75):
    """First ~75% of calendar days -> train, the rest -> test. No shuffling,
    because in real life we always predict the future from the past."""
    dates = sorted(df["date"].unique())
    cutoff = dates[int(len(dates) * train_frac)]
    return df[df["date"] < cutoff], df[df["date"] >= cutoff]


def train(data_dir=DATA_DIR, model_dir=MODEL_DIR, verbose=True):
    data_dir, model_dir = Path(data_dir), Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if not (data_dir / "sensor_readings.csv").exists():
        generate(out_dir=data_dir)

    feats = build_features(*load_raw(data_dir))
    train_df, test_df = time_split(feats)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",  # compensate for rare failures
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_df[FEATURE_COLUMNS], train_df[LABEL])

    proba = model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    y = test_df[LABEL]
    metrics = {
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "threshold": THRESHOLD,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "test_positive_rate": round(float(y.mean()), 4),
        "train_period": [str(train_df["date"].min().date()), str(train_df["date"].max().date())],
        "test_period": [str(test_df["date"].min().date()), str(test_df["date"].max().date())],
    }

    joblib.dump({"model": model, "features": FEATURE_COLUMNS}, model_dir / "model.joblib")
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": model.feature_importances_}) \
        .sort_values("importance", ascending=False) \
        .to_csv(model_dir / "feature_importance.csv", index=False)

    if verbose:
        print(f"Train rows: {metrics['n_train']}  Test rows: {metrics['n_test']}")
        for k in ["precision", "recall", "f1", "roc_auc", "accuracy"]:
            print(f"  {k:<10}: {metrics[k]:.3f}")
        print(f"  confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")
        print(f"Saved model to {model_dir / 'model.joblib'}")
    return metrics


if __name__ == "__main__":
    train()
