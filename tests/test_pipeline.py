"""Basic end-to-end checks for the predictive maintenance pipeline.

Uses a temporary folder so the real data/ and models/ are not touched.
Run:  pytest -q
"""
import numpy as np
import pytest

from src.features import FEATURE_COLUMNS, LABEL, build_features
from src.generate_data import SENSORS, generate
from src.predict import load_model, predict_proba
from src.train import train

EXPECTED_COLUMNS = ["timestamp", "equipment_id", "equipment_type", *SENSORS, "operating_hours"]


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    model_dir = tmp_path_factory.mktemp("models")
    readings, failures, maintenance = generate(n_days=90, out_dir=data_dir)
    train(data_dir=data_dir, model_dir=model_dir, verbose=False)
    return readings, failures, maintenance, model_dir


def test_generator_columns_and_no_empty_columns(artifacts):
    readings, failures, maintenance, _ = artifacts
    for col in EXPECTED_COLUMNS:
        assert col in readings.columns
    assert not readings.isna().all().any(), "a column is fully null"
    assert readings["equipment_id"].nunique() == 20
    assert {"equipment_id", "failure_time", "failure_mode"} <= set(failures.columns)
    assert len(maintenance) > 0


def test_label_has_both_classes(artifacts):
    readings, failures, maintenance, _ = artifacts
    feats = build_features(readings, failures, maintenance)
    assert set(feats[LABEL].unique()) == {0, 1}
    assert set(readings["failure_within_7d"].unique()) == {0, 1}


def test_model_saved_and_probabilities_valid(artifacts):
    readings, failures, maintenance, model_dir = artifacts
    assert (model_dir / "model.joblib").exists()
    assert (model_dir / "metrics.json").exists()
    bundle = load_model(model_dir)
    feats = build_features(readings, failures, maintenance)
    proba = predict_proba(bundle, feats[FEATURE_COLUMNS])
    assert len(proba) == len(feats)
    assert np.all((proba >= 0) & (proba <= 1))
