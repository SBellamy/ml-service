import importlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def write_production_artifacts(artifacts_dir: Path):
    prod = artifacts_dir / "models" / "production"
    prod.mkdir(parents=True, exist_ok=True)

    # Tiny synthetic training data
    X = np.array(
        [
            [25, 52000, 1200, 14, 0],
            [42, 88000, 5400, 33, 1],
            [31, 61000, 2300, 18, 0],
            [58, 99000, 15000, 41, 1],
        ],
        dtype=float,
    )
    y = np.array([0, 1, 0, 1], dtype=int)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(X, y)

    joblib.dump(model, prod / "model.joblib")
    (prod / "metadata.json").write_text(
        json.dumps({"metric_name": "f1", "score": 1.0, "trained_at": 0}, indent=2)
    )


def load_app_with_env(monkeypatch, artifacts_dir: Path):
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.delenv("MODEL_SUBDIR", raising=False)  # use default models/production

    import api.main as main_mod
    importlib.reload(main_mod)
    return main_mod.app


def test_healthz_always_ok(tmp_path, monkeypatch):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    app = load_app_with_env(monkeypatch, artifacts_dir)
    client = TestClient(app)

    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_readyz_false_when_no_model(tmp_path, monkeypatch):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    app = load_app_with_env(monkeypatch, artifacts_dir)
    client = TestClient(app)

    r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["ready"] is False


def test_predict_503_when_not_ready(tmp_path, monkeypatch):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    app = load_app_with_env(monkeypatch, artifacts_dir)
    client = TestClient(app)

    r = client.post(
        "/predict",
        json={
            "age": 42,
            "income": 88000,
            "account_balance": 5400,
            "transactions_last_30d": 33,
            "is_premium": 1,
        },
    )
    assert r.status_code == 503


def test_predict_ok_after_model_reload(tmp_path, monkeypatch):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    # Start app with empty artifacts
    app = load_app_with_env(monkeypatch, artifacts_dir)
    client = TestClient(app)

    # Write artifacts after app startup - then reload
    write_production_artifacts(artifacts_dir)

    r = client.post("/model/reload")
    assert r.status_code == 200
    assert r.json()["reloaded"] is True

    r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["ready"] is True

    r = client.post(
        "/predict",
        json={
            "age": 42,
            "income": 88000,
            "account_balance": 5400,
            "transactions_last_30d": 33,
            "is_premium": 1,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body
    assert "probability" in body
    assert body["prediction"] in [0, 1]
    assert 0.0 <= body["probability"] <= 1.0