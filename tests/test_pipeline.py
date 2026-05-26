"""
tests/test_pipeline.py
───────────────────────
Unit and integration tests for the house-price MLOps pipeline.

Run with:
    pytest tests/ -v --cov=src --cov-report=term-missing
"""

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parents[1]))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_ames_df() -> pd.DataFrame:
    """Minimal Ames Housing-shaped DataFrame for testing."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "OverallQual":  rng.integers(1, 10, n),
        "GrLivArea":    rng.integers(600, 3000, n).astype(float),
        "GarageCars":   rng.integers(0, 4, n),
        "TotalBsmtSF":  rng.integers(0, 2000, n).astype(float),
        "YearBuilt":    rng.integers(1950, 2010, n),
        "YrSold":       rng.integers(2006, 2011, n),
        "LotArea":      rng.integers(3000, 20000, n).astype(float),
        "FullBath":     rng.integers(1, 3, n),
        "HalfBath":     rng.integers(0, 2, n),
        "BsmtFullBath": rng.integers(0, 2, n),
        "BsmtHalfBath": rng.integers(0, 1, n),
        "OpenPorchSF":  rng.integers(0, 200, n).astype(float),
        "WoodDeckSF":   rng.integers(0, 400, n).astype(float),
        "Neighborhood": rng.choice(["NAmes", "CollgCr", "OldTown", "Edwards"], n),
        "SalePrice":    rng.integers(80000, 400000, n).astype(float),
    })


@pytest.fixture
def sample_csv(tmp_path, sample_ames_df) -> Path:
    p = tmp_path / "ames_housing_test.csv"
    sample_ames_df.to_csv(p, index=False)
    return p


# ── Ingest tests ──────────────────────────────────────────────────────────────

class TestIngest:
    def test_sha256_stable(self, sample_ames_df):
        from src.ingest.ingest import _sha256
        h1 = _sha256(sample_ames_df)
        h2 = _sha256(sample_ames_df.copy())
        assert h1 == h2
        assert len(h1) == 16

    def test_validate_detects_missing_cols(self, sample_ames_df):
        from src.ingest.ingest import _validate
        df_missing = sample_ames_df.drop(columns=["SalePrice"])
        report = _validate(df_missing)
        assert "SalePrice" in report["missing_required_cols"]

    def test_validate_null_rates(self, sample_ames_df):
        from src.ingest.ingest import _validate
        report = _validate(sample_ames_df)
        assert "null_rate" in report
        assert report["rows"] == len(sample_ames_df)

    @patch("src.ingest.ingest.mlflow")
    @patch("src.ingest.ingest._download")
    def test_ingest_saves_csv(self, mock_download, mock_mlflow, tmp_path, sample_ames_df):
        mock_download.return_value = sample_ames_df
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_mlflow.start_run.return_value.__exit__  = MagicMock(return_value=False)

        from src.ingest import ingest as ingest_mod
        ingest_mod.settings.RAW_DIR = tmp_path / "raw"
        ingest_mod.settings.RAW_DIR.mkdir(parents=True)

        path = ingest_mod.ingest(source="primary")
        assert path.exists()
        loaded = pd.read_csv(path)
        assert len(loaded) == len(sample_ames_df)


# ── Feature engineering tests ─────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_log_features_created(self, sample_ames_df, tmp_path):
        from src.features import engineer as eng_mod
        import numpy as np

        eng_mod.settings.RAW_DIR      = tmp_path / "raw"
        eng_mod.settings.PROCESSED_DIR = tmp_path / "processed"
        eng_mod.settings.RAW_DIR.mkdir(parents=True)
        eng_mod.settings.PROCESSED_DIR.mkdir(parents=True)

        csv_path = eng_mod.settings.RAW_DIR / "ames_housing_abc123.csv"
        sample_ames_df.to_csv(csv_path, index=False)

        with patch("src.features.engineer.mlflow"):
            result_path = eng_mod.engineer(input_csv=csv_path)

        result = pd.read_csv(result_path)
        assert "log_GrLivArea"    in result.columns
        assert "log_LotArea"      in result.columns
        assert "qual_area"        in result.columns
        assert "age_at_sale"      in result.columns
        assert "total_bathrooms"  in result.columns

    def test_log_features_nonnegative(self, sample_ames_df, tmp_path):
        from src.features import engineer as eng_mod

        eng_mod.settings.RAW_DIR       = tmp_path / "raw"
        eng_mod.settings.PROCESSED_DIR = tmp_path / "processed"
        eng_mod.settings.RAW_DIR.mkdir(parents=True)
        eng_mod.settings.PROCESSED_DIR.mkdir(parents=True)

        csv_path = eng_mod.settings.RAW_DIR / "ames_housing_abc123.csv"
        sample_ames_df.to_csv(csv_path, index=False)

        with patch("src.features.engineer.mlflow"):
            result_path = eng_mod.engineer(input_csv=csv_path)

        result = pd.read_csv(result_path)
        assert (result["log_GrLivArea"] >= 0).all()
        assert (result["log_LotArea"]   >= 0).all()


# ── Monitoring / PSI tests ────────────────────────────────────────────────────

class TestMonitoring:
    def test_psi_identical_distributions(self):
        from src.monitor.monitor import _psi
        x = np.random.default_rng(0).normal(0, 1, 1000)
        assert _psi(x, x) < 0.05

    def test_psi_different_distributions(self):
        from src.monitor.monitor import _psi
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 1000)
        y = rng.normal(5, 1, 1000)   # very different mean
        assert _psi(x, y) > 0.2

    def test_ks_test_same_distribution(self):
        from src.monitor.monitor import _ks_test
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 500)
        y = rng.normal(0, 1, 500)
        _, p = _ks_test(x, y)
        assert p > 0.05   # no significant difference expected

    def test_ks_test_different_distribution(self):
        from src.monitor.monitor import _ks_test
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 500)
        y = rng.normal(10, 1, 500)
        _, p = _ks_test(x, y)
        assert p < 0.05


# ── API tests ─────────────────────────────────────────────────────────────────

class TestAPI:
    @pytest.fixture(autouse=True)
    def mock_model(self):
        """Inject a fake model bundle so tests don't need a real trained model."""
        import xgboost as xgb
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.compose import ColumnTransformer

        rng = np.random.default_rng(0)
        n, f = 200, 10
        X = rng.random((n, f))
        y = rng.random(n) * 12 + 11   # log-scale ~$60k–$300k

        model = xgb.XGBRegressor(n_estimators=10, random_state=0)
        model.fit(X, y)

        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())]),
             list(range(f))),
        ])
        pre.fit(X)

        fake_bundle = {
            "preprocessor": pre,
            "model": model,
            "num_cols": [],
            "cat_cols": [],
        }

        import src.api.main as api_mod
        api_mod._MODEL_BUNDLE = fake_bundle
        api_mod._MODEL_META   = {"model_version": "test-1", "run_id": "abc", "stage": "Testing"}
        yield

    @pytest.fixture
    def client(self):
        from src.api.main import app
        return TestClient(app)

    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["model_loaded"] is True

    def test_predict_returns_price(self, client):
        payload = {
            "OverallQual": 7, "GrLivArea": 1500,
            "GarageCars": 2, "TotalBsmtSF": 900,
            "YearBuilt": 2003, "Neighborhood": "NridgHt",
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "predicted_price_usd" in data
        assert data["predicted_price_usd"] > 0
        assert data["model_version"] == "test-1"

    def test_predict_confidence_interval(self, client):
        payload = {"OverallQual": 5, "GrLivArea": 1000}
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["confidence_interval_low"]  < data["predicted_price_usd"]
        assert data["confidence_interval_high"] > data["predicted_price_usd"]

    def test_predict_batch(self, client):
        houses = [{"OverallQual": i, "GrLivArea": 1000 + i * 100} for i in range(1, 6)]
        r = client.post("/predict/batch", json=houses)
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 5
        assert len(data["predictions"]) == 5

    def test_batch_limit(self, client):
        houses = [{"OverallQual": 5}] * 501
        r = client.post("/predict/batch", json=houses)
        assert r.status_code == 400

    def test_invalid_qual(self, client):
        r = client.post("/predict", json={"OverallQual": 99})
        assert r.status_code == 422   # Pydantic validation error
