"""
Tests for api/serve.py — FastAPI endpoint integration tests.
Uses TestClient (no real HTTP server needed).
"""
import os
import sys
import json
import pytest
import warnings as stdlib_warnings

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Suppress the sklearn version mismatch warning in tests
stdlib_warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from fastapi.testclient import TestClient
from api.serve import app


@pytest.fixture(scope="module")
def client():
    """Create a TestClient for the FastAPI app."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_shape(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert "device" in data
        assert "buffer" in data
        assert "window" in data
        assert isinstance(data["window"], int)
        assert data["window"] > 0


class TestMetricsEndpoint:
    """Tests for GET /metrics."""

    def test_metrics_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_contains_forecast(self, client):
        data = client.get("/metrics").json()
        assert "forecast" in data
        assert "mape" in data["forecast"]
        assert "r2" in data["forecast"]

    def test_metrics_contains_targets(self, client):
        data = client.get("/metrics").json()
        assert "targets_met" in data


class TestStreamTelemetryEndpoint:
    """Tests for POST /stream/telemetry."""

    def _make_row(self, **overrides):
        row = {
            "island_load_mw": 8.5,
            "load_lag_1h": 8.3,
            "load_lag_24h": 7.9,
            "bess_soc_pct": 65.0,
            "headroom_mw": 1.5,
            "dry_bulb_temp": 32.0,
            "heat_index": 38.0,
            "rel_humidity": 75.0,
            "hour_of_day": 14.0,
            "is_high_season": 1.0,
            "is_thai_holiday": 0.0
        }
        row.update(overrides)
        return row

    def test_ingest_single_row(self, client):
        resp = client.post("/stream/telemetry", json={"row": self._make_row()})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ingested"
        assert "buffer_size" in data
        assert "ready" in data

    def test_rejects_missing_fields(self, client):
        """Missing required TelemetryRow field should return 422."""
        incomplete = {"island_load_mw": 8.5}  # missing other fields
        resp = client.post("/stream/telemetry", json={"row": incomplete})
        assert resp.status_code == 422

    def test_rejects_empty_body(self, client):
        resp = client.post("/stream/telemetry", json={})
        assert resp.status_code == 422


class TestStreamActualEndpoint:
    """Tests for POST /stream/actual."""

    def test_record_actual(self, client):
        resp = client.post("/stream/actual", json={
            "timestamp_iso": "2029-04-02T08:00:00",
            "actual_load_mw": 8.5,
            "forecast_load_mw": 8.3,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "metrics" in data
        assert "mae" in data["metrics"]

    def test_rejects_missing_fields(self, client):
        resp = client.post("/stream/actual", json={"actual_load_mw": 8.5})
        assert resp.status_code == 422


class TestStreamMetricsEndpoint:
    """Tests for GET /stream/metrics."""

    def test_returns_200(self, client):
        resp = client.get("/stream/metrics")
        assert resp.status_code == 200

    def test_response_shape(self, client):
        data = client.get("/stream/metrics").json()
        assert "n" in data
        # MAE/RMSE/MAPE can be None if no actuals recorded yet
        assert "mae" in data
        assert "rmse" in data
        assert "mape" in data


class TestWarningsEndpoint:
    """Tests for POST /warnings."""

    def test_stable_grid_no_warnings(self, client):
        resp = client.post("/warnings", json={
            "load_forecast": [6.0] * 24,
            "circuit_forecast": [14.0] * 24,
            "current_soc": 0.65,
            "lookahead_hours": 6,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["critical"] == 0

    def test_bottleneck_triggers_warnings(self, client):
        circuit = [14.0] * 18 + [3.0, 2.5, 3.5, 4.0] + [14.0, 14.0]
        resp = client.post("/warnings", json={
            "load_forecast": [8.0] * 24,
            "circuit_forecast": circuit,
            "current_soc": 0.30,
            "lookahead_hours": 24,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] > 0

    def test_rejects_wrong_length(self, client):
        """Load forecast must have 1–24 elements."""
        resp = client.post("/warnings", json={
            "load_forecast": [],
            "circuit_forecast": [14.0] * 24,
            "current_soc": 0.65,
        })
        assert resp.status_code == 422

    def test_rejects_soc_out_of_range(self, client):
        resp = client.post("/warnings", json={
            "load_forecast": [8.0] * 24,
            "circuit_forecast": [14.0] * 24,
            "current_soc": 1.5,  # > 1.0
        })
        assert resp.status_code == 422

    def test_summary_is_string(self, client):
        resp = client.post("/warnings", json={
            "load_forecast": [6.0] * 24,
            "circuit_forecast": [14.0] * 24,
            "current_soc": 0.65,
        })
        data = resp.json()
        assert isinstance(data["summary"], str)


class TestForecastEndpoint:
    """Tests for POST /forecast."""

    def test_rejects_wrong_history_length(self, client):
        """History must have exactly window_size rows."""
        row = {
            "island_load_mw": 8.0, "load_lag_1h": 7.8, "load_lag_24h": 7.5,
            "bess_soc_pct": 65.0, "dry_bulb_temp": 30.0, "heat_index": 35.0,
            "rel_humidity": 70.0, "hour_of_day": 12.0, "is_high_season": 1.0,
        }
        resp = client.post("/forecast", json={
            "history": [row] * 10,  # too few — need 48
            "circuit_forecast": [14.0] * 24,
            "lgbm_features": {"Dry_Bulb_Temp": 30},
        })
        assert resp.status_code == 422

    def test_rejects_missing_circuit_forecast(self, client):
        row = {
            "island_load_mw": 8.0, "load_lag_1h": 7.8, "load_lag_24h": 7.5,
            "bess_soc_pct": 65.0, "dry_bulb_temp": 30.0, "heat_index": 35.0,
            "rel_humidity": 70.0, "hour_of_day": 12.0, "is_high_season": 1.0,
        }
        resp = client.post("/forecast", json={
            "history": [row] * 48,
            "lgbm_features": {"Dry_Bulb_Temp": 30},
        })
        assert resp.status_code == 422


class TestOpenAPIDocs:
    """Tests for GET /docs — OpenAPI documentation."""

    def test_docs_accessible(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_openapi_json(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "paths" in schema
        assert "/health" in schema["paths"]
        assert "/forecast" in schema["paths"]
