"""
FastAPI service for Ko Tao grid predictive dispatch.

POST /stream/telemetry  — ingest one row; returns forecast when buffer is full
POST /stream/actual     — record actual load; returns live RMSE/MAE/MAPE
GET  /stream/metrics    — current running error metrics
GET  /health            — liveness check
GET  /metrics           — last evaluation_report.json
POST /forecast          — batch 24h forecast (full history window required)
POST /warnings          — early-warning check
"""
import os, sys, pickle, subprocess, tempfile
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

import json, math, sqlite3
import numpy as np
import torch
import yaml
import mlflow
from collections import deque
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

mlflow.set_tracking_uri(f"sqlite:///{os.path.join(ROOT, 'mlflow.db')}")
mlflow.set_experiment("GridTokenX_API")

from models.tcn_model import TCN, SEQ_FEATURES
from models.device import get_device
from optimizer.dispatch import run_dispatch, schedule_summary
from optimizer.early_warning import check_warnings, format_warnings

# ── Load models at startup ────────────────────────────────────────────────────
with open(os.path.join(ROOT, "config.yaml")) as f:
    CFG = yaml.safe_load(f)

DEVICE = get_device()
CKPT   = torch.load(os.path.join(ROOT, "models/tcn.pt"), map_location=DEVICE,
                    weights_only=False)

with open(os.path.join(ROOT, "models/meta_learner.pkl"), "rb") as f:
    META = pickle.load(f)

with open(os.path.join(ROOT, "models/lgbm.pkl"), "rb") as f:
    _LGBM_MODEL = pickle.load(f)
    # Force single thread for inference to avoid macOS OpenMP segfaults
    _LGBM_MODEL.set_params(n_jobs=1)

from models.lgbm_model import FEATURES as LGBM_FEATURES

tc = CKPT["config"]
_NET = TCN(CKPT["in_features"], tc["filters"], tc["kernel_size"],
           tc["layers"], tc["forecast_horizon"],
           dropout=CKPT.get("dropout", 0.0)).to(DEVICE)
_NET.load_state_dict(CKPT["state_dict"])
_NET.eval()

# SEQ_FEATURES maps to TelemetryRow field names (snake_case)
_SEQ_FIELD = [f.lower() for f in SEQ_FEATURES]

# ── Schemas ───────────────────────────────────────────────────────────────────

class TelemetryRow(BaseModel):
    """One hour of real-time telemetry — matches SEQ_FEATURES exactly."""
    island_load_mw:  float
    load_lag_1h:     float
    load_lag_24h:    float
    bess_soc_pct:    float
    headroom_mw:     float
    dry_bulb_temp:   float
    heat_index:      float
    rel_humidity:    float
    hour_of_day:     float
    is_high_season:  float
    is_thai_holiday: float


class TelemetryStreamRequest(BaseModel):
    row: TelemetryRow
    circuit_forecast: Optional[List[float]] = Field(None, min_length=24, max_length=96)
    lgbm_features: Optional[dict] = None


class ActualRequest(BaseModel):
    timestamp_iso: str
    actual_load_mw: float
    forecast_load_mw: float


class ForecastRequest(BaseModel):
    history: List[TelemetryRow]
    circuit_forecast: List[float] = Field(..., min_length=24, max_length=96)
    initial_soc: float = Field(0.65, ge=0.2, le=0.95)
    lgbm_features: dict


class WarningRequest(BaseModel):
    load_forecast:    List[float] = Field(..., min_length=1, max_length=96)
    circuit_forecast: List[float] = Field(..., min_length=1, max_length=96)
    current_soc:      float = Field(..., ge=0.0, le=1.0)
    lookahead_hours:  int   = Field(6, ge=1, le=24)


# ── Streaming state ───────────────────────────────────────────────────────────

class StreamingEngine:
    def __init__(self, window_size: int, db_path: str):
        self.window_size = window_size
        self.db_path = db_path
        self.buffer: deque = deque(maxlen=window_size)
        self._actuals:   List[float] = []
        self._forecasts: List[float] = []
        self._init_db()
        self._load_state()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cols = ", ".join([f"{f} REAL" for f in _SEQ_FIELD])
            conn.execute(f"CREATE TABLE IF NOT EXISTS telemetry (id INTEGER PRIMARY KEY AUTOINCREMENT, {cols})")
            conn.execute("CREATE TABLE IF NOT EXISTS metrics (id INTEGER PRIMARY KEY AUTOINCREMENT, actual REAL, forecast REAL)")
            conn.commit()

    def _load_state(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f"SELECT COUNT(*) FROM telemetry")
            if cursor.fetchone()[0] > 0:
                rows = conn.execute(f"SELECT {', '.join(_SEQ_FIELD)} FROM (SELECT * FROM telemetry ORDER BY id DESC LIMIT ?) ORDER BY id ASC", (self.window_size,)).fetchall()
                for r in rows:
                    self.buffer.append(TelemetryRow(**dict(zip(_SEQ_FIELD, r))))
            metrics = conn.execute("SELECT actual, forecast FROM metrics ORDER BY id ASC").fetchall()
            for a, f in metrics:
                self._actuals.append(a)
                self._forecasts.append(f)
        print(f"   [API] Restored {len(self.buffer)} telemetry rows and {len(self._actuals)} metric pairs.")

    def ingest(self, row: TelemetryRow):
        self.buffer.append(row)
        with sqlite3.connect(self.db_path) as conn:
            vals = [getattr(row, f) for f in _SEQ_FIELD]
            placeholders = ", ".join(["?"] * len(_SEQ_FIELD))
            conn.execute(f"INSERT INTO telemetry ({', '.join(_SEQ_FIELD)}) VALUES ({placeholders})", vals)
            conn.execute("DELETE FROM telemetry WHERE id NOT IN (SELECT id FROM telemetry ORDER BY id DESC LIMIT 2000)")
            conn.commit()

    def is_ready(self) -> bool:
        return len(self.buffer) == self.window_size

    def record_actual(self, actual: float, forecast: float):
        self._actuals.append(actual)
        self._forecasts.append(forecast)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO metrics (actual, forecast) VALUES (?, ?)", (actual, forecast))
            conn.commit()

    def live_metrics(self) -> dict:
        n = len(self._actuals)
        if n == 0:
            return {"n": 0, "mae": None, "rmse": None, "mape": None}
        a = np.array(self._actuals)
        f = np.array(self._forecasts)
        err = a - f
        mae  = float(np.mean(np.abs(err)))
        rmse = float(math.sqrt(np.mean(err ** 2)))
        mape = float(np.mean(np.abs(err / (a + 1e-8))) * 100)
        return {"n": n, "mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 4)}


STREAM = StreamingEngine(tc["window_size"], os.path.join(ROOT, "api_state.db"))

# ── Helpers ───────────────────────────────────────────────────────────────────

@mlflow.trace(span_type="FUNC", name="lgbm_predict")
def _lgbm_predict(features: dict) -> float:
    X_row = [features.get(c, 0.0) for c in LGBM_FEATURES]
    X = np.array([X_row])
    return float(_LGBM_MODEL.predict(X)[0])


@mlflow.trace(span_type="FUNC", name="tcn_predict")
def _tcn_predict(history) -> np.ndarray:
    arr = np.array([[getattr(r, f) for f in _SEQ_FIELD] for r in history], dtype=np.float32)
    x = torch.tensor(arr).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return _NET(x).cpu().numpy()[0]


@mlflow.trace(span_type="FUNC", name="hybrid_forecast")
def _hybrid_forecast(history, lgbm_features: dict) -> List[float]:
    horizon = tc["forecast_horizon"]
    lgbm_pred = _lgbm_predict(lgbm_features)
    tcn_preds = _tcn_predict(history)
    X_meta = np.column_stack([np.full(horizon, lgbm_pred), tcn_preds])
    return META.predict(X_meta).tolist()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="GridTokenX Forecast API", version="2.0.0")


@app.post("/stream/telemetry")
def stream_telemetry(req: TelemetryStreamRequest):
    STREAM.ingest(req.row)
    ready = STREAM.is_ready()
    if req.circuit_forecast and req.lgbm_features and ready:
        try:
            forecast = _hybrid_forecast(list(STREAM.buffer), req.lgbm_features)
            schedule = run_dispatch(np.array(forecast), np.array(req.circuit_forecast),
                                    initial_soc=req.row.bess_soc_pct / 100.0, cfg=CFG)
            return {
                "status": "forecast",
                "buffer_size": len(STREAM.buffer),
                "forecast_mw": forecast,
                "summary": schedule_summary(schedule),
                "live_metrics": STREAM.live_metrics(),
            }
        except Exception as e:
            raise HTTPException(500, str(e))
    return {"status": "ingested", "buffer_size": len(STREAM.buffer), "ready": ready}


@app.post("/stream/actual")
def stream_actual(req: ActualRequest):
    STREAM.record_actual(req.actual_load_mw, req.forecast_load_mw)
    return {"metrics": STREAM.live_metrics()}


@app.get("/stream/metrics")
def stream_metrics():
    return STREAM.live_metrics()


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "buffer": len(STREAM.buffer)}


@app.get("/metrics")
def metrics():
    path = os.path.join(ROOT, "results/evaluation_report.json")
    if not os.path.exists(path): raise HTTPException(404, "Run evaluate.py first.")
    with open(path) as f: return json.load(f)


@app.post("/warnings")
def warnings(req: WarningRequest):
    w = check_warnings(np.array(req.load_forecast), np.array(req.circuit_forecast),
                       req.current_soc, cfg=CFG, lookahead_hours=req.lookahead_hours)
    return {"count": len(w), "critical": sum(1 for x in w if x.level == "CRITICAL"),
            "warnings": [x.__dict__ for x in w], "summary": format_warnings(w)}


@app.post("/forecast")
def forecast(req: ForecastRequest):
    if len(req.history) != tc["window_size"]:
        raise HTTPException(422, f"history must have {tc['window_size']} rows")
    try:
        fc = _hybrid_forecast(req.history, req.lgbm_features)
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    schedule = run_dispatch(np.array(fc), np.array(req.circuit_forecast),
                            initial_soc=req.initial_soc, cfg=CFG)
    return {"forecast_mw": fc, "summary": schedule_summary(schedule)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.serve:app", host="0.0.0.0", port=8000, reload=False)
