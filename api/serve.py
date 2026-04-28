"""
FastAPI service for Ko Tao grid predictive dispatch.

POST /forecast  — 24h telemetry → load forecast + dispatch schedule
GET  /health    — liveness check
GET  /metrics   — last evaluation report
"""
import os, sys, pickle, subprocess, tempfile
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

import json
import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

from models.tcn_model import TCN, SEQ_FEATURES
from models.device import get_device
from optimizer.dispatch import run_dispatch, schedule_summary
from optimizer.early_warning import check_warnings, format_warnings

# ── Load models at startup ────────────────────────────────────────────────────
with open(os.path.join(ROOT, "config.yaml")) as f:
    CFG = yaml.safe_load(f)

DEVICE = get_device()
CKPT   = torch.load(os.path.join(ROOT, "models/tcn.pt"), map_location=DEVICE)

with open(os.path.join(ROOT, "models/meta_learner.pkl"), "rb") as f:
    META = pickle.load(f)

tc = CKPT["config"]
_NET = TCN(CKPT["in_features"], tc["filters"], tc["kernel_size"],
           tc["layers"], tc["forecast_horizon"]).to(DEVICE)
_NET.load_state_dict(CKPT["state_dict"])
_NET.eval()

# ── Schemas ───────────────────────────────────────────────────────────────────

class TelemetryRow(BaseModel):
    island_load_mw:   float
    circuit_cap_mw:   float
    bess_soc_pct:     float
    net_delta_mw:     float
    load_lag_1h:      float
    load_lag_24h:     float

class ForecastRequest(BaseModel):
    # Exactly window_size rows of sequential telemetry
    history: List[TelemetryRow]
    # 24h ahead circuit capacity forecast (from SCADA/planner)
    circuit_forecast: List[float] = Field(..., min_length=24, max_length=24)
    initial_soc: float = Field(0.65, ge=0.2, le=0.95)
    # Tabular features for LightGBM (latest snapshot)
    lgbm_features: dict

class HourlyResult(BaseModel):
    hour: int
    load_mw: float
    circuit_mw: float
    diesel_mw: float
    bess_mw: float
    bess_soc: float
    fuel_kg: float
    carbon_kg: float

class ForecastResponse(BaseModel):
    forecast_mw: List[float]
    schedule: List[HourlyResult]
    summary: dict
    device: str

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="GridTokenX Forecast API", version="1.0.0")


def _lgbm_predict_subprocess(features: dict) -> float:
    """Run single LightGBM prediction in isolated subprocess."""
    out = tempfile.mktemp(suffix=".npy")
    script = f"""
import os, sys, pickle, numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, {repr(ROOT)})
from models.lgbm_model import FEATURES
with open("models/lgbm.pkl","rb") as f: model = pickle.load(f)
row = {repr(features)}
X = np.array([[row[c] for c in FEATURES]])
np.save({repr(out)}, model.predict(X))
"""
    r = subprocess.run([sys.executable, "-c", script],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    val = float(np.load(out)[0])
    os.unlink(out)
    return val


def _tcn_predict(history: List[TelemetryRow]) -> np.ndarray:
    """Run TCN on history window → 24h forecast."""
    seq_map = {
        "island_load_mw": "Island_Load_MW",
        "bess_soc_pct":   "BESS_SoC_Pct",
        "net_delta_mw":   "Net_Delta_MW",
        "load_lag_1h":    "Load_Lag_1h",
        "load_lag_24h":   "Load_Lag_24h",
    }
    col_order = ["Island_Load_MW", "BESS_SoC_Pct", "Net_Delta_MW",
                 "Load_Lag_1h", "Load_Lag_24h"]
    arr = np.array([[getattr(r, k) for k in seq_map] for r in history],
                   dtype=np.float32)  # (T, 5)
    x = torch.tensor(arr).unsqueeze(0).to(DEVICE)  # (1, T, 5)
    with torch.no_grad():
        out = _NET(x).cpu().numpy()[0]  # (24,)
    return out


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.get("/metrics")
def metrics():
    path = os.path.join(ROOT, "results/evaluation_report.json")
    if not os.path.exists(path):
        raise HTTPException(404, "No evaluation report found. Run evaluate.py first.")
    with open(path) as f:
        return json.load(f)


class WarningRequest(BaseModel):
    load_forecast:    List[float] = Field(..., min_length=1, max_length=24)
    circuit_forecast: List[float] = Field(..., min_length=1, max_length=24)
    current_soc:      float = Field(..., ge=0.0, le=1.0)
    lookahead_hours:  int   = Field(6, ge=1, le=24)


@app.post("/warnings")
def warnings(req: WarningRequest):
    w = check_warnings(
        np.array(req.load_forecast),
        np.array(req.circuit_forecast),
        req.current_soc,
        cfg=CFG,
        lookahead_hours=req.lookahead_hours,
    )
    return {
        "count":    len(w),
        "critical": sum(1 for x in w if x.level == "CRITICAL"),
        "warnings": [x.__dict__ for x in w],
        "summary":  format_warnings(w),
    }


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    window = tc["window_size"]
    if len(req.history) != window:
        raise HTTPException(422, f"history must have exactly {window} rows, got {len(req.history)}")

    # LightGBM point forecast (latest snapshot)
    try:
        lgbm_pred = _lgbm_predict_subprocess(req.lgbm_features)
    except RuntimeError as e:
        raise HTTPException(500, f"LightGBM error: {e}")

    # TCN 24h forecast
    tcn_preds = _tcn_predict(req.history)  # shape (24,)

    # Meta-learner: combine lgbm scalar with each TCN step
    X_meta = np.column_stack([np.full(24, lgbm_pred), tcn_preds])
    hybrid_forecast = META.predict(X_meta).tolist()

    # Dispatch schedule
    schedule = run_dispatch(
        np.array(hybrid_forecast),
        np.array(req.circuit_forecast),
        initial_soc=req.initial_soc,
        cfg=CFG,
    )
    summary = schedule_summary(schedule)

    return ForecastResponse(
        forecast_mw=hybrid_forecast,
        schedule=[HourlyResult(**s.__dict__) for s in schedule],
        summary=summary,
        device=DEVICE,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.serve:app", host="0.0.0.0", port=8000, reload=False)
