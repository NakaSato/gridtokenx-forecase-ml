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
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
from sqlalchemy import create_engine, text
import mlflow
from models.mlflow_utils import setup_mlflow

setup_mlflow()
mlflow.set_experiment("GridTokenX_API")

# ── Database Connections ─────────────────────────────────────────────────────
POSTGIS_URL = os.getenv("POSTGIS_URL", "postgresql://gridtokenx:gridtokenx_pass@localhost:5432/gridtokenx_geo")
try:
    pg_engine = create_engine(POSTGIS_URL, echo=False)
except Exception as e:
    print(f"⚠️  PostGIS connection failed: {e}")
    pg_engine = None

from models.tcn_model import TCN, SEQ_FEATURES
from models.device import get_device
from optimizer.dispatch import run_dispatch, schedule_summary
from optimizer.early_warning import check_warnings, format_warnings
from optimizer.cluster_dispatch_admm import get_cluster_dispatch

# ── Load models at startup ────────────────────────────────────────────────────
with open(os.path.join(ROOT, "config.yaml")) as f:
    CFG = yaml.safe_load(f)

DEVICE = get_device()
_NET = None
META = None
tc = {"window_size": 48, "forecast_horizon": 24} # Defaults

try:
    CKPT = torch.load(os.path.join(ROOT, "models/tcn.pt"), map_location=DEVICE,
                        weights_only=False)
    tc = CKPT["config"]
    _NET = TCN(CKPT["in_features"], tc["filters"], tc["kernel_size"],
               tc["layers"], tc["forecast_horizon"],
               dropout=CKPT.get("dropout", 0.0)).to(DEVICE)
    _NET.load_state_dict(CKPT["state_dict"])
    _NET.eval()
    print("✅ TCN model loaded.")
except Exception as e:
    print(f"⚠️  TCN model not loaded: {e}")

try:
    with open(os.path.join(ROOT, "models/meta_learner.pkl"), "rb") as f:
        META = pickle.load(f)
    print("✅ Meta-learner loaded.")
except Exception as e:
    print(f"⚠️  Meta-learner not loaded: {e}")

# LightGBM is run via subprocess on macOS to avoid OpenMP conflicts with Torch
LGBM_FEATURES = [
    "Dry_Bulb_Temp", "Rel_Humidity", "Solar_Irradiance", "Heat_Index",
    "Temp_Roll_Mean_3h", "Temp_Roll_Mean_6h",
    "Humid_Roll_Mean_3h", "Humid_Roll_Mean_6h", "Temp_Gradient",
    "Carbon_Intensity", "Market_Price",
    "Tourist_Index", "Is_High_Season",
    "Hour_of_Day", "Day_of_Week",
    "Is_Thai_Holiday", "Is_Songkran",
    "Load_Lag_1h", "Load_Lag_24h", "Load_Lag_168h",
    "Load_Roll_Mean_3h", "Load_Roll_Std_3h",
    "Load_Roll_Mean_6h", "Load_Roll_Std_6h",
    "Max_Capacity_MW", "Headroom_MW",
    "Phangan_Load_Lag_1h", "Phangan_Load_Roll_Mean_3h", "Phangan_Load_Roll_Mean_6h",
    "Samui_Load_Lag_1h", "Samui_Load_Roll_Mean_3h", "Samui_Load_Roll_Mean_6h",
]

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
    samui_load_mw: Optional[float] = None
    phangan_load_mw: Optional[float] = None


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
    phangan_forecast: Optional[List[float]] = None
    samui_forecast:   Optional[List[float]] = None


class ClusterDispatchRequest(BaseModel):
    samui_load_mw:   float = Field(..., ge=0.0)
    phangan_load_mw: float = Field(..., ge=0.0)
    tao_load_mw:     float = Field(..., ge=0.0)


class AgentExplainRequest(BaseModel):
    optimized_schedule: dict
    baseline_schedule: dict


class AgentActionPlanRequest(BaseModel):
    incident: dict


class AgentForecastNarrativeRequest(BaseModel):
    forecast_mw: List[float]
    lgbm_features: dict


class AgentExecutiveReportRequest(BaseModel):
    backtest_logs: dict


from api.grid_core import IslandGrid

# ── Streaming state ───────────────────────────────────────────────────────────

class StreamingEngine:
    def __init__(self, window_size: int, db_path: str, cfg: dict):
        self.window_size = window_size
        self.db_path = db_path
        self.cfg = cfg
        self.buffer: deque = deque(maxlen=window_size)
        self._actuals:   List[float] = []
        self._forecasts: List[float] = []
        
        # Initialize physical grid state
        self.grid = IslandGrid("Ko Tao", cfg)
        
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
                    row_obj = TelemetryRow(**dict(zip(_SEQ_FIELD, r)))
                    self.buffer.append(row_obj)
                    # Sync grid state with last known headroom/load
                    # (Note: real state reconstruction would need more history, but this syncs current)
                    self.grid.update(row_obj.island_load_mw, self.cfg["data"]["circuit_cap_max"] - row_obj.headroom_mw)

            metrics = conn.execute("SELECT actual, forecast FROM metrics ORDER BY id ASC").fetchall()
            for a, f in metrics:
                self._actuals.append(a)
                self._forecasts.append(f)
        print(f"   [API] Restored {len(self.buffer)} telemetry rows and {len(self._actuals)} metric pairs.")

    def ingest(self, row: TelemetryRow, circuit_cap_mw: Optional[float] = None):
        self.buffer.append(row)
        
        # Update physical grid simulation state
        # If circuit_cap_mw not provided, estimate from headroom: Cap = Headroom + Load
        cap = circuit_cap_mw if circuit_cap_mw is not None else (row.headroom_mw + row.island_load_mw)
        self.grid.update(row.island_load_mw, cap)

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
        grid_status = self.grid.get_status()
        
        if n == 0:
            return {
                "n": 0, "mae": None, "rmse": None, "mape": None,
                "grid_status": grid_status
            }
        
        a = np.array(self._actuals)
        f = np.array(self._forecasts)
        err = a - f
        mae  = float(np.mean(np.abs(err)))
        rmse = float(math.sqrt(np.mean(err ** 2)))
        mape = float(np.mean(np.abs(err / (a + 1e-8))) * 100)
        
        return {
            "n": n, 
            "mae": round(mae, 4), 
            "rmse": round(rmse, 4), 
            "mape": round(mape, 4),
            "grid_status": grid_status
        }


STREAM = StreamingEngine(tc["window_size"], os.path.join(ROOT, "api_state.db"), CFG)

# ── Helpers ───────────────────────────────────────────────────────────────────

@mlflow.trace(span_type="FUNC", name="lgbm_predict")
def _lgbm_predict(features: dict) -> float:
    """Run LightGBM prediction in a clean subprocess to avoid macOS OpenMP conflicts."""
    script = f"""
import pickle, os, sys, json
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = {repr(ROOT)}
sys.path.insert(0, ROOT)
from models.lgbm_model import FEATURES
with open(os.path.join(ROOT, "models/lgbm.pkl"), "rb") as f:
    model = pickle.load(f)
features = json.loads({repr(json.dumps(features))})
X_row = [features.get(c, 0.0) for c in FEATURES]
X = np.array([X_row])
pred = float(model.predict(X)[0])
print(pred)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, cwd=ROOT, timeout=10
        )
        if result.returncode != 0:
            print(f"⚠️  LGBM subprocess error: {result.stderr}")
            return 0.0
        return float(result.stdout.strip())
    except Exception as e:
        print(f"⚠️  LGBM prediction failed: {e}")
        return 0.0


@mlflow.trace(span_type="FUNC", name="tcn_predict")
def _tcn_predict(history) -> np.ndarray:
    if not _NET: return np.zeros(tc["forecast_horizon"])
    arr = np.array([[getattr(r, f) for f in _SEQ_FIELD] for r in history], dtype=np.float32)
    x = torch.tensor(arr).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return _NET(x).cpu().numpy()[0]


@mlflow.trace(span_type="FUNC", name="hybrid_forecast")
def _hybrid_forecast(history, lgbm_features: dict) -> List[float]:
    if not META: return [0.0] * tc["forecast_horizon"]
    horizon = tc["forecast_horizon"]
    lgbm_pred = _lgbm_predict(lgbm_features)
    tcn_preds = _tcn_predict(history)
    X_meta = np.column_stack([np.full(horizon, lgbm_pred), tcn_preds])
    return META.predict(X_meta).tolist()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="GridTokenX Forecast API", version="2.0.0")


@app.get("/grid/assets")
def get_grid_assets(table: str = Query("egat_power_plants"), limit: int = 100):
    """Retrieve spatial assets from PostGIS as GeoJSON."""
    if not pg_engine:
        raise HTTPException(503, "PostGIS database not available")
    
    # Whitelist allowed tables for security
    allowed = ["egat_power_plants", "egat_substations", "power_plants", "egat_lines", "egat_towers", "koh_samui_grid"]
    if table not in allowed:
        raise HTTPException(400, f"Table '{table}' not accessible or does not exist")

    try:
        with pg_engine.connect() as conn:
            # Query to convert geometry to GeoJSON directly in PostGIS
            # ST_AsGeoJSON returns a string, so we cast to ::jsonb
            query = text(f"""
                SELECT jsonb_build_object(
                    'type',     'FeatureCollection',
                    'features', COALESCE(jsonb_agg(features.feature), '[]'::jsonb)
                )
                FROM (
                  SELECT jsonb_build_object(
                    'type',       'Feature',
                    'id',         id,
                    'geometry',   ST_AsGeoJSON(geom)::jsonb,
                    'properties', to_jsonb(inputs) - 'geom' - 'id'
                  ) AS feature
                  FROM (SELECT * FROM {table} LIMIT :limit) inputs
                ) features;
            """)
            result = conn.execute(query, {"limit": limit}).scalar()
            return result or {"type": "FeatureCollection", "features": []}
    except Exception as e:
        raise HTTPException(500, f"Database error: {str(e)}")


@app.post("/stream/telemetry")
def stream_telemetry(req: TelemetryStreamRequest):
    # Pass current circuit capacity if available in the forecast
    current_cap = req.circuit_forecast[0] if req.circuit_forecast else None
    STREAM.ingest(req.row, circuit_cap_mw=current_cap)
    ready = STREAM.is_ready()
    
    # Automatic Cluster ADMM Check
    cluster_dispatch = None
    if req.samui_load_mw is not None and req.phangan_load_mw is not None:
        try:
            cluster_dispatch = get_cluster_dispatch(
                samui_load=req.samui_load_mw,
                phangan_load=req.phangan_load_mw,
                tao_load=req.row.island_load_mw
            )
        except Exception as e:
            print(f"⚠️  Cluster ADMM failed: {e}")

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
                "cluster_dispatch": cluster_dispatch
            }
        except Exception as e:
            raise HTTPException(500, str(e))
            
    return {
        "status": "ingested", 
        "buffer_size": len(STREAM.buffer), 
        "ready": ready,
        "grid_status": STREAM.live_metrics()["grid_status"],
        "cluster_dispatch": cluster_dispatch
    }


@app.post("/stream/actual")
def stream_actual(req: ActualRequest):
    STREAM.record_actual(req.actual_load_mw, req.forecast_load_mw)
    return {"metrics": STREAM.live_metrics()}


@app.get("/stream/metrics")
def stream_metrics():
    return STREAM.live_metrics()


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "buffer": len(STREAM.buffer), "window": STREAM.window_size}


@app.get("/metrics")
def metrics():
    path = os.path.join(ROOT, "results/evaluation_report.json")
    if not os.path.exists(path): raise HTTPException(404, "Run evaluate.py first.")
    with open(path) as f: return json.load(f)


@app.post("/warnings")
def warnings(req: WarningRequest):
    ph_f = np.array(req.phangan_forecast) if req.phangan_forecast else None
    sa_f = np.array(req.samui_forecast) if req.samui_forecast else None
    
    w = check_warnings(np.array(req.load_forecast), np.array(req.circuit_forecast),
                       req.current_soc, cfg=CFG, lookahead_hours=req.lookahead_hours,
                       phangan_forecast=ph_f, samui_forecast=sa_f)
    return {"count": len(w), "critical": sum(1 for x in w if x.level == "CRITICAL"),
            "warnings": [x.__dict__ for x in w], "summary": format_warnings(w)}


@app.post("/dispatch/cluster")
def dispatch_cluster(req: ClusterDispatchRequest):
    """Run ADMM multi-island diesel coordination."""
    try:
        result = get_cluster_dispatch(
            samui_load=req.samui_load_mw,
            phangan_load=req.phangan_load_mw,
            tao_load=req.tao_load_mw
        )
        return result
    except Exception as e:
        raise HTTPException(500, f"ADMM Dispatch failed: {str(e)}")


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


from agent.use_cases import generate_decision_explanation, generate_action_plan, generate_forecast_narrative, generate_executive_report

@app.post("/agent/explain-dispatch")
def agent_explain_dispatch(req: AgentExplainRequest):
    try:
        explanation = generate_decision_explanation(req.optimized_schedule, req.baseline_schedule)
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(500, f"Gemma explanation failed: {str(e)}")


@app.post("/agent/action-plan")
def agent_action_plan(req: AgentActionPlanRequest):
    try:
        plan = generate_action_plan(req.incident)
        return {"action_plan": plan}
    except Exception as e:
        raise HTTPException(500, f"Gemma action plan failed: {str(e)}")


@app.post("/agent/forecast-narrative")
def agent_forecast_narrative(req: AgentForecastNarrativeRequest):
    try:
        narrative = generate_forecast_narrative(req.forecast_mw, req.lgbm_features)
        return {"narrative": narrative}
    except Exception as e:
        raise HTTPException(500, f"Gemma forecast narrative failed: {str(e)}")


@app.post("/agent/executive-report")
def agent_executive_report(req: AgentExecutiveReportRequest):
    try:
        report = generate_executive_report(req.backtest_logs)
        return {"report": report}
    except Exception as e:
        raise HTTPException(500, f"Gemma executive report failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.serve:app", host="0.0.0.0", port=8000, reload=False)
