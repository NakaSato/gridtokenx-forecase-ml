from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from dataclasses import dataclass

# ── Base Physical State ───────────────────────────────────────────────────────

class BESSState(BaseModel):
    soc_mwh: float
    soc_pct: float
    is_charging: bool
    power_mw: float
    temp_c: float = 25.0

@dataclass
class HourlyDispatch:
    hour: int
    load_mw: float
    circuit_mw: float
    diesel_mw: float
    bess_mw: float        # positive = discharge, negative = charge
    bess_soc: float       # 0–1
    fuel_kg: float
    carbon_kg: float

# ── Telemetry & Streaming ─────────────────────────────────────────────────────

class TelemetryRow(BaseModel):
    """One hour of real-time telemetry — matches SEQ_FEATURES exactly."""
    # 1. Per-location load + weather
    phangan_load_mw:  float
    samui_load_mw:    float
    phangan_t2m:      float
    samui_t2m:        float
    t2m_celsius:      float
    rh_pct:           float
    ghi_w_m2:         float
    wind_ms:          float
    
    # 2. System state
    headroom_mw:      float
    max_capacity_mw:  float
    capacity_mw:      float
    bess_soc_pct:     float
    phangan_soc_pct:  float
    samui_soc_pct:    float
    tourist_index:    float
    
    # 3. Calendar + Market
    hour_of_day:      float
    day_of_week:      float
    is_holiday:       float
    is_songkran:      float
    is_high_season:   float
    carbon_intensity: float
    market_price:     float
    
    # 4. Critical Lags
    tao_load_mw:      float
    tao_load_mw_lag_1h: float
    cable_flow_mw_lag_1h: float
    kmb_flow_mw_lag_1h: float
    kmb_trend:        float
    kmb_seasonal:     float
    kmb_resid:        float

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

# ── Forecast & Dispatch Requests ──────────────────────────────────────────────

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

# ── Agent / Explainability Requests ───────────────────────────────────────────

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

class AgentGridStatusRequest(BaseModel):
    grid_status: dict

@dataclass
class GridWarning:
    level: str          # "CRITICAL" | "WARNING" | "INFO"
    hour:  int          # hours from now when event occurs
    message: str
    soc_at_event: float
    hvdc_loading_pct: Optional[float] = None   # pandapower result, None if not run
