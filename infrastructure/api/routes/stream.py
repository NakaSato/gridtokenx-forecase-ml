from fastapi import APIRouter, Depends, HTTPException
from domain.entities import TelemetryStreamRequest, ActualRequest
from infrastructure.api.dependencies import get_streaming_engine_dep, get_predictor_dep, get_config_dep
from domain.dispatch import run_dispatch, schedule_summary
from optimizer.early_warning import check_warnings, format_warnings
from optimizer.cluster_dispatch_admm import get_cluster_dispatch
import numpy as np

router = APIRouter(prefix="/stream", tags=["streaming"])

@router.post("/telemetry")
def stream_telemetry(
    req: TelemetryStreamRequest,
    stream=Depends(get_streaming_engine_dep),
    predictor=Depends(get_predictor_dep),
    cfg=Depends(get_config_dep)
):
    current_cap = req.circuit_forecast[0] if req.circuit_forecast else None
    stream.ingest(req.row, circuit_cap_mw=current_cap)
    ready = stream.is_ready()
    print(f"   [STREAM] Ingested step. Buffer: {len(stream.buffer)}/{stream.window_size}, Ready: {ready}, HasFeatures: {req.lgbm_features is not None}")
    
    cluster_dispatch = None
    if req.samui_load_mw is not None and req.phangan_load_mw is not None:
        try:
            cluster_dispatch = get_cluster_dispatch(
                samui_load=req.samui_load_mw,
                phangan_load=req.phangan_load_mw,
                tao_load=req.row.tao_load_mw
            )
        except Exception as e:
            print(f"⚠️  Cluster ADMM failed: {e}")

    if ready and req.lgbm_features and req.circuit_forecast:
        try:
            forecast = predictor.predict(list(stream.buffer), req.lgbm_features)
            
            # ── Early Warning Check ──
            warnings = check_warnings(
                load_forecast=np.array(forecast),
                circuit_forecast=np.array(req.circuit_forecast),
                current_soc=req.row.bess_soc_pct / 100.0,
                cfg=cfg,
                phangan_forecast=None,
                samui_forecast=None
            )
            warning_summary = format_warnings(warnings)

            schedule = run_dispatch(np.array(forecast), np.array(req.circuit_forecast),
                                    initial_soc=req.row.bess_soc_pct / 100.0, cfg=cfg)
            
            return {
                "status": "forecast",
                "buffer_size": len(stream.buffer),
                "forecast_mw": forecast,
                "summary": warning_summary,
                "schedule": schedule_summary(schedule),
                "live_metrics": stream.live_metrics(),
                "cluster_dispatch": cluster_dispatch
            }
        except Exception as e:
            print(f"❌ Forecast cycle failed: {e}")

    return {
        "status": "ingested", 
        "buffer_size": len(stream.buffer), 
        "ready": ready,
        "grid_status": stream.live_metrics()["grid_status"],
        "cluster_dispatch": cluster_dispatch
    }

@router.post("/actual")
def stream_actual(req: ActualRequest, stream=Depends(get_streaming_engine_dep)):
    stream.record_actual(req.actual_load_mw, req.forecast_load_mw)
    return {"metrics": stream.live_metrics()}

@router.get("/metrics")
def stream_metrics(stream=Depends(get_streaming_engine_dep)):
    return stream.live_metrics()
