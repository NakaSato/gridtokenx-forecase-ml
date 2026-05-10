from fastapi import APIRouter, Depends, HTTPException
from domain.entities import ForecastRequest, WarningRequest
from infrastructure.api.dependencies import get_predictor_dep, get_config_dep
from domain.dispatch import run_dispatch, schedule_summary
from optimizer.early_warning import check_warnings, format_warnings
import numpy as np

router = APIRouter(tags=["forecast"])

@router.post("/forecast")
def forecast(
    req: ForecastRequest,
    predictor=Depends(get_predictor_dep),
    cfg=Depends(get_config_dep)
):
    if len(req.history) != predictor.window_size:
        raise HTTPException(422, f"history must have {predictor.window_size} rows")
    try:
        fc = predictor.predict(req.history, req.lgbm_features)
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    schedule = run_dispatch(np.array(fc), np.array(req.circuit_forecast),
                            initial_soc=req.initial_soc, cfg=cfg)
    return {"forecast_mw": fc, "summary": schedule_summary(schedule)}

@router.post("/warnings")
def warnings(req: WarningRequest, cfg=Depends(get_config_dep)):
    ph_f = np.array(req.phangan_forecast) if req.phangan_forecast else None
    sa_f = np.array(req.samui_forecast) if req.samui_forecast else None
    
    w = check_warnings(np.array(req.load_forecast), np.array(req.circuit_forecast),
                       req.current_soc, cfg=cfg, lookahead_hours=req.lookahead_hours,
                       phangan_forecast=ph_f, samui_forecast=sa_f)
    return {"count": len(w), "critical": sum(1 for x in w if x.level == "CRITICAL"),
            "warnings": [x.__dict__ for x in w], "summary": format_warnings(w)}
