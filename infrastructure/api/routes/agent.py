from fastapi import APIRouter, HTTPException
from domain.entities import (
    AgentExplainRequest, AgentActionPlanRequest,
    AgentForecastNarrativeRequest, AgentExecutiveReportRequest,
    AgentGridStatusRequest
)
from agent.use_cases import (
    generate_decision_explanation, generate_action_plan,
    generate_forecast_narrative, generate_executive_report,
    generate_grid_status_explanation
)

router = APIRouter(prefix="/agent", tags=["agent"])

@router.post("/explain-dispatch")
def agent_explain_dispatch(req: AgentExplainRequest):
    try:
        explanation = generate_decision_explanation(req.optimized_schedule, req.baseline_schedule)
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(500, f"Gemma explanation failed: {str(e)}")

@router.post("/action-plan")
def agent_action_plan(req: AgentActionPlanRequest):
    try:
        plan = generate_action_plan(req.incident)
        return {"action_plan": plan}
    except Exception as e:
        raise HTTPException(500, f"Gemma action plan failed: {str(e)}")

@router.post("/forecast-narrative")
def agent_forecast_narrative(req: AgentForecastNarrativeRequest):
    try:
        narrative = generate_forecast_narrative(req.forecast_mw, req.lgbm_features)
        return {"narrative": narrative}
    except Exception as e:
        raise HTTPException(500, f"Gemma forecast narrative failed: {str(e)}")

@router.post("/executive-report")
def agent_executive_report(req: AgentExecutiveReportRequest):
    try:
        report = generate_executive_report(req.backtest_logs)
        return {"report": report}
    except Exception as e:
        raise HTTPException(500, f"Gemma executive report failed: {str(e)}")

@router.post("/grid-status")
def agent_grid_status(req: AgentGridStatusRequest):
    try:
        explanation = generate_grid_status_explanation(req.grid_status)
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(500, f"Gemma grid status failed: {str(e)}")
