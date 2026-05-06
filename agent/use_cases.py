from agent.gemma_client import gemma_client
from agent.tools import sop_tool
import json

def generate_decision_explanation(optimized_schedule: dict, baseline_schedule: dict) -> str:
    """
    Acts as the Counterfactual Narrator (Use Case #1).
    Compares the optimized schedule with the baseline and generates a Thai/English narrative.
    """
    system_prompt = (
        "You are Gemma 4, an AI operator assistant for the Ko Tao power grid. "
        "Your job is to compare an optimized dispatch schedule against a baseline "
        "and concisely explain the fuel savings in liters and the operational reasons. "
        "Use Thai language for the operator."
    )
    
    prompt = f"""
    Compare these two schedules:
    
    Optimized: {json.dumps(optimized_schedule)}
    Baseline: {json.dumps(baseline_schedule)}
    
    Calculate the difference in diesel usage (assume some standard conversion if missing) 
    and output a short, actionable summary of why the optimized schedule is better.
    Highlight specific hours where BESS is used instead of Diesel, or where ramping is reduced.
    """
    
    return gemma_client.generate(prompt=prompt, system_instruction=system_prompt)


def generate_action_plan(incident: dict) -> str:
    """
    Acts as the Action Plan Agent (Use Case #2).
    Reads an incident dict (from the Early Warning system) and retrieves SOPs to generate a 3-step action plan.
    """
    # Tool call: Retrieve SOP context
    sop_context = sop_tool(query=incident.get("message", ""))
    
    system_prompt = (
        "You are Gemma 4, an AI operator assistant for the Ko Tao power grid. "
        "An early warning incident has been detected. Using the provided SOP context, "
        "generate a strict 3-step actionable plan for the operator. "
        "Translate technical jargon into actionable steps and cite the relevant SOP reference. "
        "Format clearly with bold headers and bullet points."
    )
    
    prompt = f"""
    INCIDENT DETAILS:
    {json.dumps(incident, indent=2)}
    
    RELEVANT SOP CONTEXT:
    {sop_context}
    
    Generate the 3-step Action Plan now. Include what happens if ignored.
    """
    
    return gemma_client.generate(prompt=prompt, system_instruction=system_prompt)


def generate_forecast_narrative(forecast_mw: list, lgbm_features: dict) -> str:
    """
    Acts as the Forecast Explainer (Requirement D2).
    Generates a narrative explaining the 24h load forecast, especially focusing on load spikes.
    """
    system_prompt = (
        "You are Gemma 4, an AI operator assistant for the Ko Tao power grid. "
        "Analyze the provided 24-hour load forecast alongside the current external features (weather, calendar, holidays). "
        "Provide a concise narrative explaining the forecast curve, specifically highlighting why the load might be spiking "
        "(e.g., 'Due to Songkran festival and high heat index, we expect a peak load of X MW...'). "
        "Keep the explanation brief, operator-friendly, and output in Thai."
    )
    
    prompt = f"""
    FORECAST (Next 24h MW):
    {forecast_mw}
    
    FEATURES (Weather, Calendar):
    {json.dumps(lgbm_features, indent=2)}
    
    Please provide the narrative:
    """
    
    return gemma_client.generate(prompt=prompt, system_instruction=system_prompt)


def generate_executive_report(backtest_logs: dict) -> str:
    """
    Acts as the Auto Financial Reporter (Requirement V - Viability).
    Reads 12-month MILP backtest logs and generates an Executive Summary for proposal inclusion.
    """
    system_prompt = (
        "You are Gemma 4, an Executive Reporting AI for the GridTokenX predictive intelligence layer. "
        "Your task is to analyze 12 months of backtest logs and generate a professional, defense-ready "
        "Executive Summary (in markdown format, suitable for PDF/DOCX conversion). "
        "The report must highlight total diesel savings (liters and THB), CO2 emissions avoided, "
        "BESS utilization gains, and estimated ROI. Keep the tone professional, persuasive, and data-driven."
    )
    
    prompt = f"""
    BACKTEST LOGS SUMMARY:
    {json.dumps(backtest_logs, indent=2)}
    
    Please generate the Executive Summary Report. Structure it with clear headings:
    1. Executive Overview
    2. Financial Impact (Savings in THB/Year)
    3. Environmental Impact (CO2 avoided)
    4. Operational Efficiency (BESS & Dispatch)
    5. Deployment Timeline & ROI
    """
    
    return gemma_client.generate(prompt=prompt, system_instruction=system_prompt)
