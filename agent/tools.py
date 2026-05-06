import json
import os
from typing import Dict, Any

ROOT = os.path.dirname(os.path.dirname(__file__))

def sop_tool(query: str = "") -> str:
    """
    Retrieves standard operating procedures relevant to the given query.
    In a real system, this would be a RAG / Vector DB call.
    For this prototype, it returns the contents of the mock SOPs.
    """
    sop_path = os.path.join(ROOT, "data", "sops", "mock_sops.json")
    if not os.path.exists(sop_path):
        return "No SOPs found."
    
    with open(sop_path, "r") as f:
        sops = json.load(f)
        
    # Return all SOPs as text (could be filtered by query)
    output = []
    for doc_id, doc_content in sops.items():
        output.append(f"Document ID: {doc_id} - {doc_content.get('title', '')}")
        for k, v in doc_content.items():
            if k != "title":
                output.append(f"  {k}: {v}")
    
    return "\n".join(output)

def forecast_tool() -> str:
    """Mock forecast tool returning standard load forecasts."""
    return "Forecast implies 15% increase in peak load due to weather."

def dispatch_tool() -> str:
    """Mock dispatch tool representing the MILP scheduler."""
    return "MILP scheduler recommends charging BESS to 90% and maintaining DG-1 offline."
