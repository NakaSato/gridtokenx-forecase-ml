import sys
import os

# Add root to python path so we can import app
sys.path.insert(0, "/Users/chanthawat/Developments/gridtokenx-forecase-ml")

from fastapi.testclient import TestClient
from api.serve import app

client = TestClient(app)

def test_explain_dispatch():
    print("Testing /agent/explain-dispatch...")
    res = client.post("/agent/explain-dispatch", json={
        "optimized_schedule": {"00:00": "DG-1 5MW"},
        "baseline_schedule": {"00:00": "DG-1 6MW"}
    })
    print(res.status_code)
    print(res.json())

def test_action_plan():
    print("\nTesting /agent/action-plan...")
    res = client.post("/agent/action-plan", json={
        "incident": {
            "type": "PV_ramp_down_predicted",
            "message": "PV dropping by 3.5 MW",
            "severity": 0.78
        }
    })
    print(res.status_code)
    print(res.json())

if __name__ == "__main__":
    test_explain_dispatch()
    test_action_plan()
