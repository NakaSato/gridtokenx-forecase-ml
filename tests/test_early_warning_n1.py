import requests
import numpy as np

API = "http://localhost:8000/warnings"

def test_n1_warning():
    # Normal load: Tao=7, Phangan=25, Samui=90 -> Total=122 (No warning)
    # High load: Tao=10, Phangan=35, Samui=110 -> Total=155 (Alert > 145)
    # Critical load: Tao=12, Phangan=45, Samui=125 -> Total=182 (Critical > 160.8)
    
    payloads = [
        {
            "name": "Normal Scenario",
            "data": {
                "load_forecast": [7.0] * 6,
                "circuit_forecast": [13.3] * 6,
                "current_soc": 0.8,
                "lookahead_hours": 6,
                "phangan_forecast": [25.0] * 6,
                "samui_forecast": [90.0] * 6
            }
        },
        {
            "name": "N-1 Alert Scenario",
            "data": {
                "load_forecast": [10.0] * 6,
                "circuit_forecast": [13.3] * 6,
                "current_soc": 0.8,
                "lookahead_hours": 6,
                "phangan_forecast": [35.0] * 6,
                "samui_forecast": [110.0] * 6
            }
        },
        {
            "name": "N-1 Critical Scenario",
            "data": {
                "load_forecast": [12.0] * 6,
                "circuit_forecast": [13.3] * 6,
                "current_soc": 0.8,
                "lookahead_hours": 6,
                "phangan_forecast": [45.0] * 6,
                "samui_forecast": [125.0] * 6
            }
        }
    ]

    for p in payloads:
        print(f"\n--- Testing: {p['name']} ---")
        try:
            resp = requests.post(API, json=p['data'])
            resp.raise_for_status()
            res = resp.json()
            print(f"Count: {res['count']}")
            print(res['summary'])
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_n1_warning()
