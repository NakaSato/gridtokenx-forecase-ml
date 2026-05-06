import requests

API = "http://localhost:8000/stream/telemetry"

def test_auto_admm():
    # Scenario: Tao load 12 MW + Phangan 45 MW + Samui 125 MW -> Total 182 MW
    # Should trigger ADMM dispatch because Total > 160.8 MW
    
    payload = {
        "row": {
            "island_load_mw": 12.0,
            "load_lag_1h": 11.5,
            "load_lag_24h": 10.0,
            "bess_soc_pct": 80.0,
            "headroom_mw": 0.0,
            "dry_bulb_temp": 28.5,
            "heat_index": 32.0,
            "rel_humidity": 80.0,
            "hour_of_day": 19.0,
            "is_high_season": 1.0,
            "is_thai_holiday": 0.0
        },
        "samui_load_mw": 125.0,
        "phangan_load_mw": 45.0
    }

    print("--- Testing Automated ADMM in Telemetry Stream ---")
    try:
        resp = requests.post(API, json=payload)
        resp.raise_for_status()
        res = resp.json()
        
        cd = res.get("cluster_dispatch")
        if cd and cd.get("deficit_mw", 0) > 0:
            print(f"✅ Auto-ADMM Triggered! Deficit: {cd['deficit_mw']} MW")
            for agent in cd['agents']:
                print(f"  {agent['name']}: {agent['diesel_output_mw']} MW")
        else:
            print("❌ Auto-ADMM did not return dispatch data.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_auto_admm()
