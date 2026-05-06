import requests

API = "http://localhost:8000/dispatch/cluster"

def test_cluster_dispatch():
    payloads = [
        {
            "name": "Normal (No Diesel)",
            "data": {
                "samui_load_mw": 90.0,
                "phangan_load_mw": 25.0,
                "tao_load_mw": 7.0
            }
        },
        {
            "name": "N-1 Alert (Samui picks up load)",
            "data": {
                "samui_load_mw": 120.0,
                "phangan_load_mw": 35.0,
                "tao_load_mw": 10.0
            }
        },
        {
            "name": "Severe Overload (Samui + Phangan)",
            "data": {
                "samui_load_mw": 130.0,
                "phangan_load_mw": 40.0,
                "tao_load_mw": 12.0
            }
        }
    ]

    for p in payloads:
        print(f"\n--- Testing API: {p['name']} ---")
        try:
            resp = requests.post(API, json=p['data'])
            resp.raise_for_status()
            res = resp.json()
            print(f"Deficit: {res['deficit_mw']} MW")
            print(f"Total Dispatch: {res['total_dispatch_mw']} MW")
            for agent in res.get('agents', []):
                print(f"  {agent['name']}: {agent['diesel_output_mw']} MW")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_cluster_dispatch()
