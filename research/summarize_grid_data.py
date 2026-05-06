import json

files = [
    "data/geojson/egat_substations.geojson",
    "data/geojson/egat_lines.geojson",
    "frontend/public/ko_tao_network.geojson",
    "data/geojson/koh_samui_grid_infrastructure.geojson"
]

def summarize_geojson(path):
    print(f"\n--- Summary for: {path} ---")
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        features = data.get('features', [])
        print(f"Total features: {len(features)}")
        
        # Filter for island related names or locations
        island_keywords = ["SAMUI", "PHANGAN", "TAO", "KHANOM"]
        
        relevant_features = []
        for feat in features:
            props = feat.get('properties', {})
            # Check if any property value matches keywords
            if any(key in str(val).upper() for val in props.values() for key in island_keywords):
                relevant_features.append(props)
        
        print(f"Relevant features found: {len(relevant_features)}")
        for props in relevant_features[:10]: # Show first 10
            print(f"  - {props}")
            
    except Exception as e:
        print(f"Error reading {path}: {e}")

for f in files:
    summarize_geojson(f)
