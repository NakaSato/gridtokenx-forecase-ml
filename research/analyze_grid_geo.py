import os
import json
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# Constants for analysis
KO_TAO_CENTER = Point(99.84, 10.08)
SEARCH_RADIUS_DEG = 1.0  # Approx 110km

# Thailand uses UTM Zone 47N (EPSG:32647) for accurate distance measurements
UTM_THAILAND = "EPSG:32647"

def analyze_grid():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║             GridTokenX — Island Cluster Spatial Analysis             ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # 1. Line Length Analysis
    print("\n[1] TRANSMISSION LINE ANALYSIS")
    line_files = [
        "data/geojson/egat_lines.geojson",
        "data/geojson/koh_samui_grid_infrastructure.geojson",
        "frontend/public/ko_tao_network.geojson"
    ]
    
    total_km = 0
    all_lines = []
    
    for f in line_files:
        if not os.path.exists(f): continue
        gdf = gpd.read_file(f)
        # Filter for LineStrings only
        lines = gdf[gdf.geometry.type.isin(['LineString', 'MultiLineString'])]
        if lines.empty: continue
        
        # Ensure CRS is WGS84 for distance check
        if lines.crs is None: lines.set_crs(epsg=4326, inplace=True)
        
        # Filter lines within radius of Ko Tao (rough degree filter)
        lines = lines[lines.geometry.distance(KO_TAO_CENTER) < SEARCH_RADIUS_DEG]
        
        # Calculate length in KM using UTM projection
        lines_utm = lines.to_crs(UTM_THAILAND)
        lines['length_km'] = lines_utm.geometry.length / 1000.0
        
        for _, row in lines.iterrows():
            name = row.get('name') or row.get('line_name') or row.get('name_e') or "Unnamed"
            voltage = row.get('voltage_kv') or row.get('voltage') or "Unknown"
            all_lines.append({
                "source": os.path.basename(f),
                "name": name,
                "voltage": voltage,
                "length_km": round(row['length_km'], 2)
            })
            total_km += row['length_km']

    df_lines = pd.DataFrame(all_lines)
    if not df_lines.empty:
        print(df_lines.sort_values("length_km", ascending=False).to_string(index=False))
        print(f"\nTotal Line Network analyzed: {total_km:.2f} km")
    else:
        print("No transmission lines found in proximity.")

    # 2. Substation Proximity Analysis
    print("\n[2] NEAREST SUBSTATIONS TO KO TAO")
    sub_file = "data/geojson/egat_substations.geojson"
    if os.path.exists(sub_file):
        subs = gpd.read_file(sub_file)
        if subs.crs is None: subs.set_crs(epsg=4326, inplace=True)
        
        # Calculate distance to Ko Tao
        subs_utm = subs.to_crs(UTM_THAILAND)
        center_utm = gpd.GeoSeries([KO_TAO_CENTER], crs="EPSG:4326").to_crs(UTM_THAILAND).iloc[0]
        subs['dist_km'] = subs_utm.geometry.distance(center_utm) / 1000.0
        
        nearest = subs.nsmallest(5, 'dist_km')[['name_e', 'voltage230', 'voltage115', 'dist_km']]
        print(nearest.to_string(index=False))
    else:
        print("Substation file not found.")

    # 3. Power Plant Capacity Analysis
    print("\n[3] REGIONAL POWER GENERATION")
    plant_file = "data/geojson/power_plants/thailand_generators.geojson"
    if os.path.exists(plant_file):
        plants = gpd.read_file(plant_file)
        if plants.crs is None: plants.set_crs(epsg=4326, inplace=True)
        
        # Filter nearby plants (within 150km)
        plants_utm = plants.to_crs(UTM_THAILAND)
        plants['dist_km'] = plants_utm.geometry.distance(center_utm) / 1000.0
        nearby_plants = plants[plants['dist_km'] < 150].nsmallest(10, 'dist_km')
        
        # Clean up names and show relevant columns
        cols = ['Plant / Project name', 'Capacity (MW)', 'Type', 'dist_km']
        existing_cols = [c for c in cols if c in plants.columns]
        print(nearby_plants[existing_cols].to_string(index=False))
    else:
        print("Power plant file not found.")

if __name__ == "__main__":
    analyze_grid()
