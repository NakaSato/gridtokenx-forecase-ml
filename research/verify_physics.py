import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# Constants for analysis
KO_TAO_CENTER = Point(99.84, 10.08)
SEARCH_RADIUS_DEG = 1.2
UTM_THAILAND = "EPSG:32647"

# Electrical Parameters for 115 kV XLPE Submarine Cable (approximate)
# Based on typical 630mm2 Cu conductor
R_OHM_PER_KM = 0.05  # Resistance
X_OHM_PER_KM = 0.12  # Reactance
V_NOMINAL_KV = 115.0

# Load Scenarios (MW)
PEAK_LOAD_SAMUI_MW = 95.0
PEAK_LOAD_PHANGAN_MW = 26.0
PEAK_LOAD_TAO_MW = 7.7

def estimate_losses():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║          GridTokenX — Electrical Loss & Impedance Analysis           ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    line_files = [
        "data/geojson/egat_lines.geojson",
        "data/geojson/koh_samui_grid_infrastructure.geojson",
        "frontend/public/ko_tao_network.geojson"
    ]
    
    all_lines = []
    
    for f in line_files:
        if not os.path.exists(f): continue
        gdf = gpd.read_file(f)
        lines = gdf[gdf.geometry.type.isin(['LineString', 'MultiLineString'])]
        if lines.empty: continue
        if lines.crs is None: lines.set_crs(epsg=4326, inplace=True)
        
        # Re-project for distance
        lines_utm = lines.to_crs(UTM_THAILAND)
        lines['length_km'] = lines_utm.geometry.length / 1000.0
        
        for _, row in lines.iterrows():
            name = row.get('name') or row.get('line_name') or row.get('name_e') or "Unnamed"
            voltage = str(row.get('voltage_kv') or row.get('voltage') or "115")
            
            # Skip very short segments (noise)
            if row['length_km'] < 0.1: continue
            
            # Map voltage string to float
            v_kv = 115.0
            if "230" in voltage: v_kv = 230.0
            elif "22" in voltage and "115" not in voltage: v_kv = 22.0
            
            # Calculate Impedance
            r_total = R_OHM_PER_KM * row['length_km']
            
            # Loss estimation: P_loss = 3 * I^2 * R
            # Assume 50 MW throughput for transmission lines as a baseline
            p_flow_mw = 50.0
            if "Samui" in str(name): p_flow_mw = PEAK_LOAD_SAMUI_MW
            if "Tao" in str(name): p_flow_mw = PEAK_LOAD_TAO_MW
            
            # I = P / (sqrt(3) * V * pf) -- assume power factor 0.95
            i_amps = (p_flow_mw * 1e6) / (1.732 * v_kv * 1e3 * 0.95)
            p_loss_kw = (3 * (i_amps**2) * r_total) / 1000.0
            
            all_lines.append({
                "Link": name[:30],
                "V (kV)": v_kv,
                "Length (km)": round(row['length_km'], 2),
                "R (Ω)": round(r_total, 3),
                "Est. Loss (kW)": round(p_loss_kw, 2),
                "% Loss": round((p_loss_kw / (p_flow_mw * 1000)) * 100, 3) if p_flow_mw > 0 else 0
            })

    df = pd.DataFrame(all_lines).drop_duplicates(subset=["Link", "Length (km)"])
    print(df.sort_values("Est. Loss (kW)", ascending=False).to_string(index=False))
    
    total_loss = df["Est. Loss (kW)"].sum()
    print(f"\nTotal estimated cluster transmission loss: {total_loss/1000:.3f} MW")
    print("\nNote: Losses are estimated at peak load assuming 115kV XLPE cable specs.")

if __name__ == "__main__":
    estimate_losses()
