"""
Generate synthetic 2-year hourly dataset for Ko Tao island microgrid.
Physics rules from config.yaml.
"""
import os
import numpy as np
import pandas as pd
import yaml

def load_cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def generate(cfg):
    dc = cfg["data"]
    bc = cfg["bess"]
    rng = np.random.default_rng(42)

    idx = pd.date_range(dc["start_date"], dc["end_date"], freq="h")
    n = len(idx)
    hours = idx.hour.to_numpy()
    months = idx.month.to_numpy()
    is_high = np.isin(months, dc["high_season_months"]).astype(float)

    # Temperature: seasonal + diurnal
    temp = (
        29.0
        + 3.0 * np.sin(2 * np.pi * (months - 3) / 12)   # seasonal
        + 2.5 * np.sin(2 * np.pi * (hours - 6) / 24)     # diurnal
        + rng.normal(0, 0.5, n)
    ).clip(24, 36)

    humidity = (
        78.0
        - 8.0 * np.sin(2 * np.pi * (hours - 6) / 24)
        + rng.normal(0, 2, n)
    ).clip(60, 95)

    irradiance = np.where(
        (hours >= 6) & (hours <= 18),
        (1050 * np.sin(np.pi * (hours - 6) / 12)).clip(0, 1050) + rng.normal(0, 30, n),
        0.0
    ).clip(0, 1050)

    carbon_intensity = (600 + 150 * rng.standard_normal(n)).clip(400, 850)
    market_price = (70 + 25 * rng.standard_normal(n)).clip(35, 120)
    tourist_index = (0.6 + 0.3 * is_high + 0.05 * rng.standard_normal(n)).clip(0.2, 1.0)

    # Island load — AR(1) noise for temporal coherence
    base_load = 6.5 + is_high * dc["high_season_load_shift"]
    ac_load = np.maximum(0, (temp - dc["ac_threshold_temp"]) * dc["ac_coefficient"])
    diurnal = 1.5 * np.sin(np.pi * (hours - 8) / 14).clip(0, None)
    # AR(1) noise: phi=0.85 gives strong autocorrelation (realistic load)
    ar_noise = np.zeros(n)
    ar_noise[0] = rng.normal(0, 0.3)
    for i in range(1, n):
        ar_noise[i] = 0.85 * ar_noise[i-1] + rng.normal(0, 0.15)
    island_load = (base_load + ac_load + diurnal + ar_noise).clip(
        dc["load_base_min"], dc["load_base_max"]
    )

    # Circuit capacity with bottleneck events
    circuit_cap = np.full(n, 8.0)  # Base 8 MW (closer to load)
    bottleneck_mask = np.isin(hours, dc["bottleneck_hours"])
    bottleneck_events = bottleneck_mask & (rng.random(n) < 0.30) # 30% prob
    circuit_cap[bottleneck_events] = rng.uniform(0.5, 4.9, bottleneck_events.sum())
    circuit_cap += rng.normal(0, 0.3, n)
    circuit_cap = circuit_cap.clip(0, dc["circuit_cap_max"])

    # BESS SoC simulation
    soc = np.zeros(n)
    soc[0] = 0.65
    cap = bc["capacity_mwh"]
    charge_rate = bc["charge_rate_mw"]
    for i in range(1, n):
        delta = circuit_cap[i] - island_load[i]
        if delta > 0:
            soc[i] = min(bc["soc_max"], soc[i-1] + min(delta, charge_rate) / cap)
        else:
            soc[i] = max(bc["soc_min"], soc[i-1] + delta / cap)

    df = pd.DataFrame({
        "Timestamp": idx,
        "Island_Load_MW": island_load.round(3),
        "Dry_Bulb_Temp": temp.round(2),
        "Rel_Humidity": humidity.round(1),
        "Solar_Irradiance": irradiance.round(1),
        "Carbon_Intensity": carbon_intensity.round(1),
        "Market_Price": market_price.round(2),
        "Tourist_Index": tourist_index.round(3),
        "BESS_SoC_Pct": (soc * 100).round(1),
    })
    df.set_index("Timestamp", inplace=True)
    return df

def main():
    cfg = load_cfg()
    os.makedirs("data", exist_ok=True)
    df = generate(cfg)
    out = cfg["data"]["output_path"]
    df.to_parquet(out)
    print(f"Saved {len(df):,} rows → {out}")
    print(df.describe().round(2))

if __name__ == "__main__":
    main()
