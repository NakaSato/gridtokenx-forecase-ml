import pandas as pd
import numpy as np
import yaml
import os

def integrate():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    dc = cfg["data"]
    freq = dc.get("frequency", "h")
    sph = 4 if freq == "15min" else 1

    print(f"--- Starting Raw Data Integration ({freq}) ---")
    
    # 1. Process Weather (ERA5)
    print("Loading raw weather data...")
    weather_df = pd.read_parquet("data/raw/raw_weather_thailand.parquet")
    weather_df = weather_df[
        (weather_df['latitude'] >= 9.9) & (weather_df['latitude'] <= 10.3) & 
        (weather_df['longitude'] >= 98.0) & (weather_df['longitude'] <= 100.0)
    ]
    weather_hourly = weather_df.groupby('datetime')['2m_temperatures_celcius'].mean().reset_index()
    weather_hourly.rename(columns={'datetime': 'Timestamp', '2m_temperatures_celcius': 'Dry_Bulb_Temp'}, inplace=True)
    weather_hourly['Timestamp'] = pd.to_datetime(weather_hourly['Timestamp'])
    weather_hourly.set_index('Timestamp', inplace=True)

    # 2. Process Tourism & Extend Timeline
    print("Loading raw tourism data...")
    tourism_raw = pd.read_csv("data/raw/raw_tourism_samui.csv")
    # Extend timeline to May 2026
    timeline_full = pd.date_range("2023-01-01", "2026-05-02 00:00:00", freq=freq)
    tourism_final = pd.DataFrame(index=timeline_full)
    
    max_pax = tourism_raw['Passengers'].max()
    min_pax = tourism_raw['Passengers'].min()
    tourism_raw['Index'] = 0.4 + 0.6 * (tourism_raw['Passengers'] - min_pax) / (max_pax - min_pax)
    month_map = tourism_raw[tourism_raw['Year'] == 2023].set_index('Month')['Index'].to_dict()
    
    tourism_final['Month'] = tourism_final.index.month_name()
    tourism_final['Tourist_Index'] = tourism_final['Month'].map(month_map)
    tourism_final.drop(columns=['Month'], inplace=True)

    # 3. Merge and Re-generate Grid Physics
    print("Merging datasets...")
    # Upsample weather and interpolate for gaps in 2024-2026
    weather_final = weather_hourly.resample(freq).interpolate(method="linear")
    weather_extended = weather_final.reindex(timeline_full).interpolate(method="linear")
    
    master_full = tourism_final.join(weather_extended, how='inner')
    master_full = master_full.interpolate(method='linear')
    
    rng = np.random.default_rng(42)
    n = len(master_full)
    
    hours_arr = master_full.index.hour.to_numpy() + master_full.index.minute.to_numpy() / 60.0
    months_arr = master_full.index.month.to_numpy()
    is_high = np.isin(months_arr, dc["high_season_months"]).astype(float)
    temp_arr = master_full['Dry_Bulb_Temp'].values

    # Recalibrate physics for frequency
    phi = 0.78 if freq == "15min" else 0.85
    sigma = 0.12 if freq == "15min" else 0.15 
    
    base_load = 6.5 + is_high * dc["high_season_load_shift"]
    ac_load = np.maximum(0, (temp_arr - dc["ac_threshold_temp"]) * dc["ac_coefficient"])
    diurnal = 1.5 * np.sin(np.pi * (hours_arr - 8) / 14).clip(0, None)
    
    ar_noise = np.zeros(n)
    ar_noise[0] = rng.normal(0, 0.3)
    for i in range(1, n):
        ar_noise[i] = phi * ar_noise[i-1] + rng.normal(0, sigma)

    # ── Modern Grid Trends (EV + Solar) ──
    years = master_full.index.year.to_numpy()
    ev_factor = np.where(years >= 2024, 1.0 + (years - 2023) * 0.025, 1.0)
    is_daytime = (master_full.index.hour >= 9) & (master_full.index.hour <= 16)
    solar_factor = np.where((years >= 2024) & is_daytime, 1.0 - (years - 2023) * 0.015, 1.0)

    # Final Island Load
    master_full['Island_Load_MW'] = (base_load + ac_load + diurnal + ar_noise) * ev_factor * solar_factor
    master_full['Island_Load_MW'] = master_full['Island_Load_MW'].clip(5, 12)
    
    # Circuit Cap
    circuit_cap = np.full(n, 8.0)
    bottleneck_events = (np.isin(master_full.index.hour, dc["bottleneck_hours"])) & (rng.random(n) < (0.3 / sph))
    circuit_cap[bottleneck_events] = rng.uniform(0.5, 4.9, bottleneck_events.sum())
    master_full['Circuit_Cap_MW'] = circuit_cap
    
    # Remaining features
    master_full['Rel_Humidity'] = 75 + 10 * np.sin(np.pi * (hours_arr - 6) / 12) + rng.normal(0, 2, n)
    master_full['Solar_Irradiance'] = np.maximum(0, 1000 * np.sin(np.pi * (hours_arr - 6) / 12)) * (master_full.index.month.isin([2,3,4,5]).astype(int) * 0.2 + 0.8)
    
    master_full['Carbon_Intensity'] = rng.uniform(400, 700, n)
    master_full['Market_Price'] = rng.uniform(50, 100, n)
    master_full['BESS_SoC_Pct'] = 50.0 
    
    master_full['Hour_of_Day'] = master_full.index.hour
    master_full['Day_of_Week'] = master_full.index.dayofweek
    master_full['Is_High_Season'] = is_high
    
    # Cleanup and Save
    master_full.index.name = "Timestamp"
    
    print("\n--- Timestamp Integrity Check ---")
    print(f"Start: {master_full.index.min()}")
    print(f"End:   {master_full.index.max()}")
    print(f"Total Steps: {len(master_full)}")
    
    master_full.to_parquet("data/processed/ko_tao_grid_2023_locked.parquet")
    print("\nCompleted: data/processed/ko_tao_grid_2023_locked.parquet")

if __name__ == "__main__":
    integrate()
