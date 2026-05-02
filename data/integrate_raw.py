import pandas as pd
import numpy as np
import yaml
import os

def integrate():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    print("--- Starting Raw Data Integration ---")
    
    # 1. Process Weather (ERA5)
    # Ko Tao approx: Lat 10.1, Lon 99.8
    print("Loading raw weather data...")
    weather_df = pd.read_parquet("data/raw_weather_thailand.parquet")
    
    # Find closest coordinates in dataset
    # Based on data check: Lat ~9.9-10.3, Lon ~98.0-100.0 contains relevant points
    weather_df = weather_df[
        (weather_df['latitude'] >= 9.9) & (weather_df['latitude'] <= 10.3) & 
        (weather_df['longitude'] >= 98.0) & (weather_df['longitude'] <= 100.0)
    ]
    
    # Group by datetime to get average across the island area
    weather_hourly = weather_df.groupby('datetime')['2m_temperatures_celcius'].mean().reset_index()
    weather_hourly.rename(columns={'datetime': 'Timestamp', '2m_temperatures_celcius': 'Dry_Bulb_Temp'}, inplace=True)
    weather_hourly['Timestamp'] = pd.to_datetime(weather_hourly['Timestamp'])
    weather_hourly.set_index('Timestamp', inplace=True)
    
    print(f"Extracted {len(weather_hourly)} weather hours. Range: {weather_hourly.index.min()} to {weather_hourly.index.max()}")

    # 2. Process Tourism (USM Arrivals)
    print("Loading raw tourism data...")
    tourism_raw = pd.read_csv("data/raw_tourism_samui.csv")
    
    # Create hourly timeline for 2023
    timeline_2023 = pd.date_range("2023-01-01", "2023-12-31 23:00:00", freq="H")
    tourism_hourly = pd.DataFrame(index=timeline_2023)
    
    # Map monthly passengers to a normalized 0.4 - 1.0 index
    max_pax = tourism_raw['Passengers'].max()
    min_pax = tourism_raw['Passengers'].min()
    tourism_raw['Index'] = 0.4 + 0.6 * (tourism_raw['Passengers'] - min_pax) / (max_pax - min_pax)
    
    # Join monthly index to hourly timeline
    month_map = tourism_raw[tourism_raw['Year'] == 2023].set_index('Month')['Index'].to_dict()
    # Month names to numbers map
    import calendar
    month_to_num = {name: num for num, name in enumerate(calendar.month_name) if num}
    
    tourism_hourly['Month'] = tourism_hourly.index.month_name()
    tourism_hourly['Tourist_Index'] = tourism_hourly['Month'].map(month_map)
    tourism_hourly.drop(columns=['Month'], inplace=True)

    # 3. Merge and Re-generate Grid Physics for 2023
    print("Merging datasets...")
    # Align weather and tourism on 2023 timeline
    master_2023 = tourism_hourly.join(weather_hourly, how='inner')
    
    # Fill any gaps
    master_2023 = master_2023.interpolate(method='linear')
    
    # Apply Physics (Simulating 2023 based on real weather/tourism)
    rng = np.random.default_rng(42)
    n = len(master_2023)
    
    # Base load with AR(1) noise for temporal coherence (phi=0.85)
    hours_arr = master_2023.index.hour.to_numpy()
    months_arr = master_2023.index.month.to_numpy()
    is_high = np.isin(months_arr, cfg["data"]["high_season_months"]).astype(float)
    temp_arr = master_2023['Dry_Bulb_Temp'].values

    base_load = 6.5 + is_high * cfg["data"]["high_season_load_shift"]
    ac_load = np.maximum(0, (temp_arr - cfg["data"]["ac_threshold_temp"]) * cfg["data"]["ac_coefficient"])
    diurnal = 1.5 * np.sin(np.pi * (hours_arr - 8) / 14).clip(0, None)
    ar_noise = np.zeros(n)
    ar_noise[0] = rng.normal(0, 0.3)
    for i in range(1, n):
        ar_noise[i] = 0.85 * ar_noise[i-1] + rng.normal(0, 0.15)

    # A/C Load from real temp
    ac_load = np.maximum(0, (master_2023['Dry_Bulb_Temp'] - cfg["data"]["ac_threshold_temp"]) * cfg["data"]["ac_coefficient"])

    # Final Island Load
    master_2023['Island_Load_MW'] = (base_load + ac_load + diurnal + ar_noise).clip(5, 10)
    
    # Circuit Cap (Stochastic bottlenecks)
    circuit_cap = np.full(n, 8.0)
    hours = master_2023.index.hour
    bottleneck_events = (np.isin(hours, cfg["data"]["bottleneck_hours"])) & (rng.random(n) < 0.3)
    circuit_cap[bottleneck_events] = rng.uniform(0.5, 4.9, bottleneck_events.sum())
    master_2023['Circuit_Cap_MW'] = circuit_cap
    
    # Remaining features (Synthetic but anchored)
    master_2023['Rel_Humidity'] = 75 + 10 * np.sin(np.pi * (hours - 6) / 12) + rng.normal(0, 2, n)
    master_2023['Solar_Irradiance'] = np.maximum(0, 1000 * np.sin(np.pi * (hours - 6) / 12)) * (master_2023.index.month.isin([2,3,4,5]).astype(int) * 0.2 + 0.8)
    
    # Missing columns for model compatibility
    master_2023['Wind_Speed'] = rng.uniform(2.0, 6.0, n)
    master_2023['Cloud_Cover'] = rng.uniform(0, 100, n)
    master_2023['Carbon_Intensity'] = rng.uniform(400, 700, n)
    master_2023['Market_Price'] = rng.uniform(50, 100, n)
    master_2023['BESS_SoC_Pct'] = 50.0 # Initial placeholder
    
    master_2023['Net_Delta_MW'] = master_2023['Island_Load_MW'] - master_2023['Circuit_Cap_MW']
    
    # Cleanup and Save
    master_2023.index.name = "Timestamp"
    
    # Validation
    print("\n--- Timestamp Integrity Check ---")
    print(f"Start: {master_2023.index.min()}")
    print(f"End:   {master_2023.index.max()}")
    print(f"Total Hours: {len(master_2023)}")
    
    master_2023.to_parquet("data/ko_tao_grid_2023_locked.parquet")
    print("\nCompleted: data/ko_tao_grid_2023_locked.parquet")

if __name__ == "__main__":
    integrate()
