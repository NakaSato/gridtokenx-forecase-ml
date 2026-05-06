"""
Generate synthetic hourly dataset for Ko Tao-Phangan-Samui cluster.

Per-island load profiles (physics-calibrated):
  Ko Tao    — stable, AC-dominated, flat diurnal (~6.7 MW base, low variance)
  Ko Phangan — moderate tourism volatility, stronger diurnal swing (~18 MW base)
  Ko Samui   — high tourism + commercial load, strong diurnal + event spikes (~55 MW base)
"""
import os
import numpy as np
import pandas as pd
import yaml


def load_cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def _ar1(n, phi, sigma, rng):
    """AR(1) noise series."""
    noise = np.zeros(n)
    noise[0] = rng.normal(0, sigma)
    for i in range(1, n):
        noise[i] = phi * noise[i - 1] + rng.normal(0, sigma)
    return noise


def _bess_soc(circuit_cap, island_load, bc, n):
    soc = np.zeros(n)
    soc[0] = 0.65
    cap, charge_rate = bc["capacity_mwh"], bc["charge_rate_mw"]
    if cap <= 0:
        return soc
    for i in range(1, n):
        delta = circuit_cap[i] - island_load[i]
        if delta > 0:
            soc[i] = min(bc["soc_max"], soc[i - 1] + min(delta, charge_rate) / cap)
        else:
            soc[i] = max(bc["soc_min"], soc[i - 1] + delta / cap)
    return soc


def generate_ko_tao(idx, dc, bc, rng, assets, global_sim):
    """
    Ko Tao: stable, AC-dominated, nearly flat diurnal.
    """
    n = len(idx)
    a = assets["ko_tao"]
    hours = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    months = idx.month.to_numpy()
    is_high = np.isin(months, dc["high_season_months"]).astype(float)

    temp = (
        dc.get("temp_base", 27.1)
        + dc.get("temp_seasonal_amp", 1.1) * np.sin(2 * np.pi * (months - 3) / 12)
        + dc.get("temp_diurnal_amp", 0.5) * np.sin(2 * np.pi * (hours - 14) / 24)
        + rng.normal(0, 0.3, n)
    ).clip(dc.get("temp_min", 25.8), dc.get("temp_max", 28.9))

    humidity = (
        dc.get("humidity_base", 78.5)
        - dc.get("humidity_diurnal_amp", 3.0) * np.sin(2 * np.pi * (hours - 6) / 24)
        + rng.normal(0, 1.5, n)
    ).clip(dc.get("humidity_min", 74.0), dc.get("humidity_max", 84.0))

    # noise params from config
    phi = global_sim["noise"]["ar1_phi"]
    sigma = global_sim["noise"]["ar1_sigma"]

    base_load = a["load_base_mw"] + is_high * a["load_seasonal_shift_mw"]
    ac_load = np.maximum(0, (temp - dc["ac_threshold_temp"]) * a["load_ac_coeff"])
    diurnal = a["load_diurnal_amp"] * np.sin(np.pi * (hours - 10) / 14).clip(0, None)
    load = (base_load + ac_load + diurnal + _ar1(n, phi, sigma, rng)).clip(
        dc["load_base_min"], dc["load_base_max"]
    )

    # Phangan -> Tao 33kV XLPE link: Physical absolute max is 16.0 MW
    max_physical_limit = 16.0
    circuit_cap = np.full(n, dc.get("circuit_cap_max", 13.3) * 0.65)
    btl = np.isin(idx.hour, dc["bottleneck_hours"]) & (rng.random(n) < (a["bottleneck_prob"] / 4))
    circuit_cap[btl] = rng.uniform(0.5, 4.9, btl.sum())
    circuit_cap = (circuit_cap + rng.normal(0, 0.3, n)).clip(0, max_physical_limit)

    return load, temp, humidity, circuit_cap, _bess_soc(circuit_cap, load, bc, n)


def generate_ko_phangan(idx, dc, bc, rng, assets, global_sim):
    """
    Ko Phangan: moderate tourism volatility.
    """
    n = len(idx)
    a = assets["ko_phangan"]
    hours = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    months = idx.month.to_numpy()
    is_high = np.isin(months, dc["high_season_months"]).astype(float)

    temp = (
        dc.get("temp_base", 27.1)
        + dc.get("temp_seasonal_amp", 1.1) * np.sin(2 * np.pi * (months - 3) / 12)
        + dc.get("temp_diurnal_amp", 0.5) * np.sin(2 * np.pi * (hours - 14) / 24)
        + rng.normal(0, 0.4, n)
    ).clip(dc.get("temp_min", 25.8), dc.get("temp_max", 28.9))

    humidity = (
        dc.get("humidity_base", 78.5)
        - 4.0 * np.sin(2 * np.pi * (hours - 6) / 24)
        + rng.normal(0, 2.0, n)
    ).clip(70.0, 90.0)

    phi = global_sim["noise"]["ar1_phi"]
    sigma = global_sim["noise"]["ar1_sigma"]

    base_load = a["load_base_mw"] + is_high * a["load_seasonal_shift_mw"]
    ac_load = np.maximum(0, (temp - dc["ac_threshold_temp"]) * a["load_ac_coeff"])
    diurnal = a["load_diurnal_amp"] * np.sin(np.pi * (hours - 8) / 14).clip(0, None)

    is_full_moon_night = (rng.random(n) < (1 / (720*4))) & np.isin(idx.hour, [22, 23, 0, 1, 2])
    event_spike = np.where(is_full_moon_night, rng.uniform(2.0, 5.0, n), 0.0)

    load = (base_load + ac_load + diurnal + event_spike + _ar1(n, phi, sigma, rng)).clip(
        12.0, 30.0
    )

    # Samui -> Phangan link: 1x 115kV XLPE + 1x 33kV XLPE
    circuit_cap = np.full(n, 35.0)  # Combined normal capacity
    btl = np.isin(idx.hour, dc["bottleneck_hours"]) & (rng.random(n) < (a["bottleneck_prob"] / 4))
    circuit_cap[btl] = rng.uniform(5.0, 15.0, btl.sum())
    circuit_cap = (circuit_cap + rng.normal(0, 0.5, n)).clip(0, 45.0)

    return load, temp, humidity, circuit_cap, _bess_soc(circuit_cap, load, bc, n)


def generate_ko_samui(idx, dc, bc, rng, assets, global_sim):
    """
    Ko Samui: high volatility.
    """
    n = len(idx)
    a = assets["ko_samui"]
    hours = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    months = idx.month.to_numpy()
    is_high = np.isin(months, dc["high_season_months"]).astype(float)

    temp = (
        dc.get("temp_base", 27.1)
        + dc.get("temp_seasonal_amp", 1.1) * np.sin(2 * np.pi * (months - 3) / 12)
        + 0.8 * np.sin(2 * np.pi * (hours - 14) / 24)
        + rng.normal(0, 0.5, n)
    ).clip(25.0, 34.0)

    humidity = (
        dc.get("humidity_base", 78.5)
        - 5.0 * np.sin(2 * np.pi * (hours - 6) / 24)
        + rng.normal(0, 2.5, n)
    ).clip(65.0, 92.0)

    phi = global_sim["noise"]["ar1_phi"]
    sigma = global_sim["noise"]["ar1_sigma"]

    base_load = a["load_base_mw"] + is_high * a["load_seasonal_shift_mw"]
    ac_load = np.maximum(0, (temp - dc["ac_threshold_temp"]) * a["load_ac_coeff"])
    daytime_peak = a["load_diurnal_amp"] * np.sin(np.pi * (hours - 8) / 12).clip(0, None)
    night_entertainment = 3.0 * np.where(np.isin(idx.hour, [20, 21, 22, 23, 0]), 1.0, 0.0)

    is_spike = (
        (np.isin(months, [4, 12]) & (rng.random(n) < (0.05/4))) |
        (is_high.astype(bool) & (rng.random(n) < (0.03/4)))
    )
    event_spike = np.where(is_spike, rng.uniform(5.0, 15.0, n), 0.0)

    load = (
        base_load + ac_load + daytime_peak + night_entertainment
        + event_spike + _ar1(n, phi, sigma, rng)
    ).clip(35.0, 95.0)

    # Khanom -> Samui link: 2x 115kV + 2x 33kV
    circuit_cap = np.full(n, 110.0) # Normal operation
    # Simulate bottleneck (e.g., loss of one 115kV circuit)
    btl = np.isin(idx.hour, dc["bottleneck_hours"]) & (rng.random(n) < (a["bottleneck_prob"]/4))
    circuit_cap[btl] = rng.uniform(30.0, 60.0, btl.sum())
    faults = rng.random(n) < (0.005/4) # Extreme fault (both 115kV down)
    circuit_cap[faults] = rng.uniform(10.0, 20.0, faults.sum())
    circuit_cap = (circuit_cap + rng.normal(0, 1.0, n)).clip(0, 150.0)

    return load, temp, humidity, circuit_cap, _bess_soc(circuit_cap, load, bc, n)


def generate(cfg):
    dc = cfg["data"]
    bc = cfg["bess"]
    cluster = cfg["cluster"]
    assets = cluster["assets"]
    sim = cluster["simulation"]
    rng = np.random.default_rng(42)

    # Capturing 2024-2026 commissioning and modern growth
    idx = pd.date_range(dc["start_date"], dc["end_date"], freq=dc.get("frequency", "15min"))
    n = len(idx)
    
    # ── Thai Holidays (Songkran Stress Test) ──────────────────────────────────
    holiday_dates = []
    for h in dc["holidays"].values():
        holiday_dates.extend(h)
    
    md = idx.strftime("%m-%d")
    is_holiday = np.isin(md, holiday_dates)
    is_songkran = np.isin(md, dc["holidays"]["songkran"])
    
    # Shared exogenous features
    hours = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    irradiance = np.where(
        (hours >= 6) & (hours <= 18),
        (1050 * np.sin(np.pi * (hours - 6) / 12)).clip(0, 1050) + rng.normal(0, 30, n),
        0.0
    ).clip(0, 1050)

    carbon_intensity = (
        dc.get("carbon_intensity_mean", 440.0)
        + dc.get("carbon_intensity_std", 18.0) * rng.standard_normal(n)
    ).clip(dc.get("carbon_intensity_min", 400.0), dc.get("carbon_intensity_max", 484.0))

    market_price = (
        dc.get("market_price_mean", 70.0)
        + dc.get("market_price_std", 15.0) * rng.standard_normal(n)
    ).clip(dc.get("market_price_min", 35.0), dc.get("market_price_max", 120.0))

    months = idx.month.to_numpy()
    is_high = np.isin(months, dc["high_season_months"]).astype(float)
    tourist_index = (0.6 + 0.3 * is_high + 0.05 * rng.standard_normal(n)).clip(0.2, 1.0)

    # Per-island generation
    tao_load,     tao_temp,     tao_hum,     tao_cap,     tao_soc     = generate_ko_tao(idx, dc, bc, rng, assets, sim)
    phangan_load, phangan_temp, phangan_hum, phangan_cap, phangan_soc = generate_ko_phangan(idx, dc, bc, rng, assets, sim)
    samui_load,   samui_temp,   samui_hum,   samui_cap,   samui_soc   = generate_ko_samui(idx, dc, bc, rng, assets, sim)

    # ── Holiday Spikes ──
    # Songkran causes dramatic shifts on tourist islands
    hf = sim["holiday_factors"]
    holiday_factor = np.where(is_songkran, hf["songkran"], 1.0)
    holiday_factor = np.where(is_holiday & ~is_songkran, hf["default"], holiday_factor)

    # ── Modern Grid Trends (2024-2026) ────────────────────────────────────────
    years = idx.year.to_numpy()
    tr = sim["trends"]
    ev_factor = np.where(years >= 2024, 1.0 + (years - 2023) * tr["ev_annual_growth"], 1.0)
    is_daytime = (idx.hour >= 9) & (idx.hour <= 16)
    solar_factor = np.where((years >= 2024) & is_daytime, 1.0 - (years - 2023) * tr["solar_annual_offset"], 1.0)
    
    final_factor = ev_factor * solar_factor * holiday_factor
    
    tao_load     = tao_load * final_factor
    phangan_load = phangan_load * final_factor
    samui_load   = samui_load * final_factor

    # ── 115 kV NO.3 Cable Stability ──
    cable_thermal_limit = 1.0 + 0.05 * np.sin(2 * np.pi * (idx.month - 3) / 12)
    tao_cap = tao_cap * cable_thermal_limit

    # Primary output: Ko Tao (model training target)
    df = pd.DataFrame({
        "Island_Load_MW":    tao_load.round(3),
        "Dry_Bulb_Temp":     tao_temp.round(2),
        "Rel_Humidity":      tao_hum.round(1),
        "Solar_Irradiance":  irradiance.round(1),
        "Carbon_Intensity":  carbon_intensity.round(1),
        "Market_Price":      market_price.round(2),
        "Tourist_Index":     tourist_index.round(3),
        "BESS_SoC_Pct":      (tao_soc * 100).round(1),
        "Circuit_Cap_MW":    tao_cap.round(3),
        "Is_Thai_Holiday":   is_holiday.astype(int),
        "Is_Songkran":       is_songkran.astype(int),
        # Cluster columns
        "Phangan_Load_MW":   phangan_load.round(3),
        "Phangan_Temp":      phangan_temp.round(2),
        "Phangan_Circuit_MW": phangan_cap.round(2),
        "Phangan_SoC_Pct":   (phangan_soc * 100).round(1),
        "Samui_Load_MW":     samui_load.round(3),
        "Samui_Temp":        samui_temp.round(2),
        "Samui_Circuit_MW":  samui_cap.round(2),
        "Samui_SoC_Pct":     (samui_soc * 100).round(1),
    }, index=idx)
    df.index.name = "Timestamp"
    return df


def main():
    cfg = load_cfg()
    out = cfg["data"]["output_path"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df = generate(cfg)
    df.to_parquet(out)
    print(f"Saved {len(df):,} rows → {out}")
    print("\n=== Ko Tao (stable) ===")
    print(df[["Island_Load_MW", "Dry_Bulb_Temp", "Rel_Humidity"]].describe().round(2))
    print("\n=== Ko Phangan (moderate volatility) ===")
    print(df[["Phangan_Load_MW", "Phangan_Circuit_MW"]].describe().round(2))
    print("\n=== Ko Samui (high volatility) ===")
    print(df[["Samui_Load_MW", "Samui_Circuit_MW"]].describe().round(2))


if __name__ == "__main__":
    main()
