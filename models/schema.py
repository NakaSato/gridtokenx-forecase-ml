"""
Centralized schema definition for GridTokenX models.
Ensures consistency between synthetic generation, preprocessing, training, and API serving.
"""

# Targets (in order)
TARGETS = [
    "tao_load_mw", 
    "cable_flow_mw", 
    "kmb_flow_mw"
]

# Sequential Features for TCN (Order matters)
SEQ_FEATURES = [
    # 1. Per-location load + weather
    "phangan_load_mw", "samui_load_mw", "phangan_t2m", "samui_t2m",
    "t2m_celsius", "rh_pct", "ghi_w_m2", "wind_ms",

    # 2. System state
    "headroom_mw", "max_capacity_mw", "capacity_mw", "tourist_index",
    "bess_soc_pct", "phangan_soc_pct", "samui_soc_pct",

    # 3. Calendar + Market
    "hour_of_day", "day_of_week", "is_holiday", "is_songkran", "is_high_season",
    "carbon_intensity", "market_price",

    # 4. Critical Lags
    "tao_load_mw_lag_1h", "cable_flow_mw_lag_1h", "kmb_flow_mw_lag_1h",
    "tao_load_mw_lag_24h", "cable_flow_mw_lag_24h", "kmb_flow_mw_lag_24h",
]

# Tabular Features for LightGBM
TAB_FEATURES = [
    # 1. Per-location load + weather
    "phangan_load_mw", "samui_load_mw", "phangan_t2m", "samui_t2m", 
    "t2m_celsius", "rh_pct", "ghi_w_m2", "wind_ms",
    
    # 2. System state
    "headroom_mw", "max_capacity_mw", "capacity_mw", 
    "bess_soc_pct", "phangan_soc_pct", "samui_soc_pct", "tourist_index",
    
    # 3. Calendar + Market
    "hour_of_day", "day_of_week", "is_holiday", "is_songkran", "is_high_season",
    "carbon_intensity", "market_price",
    
    # 4. Pre-engineered (Lags, Rolls, Decomposition)
    "tao_load_mw_lag_1h", "tao_load_mw_lag_24h",
    "cable_flow_mw_lag_1h", "cable_flow_mw_lag_24h",
    "kmb_flow_mw_lag_1h", "kmb_flow_mw_lag_24h",
    "tao_load_roll_mean_3h", "tao_load_roll_std_3h",
    "tao_load_roll_mean_6h", "tao_load_roll_std_6h",
    "heat_index", "temp_gradient",
    "kmb_trend", "kmb_seasonal", "kmb_resid"
]
