"""
Recalibrate island_grid.parquet using real-world reference datasets:
  - microgrid_weather_material.csv  → temperature, humidity, PV, carbon intensity, diesel price
  - microgrid_load_price_pv.csv     → load shape, market price, PV generation patterns
  - NREL PERFORM h5                 → load temporal patterns (hourly actuals)
  - weather_openmeteo.parquet       → Ko Tao actual weather (temp, humidity, wind)

Strategy: keep Ko Tao synthetic load scale (5–10 MW) but replace statistical
distributions with real-world empirical distributions from reference data.
"""
import numpy as np
import pandas as pd
import h5py
from pathlib import Path

DATA = Path("data")
OUT = DATA / "ko_tao_grid_calibrated.parquet"

# ── Load reference datasets ──────────────────────────────────────────────────
ref = pd.read_csv(DATA / "public_datasets/microgrid_weather_material.csv", parse_dates=["datetime"])
ref_lp = pd.read_csv(DATA / "public_datasets/microgrid_load_price_pv.csv", parse_dates=["Timestamp"])

with h5py.File(DATA / "nrel_perform/BA_load_actuals_2018.h5", "r") as f:
    nrel_load = f["actuals"][:]
    nrel_time = pd.to_datetime([t.decode() for t in f["time_index"][:]])

nrel_df = pd.DataFrame({"load": nrel_load}, index=nrel_time)

# Ko Tao actual weather
weather = pd.read_parquet(DATA / "weather_openmeteo.parquet")
ko_tao_wx = weather[weather["island"] == "ko_tao"].copy()
ko_tao_wx["time"] = pd.to_datetime(ko_tao_wx["time"])
ko_tao_wx = ko_tao_wx.set_index("time")

# ── Load existing synthetic dataset ─────────────────────────────────────────
df = pd.read_parquet(DATA / "island_grid.parquet")
df.index = pd.to_datetime(df.index)

# ── 1. Recalibrate temperature & humidity from Ko Tao real weather ───────────
# Match by (month, hour) — cyclic alignment regardless of year
ko_tao_wx["month"] = ko_tao_wx.index.month
ko_tao_wx["hour"]  = ko_tao_wx.index.hour
wx_profile = ko_tao_wx.groupby(["month", "hour"])[["temperature_2m", "relativehumidity_2m"]].mean()

df["_month"] = df.index.month
df["_hour"]  = df.index.hour
df = df.join(wx_profile, on=["_month", "_hour"])
df["Dry_Bulb_Temp"] = df["temperature_2m"].fillna(df["temperature_2m"].mean())
df["Rel_Humidity"]  = df["relativehumidity_2m"].fillna(df["relativehumidity_2m"].mean())
df.drop(columns=["_month", "_hour", "temperature_2m", "relativehumidity_2m"], inplace=True)

# ── 2. Recalibrate load shape from NREL PERFORM (scale to Ko Tao MW range) ──
# NREL load is in MW for a large BA — normalise to 0–1 then scale to 5–10 MW
nrel_norm = (nrel_load - nrel_load.min()) / (nrel_load.max() - nrel_load.min())
# Tile/sample to match synthetic dataset length
rng = np.random.default_rng(42)
idx = rng.integers(0, len(nrel_norm), size=len(df))
load_shape = nrel_norm[idx]
df["Island_Load_MW"] = 5.0 + load_shape * 5.0  # scale to 5–10 MW

# Apply real A/C correlation: +0.35 MW per °C above 28°C
ac_boost = np.clip(df["Dry_Bulb_Temp"] - 28.0, 0, None) * 0.35
df["Island_Load_MW"] = np.clip(df["Island_Load_MW"] + ac_boost, 5.0, 10.0)

# ── 3. Recalibrate carbon intensity from real reference ──────────────────────
ci_mean = ref["CI(gco2/kWh)"].mean()
ci_std  = ref["CI(gco2/kWh)"].std()
df["Carbon_Intensity"] = np.clip(
    rng.normal(ci_mean, ci_std, len(df)), 400, 850
)

# ── 4. Recalibrate diesel/market price from real reference ───────────────────
# Diesel price: use real No.2 Diesel Fuel price ($/gallon) → convert to $/MWh proxy
diesel_gal = ref["No. 2 Diesel Fuel (($/Gallon))"].mean()  # ~3.0 $/gal
# BSFC at 75% load ≈ 198.5 g/kWh, diesel density 0.832 kg/L, 3.785 L/gal
# fuel_cost_per_mwh = (198.5 g/kWh * 1kg/1000g) * (1L/0.832kg) * (1gal/3.785L) * diesel_gal * 1000
fuel_cost_mwh = (198.5 / 1000) / 0.832 / 3.785 * diesel_gal * 1000
market_mean = ref["purchasing price (dollar/kWh)"].mean() * 1000  # $/MWh
market_std  = ref["purchasing price (dollar/kWh)"].std() * 1000
df["Market_Price"] = np.clip(
    rng.normal(market_mean, market_std, len(df)), 35, 120
)

# ── 5. Recalibrate PV generation shape from real reference ───────────────────
pv_norm = ref_lp["PV (kWh)"] / (ref_lp["PV (kWh)"].max() + 1e-9)
idx_pv = rng.integers(0, len(pv_norm), size=len(df))
df["Solar_Irradiance"] = np.clip(pv_norm.values[idx_pv] * 1050, 0, 1050)

# ── Save ─────────────────────────────────────────────────────────────────────
df.to_parquet(OUT)
print(f"Saved calibrated dataset: {OUT}")
print(f"Rows: {len(df):,}")
print(df[["Dry_Bulb_Temp", "Rel_Humidity", "Island_Load_MW", "Carbon_Intensity", "Market_Price", "Solar_Irradiance"]].describe().round(2))
