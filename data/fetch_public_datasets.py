"""
Fetch public weather + solar irradiance data for Ko Tao, Phangan, Samui.
Sources: Open-Meteo (weather) + NASA POWER (solar)
"""
import requests
import pandas as pd
from pathlib import Path

OUT = Path(__file__).parent
START, END = "2020-01-01", "2023-12-31"

ISLANDS = {
    "ko_tao":    {"lat": 10.0956, "lon": 99.8397},
    "ko_phangan": {"lat": 9.7333,  "lon": 100.0167},
    "ko_samui":  {"lat": 9.5333,  "lon": 100.0667},
}


def fetch_openmeteo(name: str, lat: float, lon: float) -> pd.DataFrame:
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": lat, "longitude": lon,
            "start_date": START, "end_date": END,
            "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m,precipitation",
            "timezone": "Asia/Bangkok",
        },
        timeout=60,
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df.insert(0, "island", name)
    return df


def fetch_nasa_power(name: str, lat: float, lon: float) -> pd.DataFrame:
    r = requests.get(
        "https://power.larc.nasa.gov/api/temporal/hourly/point",
        params={
            "parameters": "ALLSKY_SFC_SW_DWN,T2M,RH2M",
            "community": "RE",
            "longitude": lon, "latitude": lat,
            "start": START.replace("-", ""), "end": END.replace("-", ""),
            "format": "JSON",
            "time-standard": "LST",
        },
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()["properties"]["parameter"]
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index, format="%Y%m%d%H")
    df.index.name = "time"
    df = df.reset_index()
    df.insert(0, "island", name)
    return df


if __name__ == "__main__":
    om_frames, nasa_frames = [], []

    for name, coords in ISLANDS.items():
        print(f"[Open-Meteo] {name}...")
        om_frames.append(fetch_openmeteo(name, **coords))

        print(f"[NASA POWER] {name}...")
        nasa_frames.append(fetch_nasa_power(name, **coords))

    om_df = pd.concat(om_frames, ignore_index=True)
    nasa_df = pd.concat(nasa_frames, ignore_index=True)

    om_path = OUT / "weather_openmeteo.parquet"
    nasa_path = OUT / "solar_nasa_power.parquet"

    om_df.to_parquet(om_path, index=False)
    nasa_df.to_parquet(nasa_path, index=False)

    print(f"\nSaved: {om_path}  ({len(om_df):,} rows)")
    print(f"Saved: {nasa_path}  ({len(nasa_df):,} rows)")
